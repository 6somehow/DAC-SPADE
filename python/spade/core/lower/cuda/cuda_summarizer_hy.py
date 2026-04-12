from spade_hy import summarize_forward
import torch
from typing import Optional, Dict, Any

from spade.core.lower.executor import SummarizerExecutor
from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.utils.reorder import ReorderTensorTextBSNH


class CudaHYSummarizerExecutor(SummarizerExecutor):
    """
    A Cuda-based summarizer executor that is optimized for specific summarization modes
    by calling a custom CUDA kernel.
    Supported modes:
    - inter-summary: 'max' and 'min'
    - intra-summary: 'MeanSim'
    """

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        super().__init__(sparse_config, layout=layout)
        # This executor is specialized for certain summarization modes.
        supported_inter = ({'Max': 'max', 'Min': 'min'}, )
        supported_intra = ('MeanSim', None)

        if self.sparse_config.inter_summarize_enable:
            if self.sparse_config.q_inter_summarizer_mode not in supported_inter or \
               self.sparse_config.k_inter_summarizer_mode not in supported_inter:
                raise ValueError(
                    f"CudaHYSummarizerExecutor only supports inter summarizer mode {supported_inter}"
                )

        if self.sparse_config.intra_summarize_enable:
            if self.sparse_config.q_intra_summarizer_mode not in supported_intra or \
               self.sparse_config.k_intra_summarizer_mode not in supported_intra:
                raise ValueError(
                    f"CudaHYSummarizerExecutor only supports intra summarizer mode '{supported_intra}'"
                )

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 head_ids: torch.Tensor, block_q: torch.Tensor,
                 block_k: torch.Tensor, block_v: torch.Tensor,
                 realSeqlen: Optional[int]) -> Dict[str, Any]:

        if self.layout == "bsnh":
            reorder_tensor_text = ReorderTensorTextBSNH
        elif self.layout == "bnsh":
            raise ValueError("CudaHYSummarizerExecutor only supports bsnh layout")
        else:
            raise ValueError(f"Unsupported layout '{self.layout}'")

        config = self.sparse_config
        frame_dim, height_dim, width_dim = config.seqlen3d
        context_len = config.context_length

        # For Q
        frame_block_size_q, height_block_size_q, width_block_size_q = config.block_size_q
        num_context_block_q = config.context_q_num_block

        # For K, V
        frame_block_size_kv, height_block_size_kv, width_block_size_kv = config.block_size_kv
        num_context_block_kv = config.context_kv_num_block

        bsz, _, _, hidden_dim = q.shape
        num_head_ids = head_ids.shape[0]

        ceil_div = lambda a, b: (a + b - 1) // b if b > 0 else 0

        # Calculate total blocks for Q
        num_frame_block_q = ceil_div(frame_dim, frame_block_size_q)
        num_height_block_q = ceil_div(height_dim, height_block_size_q)
        num_width_block_q = ceil_div(width_dim, width_block_size_q)
        num_video_blocks_q = num_frame_block_q * num_height_block_q * num_width_block_q
        total_num_blocks_q = num_video_blocks_q + num_context_block_q

        # Calculate total blocks for K/V
        num_frame_block_kv = ceil_div(frame_dim, frame_block_size_kv)
        num_height_block_kv = ceil_div(height_dim, height_block_size_kv)
        num_width_block_kv = ceil_div(width_dim, width_block_size_kv)
        num_video_blocks_kv = num_frame_block_kv * num_height_block_kv * num_width_block_kv
        total_num_blocks_kv = num_video_blocks_kv + num_context_block_kv

        res: Dict[str, Any] = {
            'block_inter_q': None,
            'block_inter_k': None,
            'block_intra_q': None,
            'block_intra_k': None,
        }

        # The CUDA kernel requires buffers for summary results. We create them here.
        # If dynamic sparsity is disabled, these will be dummy buffers that are not used.
        # For Q
        block_max_q = torch.empty(bsz,
                                  num_head_ids,
                                  total_num_blocks_q,
                                  hidden_dim,
                                  device=q.device,
                                  dtype=q.dtype)
        block_min_q = torch.empty_like(block_max_q)
        block_cos_sim_q = torch.empty(bsz,
                                      num_head_ids,
                                      total_num_blocks_q,
                                      device=q.device,
                                      dtype=config.attn_dtype)

        # For K
        block_max_k = torch.empty(bsz,
                                  num_head_ids,
                                  total_num_blocks_kv,
                                  hidden_dim,
                                  device=k.device,
                                  dtype=k.dtype)
        block_min_k = torch.empty_like(block_max_k)
        block_cos_sim_k = torch.empty(bsz,
                                      num_head_ids,
                                      total_num_blocks_kv,
                                      device=k.device,
                                      dtype=config.attn_dtype)

        # Process Q
        summarize_forward(head_indices=head_ids,
                          x=q,
                          reorder_x=block_q,
                          block_max=block_max_q,
                          block_min=block_min_q,
                          block_cos_sim=block_cos_sim_q,
                          real_seqlen=realSeqlen,
                          frame_dim=frame_dim,
                          height_dim=height_dim,
                          width_dim=width_dim,
                          context_len=context_len,
                          frame_block_size=frame_block_size_q,
                          height_block_size=height_block_size_q,
                          width_block_size=width_block_size_q,
                          num_context_block=num_context_block_q)
        res['block_inter_q'] = {'Max': block_max_q, 'Min': block_min_q}
        res['block_intra_q'] = block_cos_sim_q if self.sparse_config.q_intra_summarizer_mode == 'MeanSim' else None

        res['block_inter_k'] = {'Max': block_max_k, 'Min': block_min_k}
        res['block_intra_k'] = block_cos_sim_k if self.sparse_config.k_intra_summarizer_mode == 'MeanSim' else None

        # Process K
        summarize_forward(head_indices=head_ids,
                          x=k,
                          reorder_x=block_k,
                          block_max=block_max_k,
                          block_min=block_min_k,
                          block_cos_sim=block_cos_sim_k,
                          real_seqlen=realSeqlen,
                          frame_dim=frame_dim,
                          height_dim=height_dim,
                          width_dim=width_dim,
                          context_len=context_len,
                          frame_block_size=frame_block_size_kv,
                          height_block_size=height_block_size_kv,
                          width_block_size=width_block_size_kv,
                          num_context_block=num_context_block_kv)

        reorder_tensor_text(
            block_v, v, head_ids, 0, (frame_dim, height_dim, width_dim),
            (frame_block_size_kv, height_block_size_kv, width_block_size_kv),
            (num_frame_block_kv, num_height_block_kv, num_width_block_kv),
            realSeqlen)

        return res
