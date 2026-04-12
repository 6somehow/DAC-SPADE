import torch
from typing import List, Optional, Union, Dict, Any
from math import prod
from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.executor import SummarizerExecutor, EstimatorExecutor, AttentionExecutor, E2EExecutor
from spade.core.lower.utils.reorder import (
    ReorderTensorBSNH,
    ReorderTensorBNSH,
    ReorderTensorTextBSNH,
    ReorderTensorTextBNSH,
)


class TorchSummarizerExecutor(SummarizerExecutor):

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        super().__init__(sparse_config, layout=layout)

    @staticmethod
    def _inter_summarize(t: torch.Tensor, summarize_mode: str, layout: str) -> torch.Tensor:
        _support_inter_summarize_mode = ['max', 'min', 'mean']
        reduction_dim = 3 if layout == "bnsh" else 2
        if summarize_mode not in _support_inter_summarize_mode:
            raise ValueError(
                f'summarize_mode must be in {_support_inter_summarize_mode}')
        if summarize_mode == 'max':
            res = t.max(dim=reduction_dim)[0]
        if summarize_mode == 'min':
            res = t.min(dim=reduction_dim)[0]
        if summarize_mode == 'mean':
            res = t.sum(dim=reduction_dim) / (t != 0).sum(dim=reduction_dim)

        if layout == "bsnh":
            res = res.permute(0, 2, 1, 3)
        return res

    @staticmethod
    def _block_summarize(t: torch.Tensor) -> torch.Tensor:
        t_norm = t.norm(p=2, dim=-1, keepdim=True)
        t_normalized = t / (t_norm + 1e-8)

        cosine_sim_matrix = t_normalized @ t_normalized.transpose(-1, -2)

        token_mask = (t_norm > 1e-8).squeeze(-1)
        pairwise_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)

        diag_mask = torch.eye(t.shape[-2], device=t.device,
                              dtype=torch.bool).expand_as(cosine_sim_matrix)

        final_mask = pairwise_mask & ~diag_mask

        cosine_sim_matrix = cosine_sim_matrix.masked_fill(~final_mask, 0)

        sim_sum = cosine_sim_matrix.sum(dim=(3, 4))

        num_pairs = final_mask.sum(dim=(3, 4))

        avg_sim = sim_sum / (num_pairs + 1e-8)
        return avg_sim

    @staticmethod
    def _intra_summarize(t: torch.Tensor, summarize_mode: str, layout: str) -> torch.Tensor:
        '''
        Docstring for _intra_summarize
        
        t: Tensor of shape (bsz, num_head_ids, numBlocks, blockSize, hidden_dim) if layout == "bnsh"
           or (bsz, numBlocks, blockSize, num_head_ids, hidden_dim)
        return: Tensor of shape (bsz, num_head_ids, numBlocks)
        '''


        if layout == "bsnh":
            t = t.permute(0, 3, 1, 2, 4)  # to (bsz, num_head_ids, numBlocks, blockSize, hidden_dim)

        _support_intra_summarize_mode = ['MeanSim']
        if summarize_mode not in _support_intra_summarize_mode:
            raise ValueError(
                f'summarize_mode must be in {_support_intra_summarize_mode}')

        if summarize_mode == 'CosSim':
            res_sim = t @ t.transpose(-1, -2)
            x_norm = torch.max(torch.abs(res_sim).flatten(-2),
                               dim=-1,
                               keepdim=True)[0].unsqueeze_(-1)
            res_sim /= x_norm
            return res_sim.mean(dim=[-1, -2])

        if summarize_mode == 'MeanSim':
            return TorchSummarizerExecutor._block_summarize(t)

    @staticmethod
    def _validate_block_layout(block_t: torch.Tensor, layout: str, name: str,
                               expected_num_heads: int,
                               expected_num_blocks: int,
                               expected_block_size: int) -> None:
        if block_t.ndim != 5:
            raise ValueError(f"{name} must be rank-5 tensor, got shape={tuple(block_t.shape)}")
        if layout == "bsnh":
            expected = (expected_num_blocks, expected_block_size,
                        expected_num_heads)
            actual = (block_t.shape[1], block_t.shape[2], block_t.shape[3])
            if actual != expected:
                raise ValueError(
                    f"{name} layout mismatch for bsnh: expected (num_blocks, block_size, num_head_ids)={expected}, got {actual}"
                )
        else:
            expected = (expected_num_heads, expected_num_blocks,
                        expected_block_size)
            actual = (block_t.shape[1], block_t.shape[2], block_t.shape[3])
            if actual != expected:
                raise ValueError(
                    f"{name} layout mismatch for bnsh: expected (num_head_ids, num_blocks, block_size)={expected}, got {actual}"
                )

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 head_ids: torch.Tensor, block_q: torch.Tensor,
                 block_k: torch.Tensor, block_v: torch.Tensor,
                 realSeqlen: Optional[int]) -> Dict[str, Any]:
        '''
        q: (bsz, seqlen, num_heads, hidden_dim)
        k: (bsz, seqlen, num_heads, hidden_dim)
        v: (bsz, seqlen, num_heads, hidden_dim)
        block_q: if layout == "bsnh" -> (bsz, qNumBlocks, qBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, qNumBlocks, qBlockSize, hidden_dim)
        block_k: if layout == "bsnh" -> (bsz, kvNumBlocks, kvBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, kvNumBlocks, kvBlockSize, hidden_dim)
        block_v: if layout == "bsnh" -> (bsz, kvNumBlocks, kvBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, kvNumBlocks, kvBlockSize, hidden_dim)
            
        return:
            if self.inter_summarize_enable:
                return block_inter_q: dict(str,tensor(bsz, num_head_ids, qNumBlocks, hidden_dim)), dict(str,tensor(bsz, num_head_ids, kNumBlocks, hidden_dim))
            if self.intra_summarize_enable:
                return block_intra_q (bsz, num_head_ids, qNumBlocks), block_intra_k (bsz, num_head_ids, kNumBlocks)
            
        '''
        if self.layout == "bsnh":
            reorder_tensor = ReorderTensorBSNH
            reorder_tensor_text = ReorderTensorTextBSNH
        elif self.layout == "bnsh":
            reorder_tensor = ReorderTensorBNSH
            reorder_tensor_text = ReorderTensorTextBNSH
        else:
            raise ValueError(f"Unsupported layout '{self.layout}'")

        seqlen3d = self.sparse_config.seqlen3d
        padding_val = 0

        kvNumBlocks3d = self.sparse_config.num_blocks_kv
        kvBlockSize3d = self.sparse_config.block_size_kv

        qNumBlocks3d = self.sparse_config.num_blocks_q
        qBlockSize3d = self.sparse_config.block_size_q

        num_head_ids = head_ids.numel()
        q_num_blocks = prod(qNumBlocks3d)
        kv_num_blocks = prod(kvNumBlocks3d)
        q_block_size = prod(qBlockSize3d)
        kv_block_size = prod(kvBlockSize3d)
        if self.sparse_config.context_length:
            q_num_blocks += self.sparse_config.context_q_num_block
            kv_num_blocks += self.sparse_config.context_kv_num_block

        self._validate_block_layout(block_q, self.layout, "block_q",
                                    num_head_ids, q_num_blocks, q_block_size)
        self._validate_block_layout(block_k, self.layout, "block_k",
                                    num_head_ids, kv_num_blocks,
                                    kv_block_size)
        self._validate_block_layout(block_v, self.layout, "block_v",
                                    num_head_ids, kv_num_blocks,
                                    kv_block_size)

        if self.sparse_config.context_length:

            reorder_tensor_text(block_q, q, head_ids, padding_val, seqlen3d,
                                qBlockSize3d, qNumBlocks3d, realSeqlen)
            reorder_tensor_text(block_k, k, head_ids, padding_val, seqlen3d,
                                kvBlockSize3d, kvNumBlocks3d, realSeqlen)
            reorder_tensor_text(block_v, v, head_ids, padding_val, seqlen3d,
                                kvBlockSize3d, kvNumBlocks3d, realSeqlen)
        else:
            reorder_tensor(block_q, q, head_ids, padding_val, seqlen3d,
                           qBlockSize3d, qNumBlocks3d)
            reorder_tensor(block_k, k, head_ids, padding_val, seqlen3d,
                           kvBlockSize3d, kvNumBlocks3d)
            reorder_tensor(block_v, v, head_ids, padding_val, seqlen3d,
                           kvBlockSize3d, kvNumBlocks3d)

        res = {
            'block_inter_q': None,
            'block_inter_k': None,
            'block_intra_q': None,
            'block_intra_k': None,
        }

        if self.sparse_config.dync_enable:

            if self.sparse_config.inter_summarize_enable:
                block_inter_q = {}
                block_inter_k = {}
                for name, mode in self.sparse_config.q_inter_summarizer_mode.items(
                ):
                    block_inter_q[name] = self._inter_summarize(
                        block_q, mode, self.layout)
                for name, mode in self.sparse_config.k_inter_summarizer_mode.items(
                ):
                    block_inter_k[name] = self._inter_summarize(
                        block_k, mode, self.layout)

                res['block_inter_q'] = block_inter_q
                res['block_inter_k'] = block_inter_k

            if self.sparse_config.intra_summarize_enable:
                if self.sparse_config.q_intra_summarizer_mode is not None:
                    block_intra_q = self._intra_summarize(
                        block_q, self.sparse_config.q_intra_summarizer_mode,
                        self.layout)
                    res['block_intra_q'] = block_intra_q.to(
                        self.sparse_config.attn_dtype)
                if self.sparse_config.k_intra_summarizer_mode is not None:
                    block_intra_k = self._intra_summarize(
                        block_k, self.sparse_config.k_intra_summarizer_mode,
                        self.layout)
                    res['block_intra_k'] = block_intra_k.to(
                        self.sparse_config.attn_dtype)

        return res


class TorchE2EExecutor(E2EExecutor):

    def __init__(self,
                 sparse_config: SparseHeadConfig,
                 layout: str,
                 backend: Optional[str] = None) -> None:
        self.summarizer_exe = TorchSummarizerExecutor(sparse_config,
                                                      layout=layout)
        self.estimator_exe = EstimatorExecutor(sparse_config)
        self.attention_exe = AttentionExecutor(sparse_config,
                                               layout=layout,
                                               backend=backend)
        super().__init__(sparse_config, self.summarizer_exe,
                         self.estimator_exe, self.attention_exe)

    def _check(self, tq: torch.Tensor) -> bool:
        if prod(self.sparse_config.seqlen3d) != tq.shape[1]:
            return False
        return True
