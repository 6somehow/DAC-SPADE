from typing import Optional, Callable
from math import prod
import torch

from spade.core.lower.utils.utils import get_num_blocks


class SparseHeadConfig:

    def __init__(
        self,
        seqlen3d: tuple[int, int, int],
        hidden_dim: int,
        block_size_q: tuple[int, int, int],
        block_size_kv: tuple[int, int, int],
        fixed_diag_width: int,
        fixed_sink_width: int,
        context_length: Optional[int] = None,
        inter_select_mode: Optional[
            str] = None,  # "topk", "topp", None: no dync
        intra_select_mode: Optional[
            str] = None,  # "topk", "topp", None: no dync
        q_inter_summarizer_mode: Optional[dict[str, str]] = None,
        k_inter_summarizer_mode: Optional[dict[str, str]] = None,
        q_intra_summarizer_mode: Optional[str] = None,
        k_intra_summarizer_mode: Optional[str] = None,
        symbol_inter_estimator: Optional[Callable] = None,
        softmax_scale: Optional[float] = None,
        attn_dtype: Optional[torch.dtype] = None,
        quant_dtype: Optional[torch.dtype] = None,
    ):
        self.hidden_dim = hidden_dim
        self.softmax_scale = softmax_scale
        self.seqlen = prod(seqlen3d)
        self.seqlen3d = seqlen3d
        self.context_length = context_length

        self.block_size_q = block_size_q
        self.block_size_q_int = prod(block_size_q)
        self.num_blocks_q = get_num_blocks(seqlen3d, block_size_q)
        self.num_blocks_q_int = prod(self.num_blocks_q)

        self.block_size_kv = block_size_kv
        self.block_size_kv_int = prod(block_size_kv)
        self.num_blocks_kv = get_num_blocks(seqlen3d, block_size_kv)
        self.num_blocks_kv_int = prod(self.num_blocks_kv)

        if self.context_length:
            blocksize = min(self.block_size_q_int, self.block_size_kv_int)
            assert self.context_length % blocksize == 0
            self.context_q_num_block = self.context_length // self.block_size_q_int
            self.context_kv_num_block = self.context_length // self.block_size_kv_int

        self.fixed_diag_width = fixed_diag_width
        self.fixed_sink_width = fixed_sink_width
        self.q_inter_summarizer_mode = q_inter_summarizer_mode
        self.k_inter_summarizer_mode = k_inter_summarizer_mode
        self.q_intra_summarizer_mode = q_intra_summarizer_mode
        self.k_intra_summarizer_mode = k_intra_summarizer_mode
        self.symbol_inter_estimator = symbol_inter_estimator
        self.attn_dtype = attn_dtype
        self.quant_dtype = quant_dtype

        self.inter_select_mode = inter_select_mode
        self.intra_select_mode = intra_select_mode

        self.dync_enable = inter_select_mode is not None or intra_select_mode is not None
        self.inter_summarize_enable = self.q_inter_summarizer_mode is not None or self.k_inter_summarizer_mode is not None
        self.intra_summarize_enable = self.q_intra_summarizer_mode is not None or self.k_intra_summarizer_mode is not None
        assert not (
            (inter_select_mode is not None) ^ self.inter_summarize_enable
        ), "inter_select_mode and inter_summarize_enable must be both True or False"
        assert not (
            (intra_select_mode is not None) ^ self.intra_summarize_enable
        ), "intra_select_mode and intra_summarize_enable must be both True or False"
        self.inter_is_topk_mode = inter_select_mode == "topk"
        self.intra_is_topk_mode = intra_select_mode == "topk"

    #Example
    @staticmethod
    def SymbolInterEstimator(block_q, block_k):
        """
        This method should be implemented by subclasses to define
        the attention estimation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        return (
            f"SparseHeadConfig(\n"
            f"  dync_enable={self.dync_enable},\n"
            f"  inter_summarize_enable={self.inter_summarize_enable},\n"
            f"  intra_summarize_enable={self.intra_summarize_enable},\n"
            f"  seqlen3d={self.seqlen3d},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  block_size_q={self.block_size_q},\n"
            f"  block_size_kv={self.block_size_kv},\n"
            f"  fixed_diag_width={self.fixed_diag_width},\n"
            f"  fixed_sink_width={self.fixed_sink_width},\n"
            f"  context_length={self.context_length},\n"
            f"  q_inter_summarizer_mode={self.q_inter_summarizer_mode},\n"
            f"  k_inter_summarizer_mode={self.k_inter_summarizer_mode},\n"
            f"  q_intra_summarizer_mode={self.q_intra_summarizer_mode},\n"
            f"  k_intra_summarizer_mode={self.k_intra_summarizer_mode},\n"
            f"  symbol_inter_estimator={self.symbol_inter_estimator is not None and self.symbol_inter_estimator.__name__ != 'SymbolInterEstimator'},\n"
            f"  softmax_scale={self.softmax_scale},\n"
            f"  attn_dtype={self.attn_dtype},\n"
            f"  quant_dtype={self.quant_dtype}\n"
            f")")

    def copy(self):
        return SparseHeadConfig(
            self.seqlen3d,
            self.hidden_dim,
            self.block_size_q,
            self.block_size_kv,
            self.fixed_diag_width,
            self.fixed_sink_width,
            self.context_length,
            self.inter_select_mode,
            self.intra_select_mode,
            self.q_inter_summarizer_mode,
            self.k_inter_summarizer_mode,
            self.q_intra_summarizer_mode,
            self.k_intra_summarizer_mode,
            self.symbol_inter_estimator,
            self.softmax_scale,
            self.attn_dtype,
            self.quant_dtype,
        )
