from typing import Optional, Dict
import torch
from spade.core.lower.config import SparseHeadConfig

HIDDEN_DIM = 128
DEFAULT_SEQLEN3D = (2, 8, 16)
DEFAULT_ATTN_DTYPE = torch.bfloat16
DEFAULT_QUANT_DTYPE = None

INTER_MAX_MIN = {"Max": "max", "Min": "min"}
INTER_MEAN = {"Mean": "mean"}


def estimate_func_minmax(block_q, block_k):
    q_max_min = block_q["Max"] + block_q["Min"]
    attn_weights_max = torch.matmul(q_max_min,
                                    block_k["Max"].transpose(-1, -2))
    attn_weights_min = torch.matmul(q_max_min,
                                    block_k["Min"].transpose(-1, -2))
    return torch.max(attn_weights_max, attn_weights_min)


def estimate_func_mean(block_q, block_k):
    return torch.matmul(block_q["Mean"], block_k["Mean"].transpose(-1, -2))


def _softmax_scale(hidden_dim: int) -> float:
    return 1.0 / (hidden_dim**0.5)


def _build_sparse_config(
    *,
    seqlen3d: tuple[int, int, int],
    block_size_q: tuple[int, int, int],
    block_size_kv: tuple[int, int, int],
    fixed_diag_width: int,
    fixed_sink_width: int,
    inter_select_mode: Optional[str],
    intra_select_mode: Optional[str],
    q_inter_summarizer_mode: Optional[Dict[str, str]],
    k_inter_summarizer_mode: Optional[Dict[str, str]],
    q_intra_summarizer_mode: Optional[str],
    k_intra_summarizer_mode: Optional[str],
    symbol_inter_estimator,
    context_length: Optional[int] = None,
    hidden_dim: int = HIDDEN_DIM,
    attn_dtype: Optional[torch.dtype] = DEFAULT_ATTN_DTYPE,
    quant_dtype: Optional[torch.dtype] = DEFAULT_QUANT_DTYPE,
) -> SparseHeadConfig:
    return SparseHeadConfig(
        seqlen3d=seqlen3d,
        hidden_dim=hidden_dim,
        block_size_q=block_size_q,
        block_size_kv=block_size_kv,
        fixed_diag_width=fixed_diag_width,
        fixed_sink_width=fixed_sink_width,
        context_length=context_length,
        inter_select_mode=inter_select_mode,
        intra_select_mode=intra_select_mode,
        q_inter_summarizer_mode=q_inter_summarizer_mode,
        k_inter_summarizer_mode=k_inter_summarizer_mode,
        q_intra_summarizer_mode=q_intra_summarizer_mode,
        k_intra_summarizer_mode=k_intra_summarizer_mode,
        symbol_inter_estimator=symbol_inter_estimator,
        softmax_scale=_softmax_scale(hidden_dim),
        attn_dtype=attn_dtype,
        quant_dtype=quant_dtype,
    )


def gen_spatial_sparse_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 8, 8),
        block_size_kv=(1, 8, 8),
        fixed_diag_width=4,
        fixed_sink_width=4,
        inter_select_mode="topk",
        intra_select_mode="topp",
        q_inter_summarizer_mode=INTER_MAX_MIN,
        k_inter_summarizer_mode=INTER_MAX_MIN,
        q_intra_summarizer_mode="MeanSim",
        k_intra_summarizer_mode="MeanSim",
        symbol_inter_estimator=estimate_func_minmax,
    )


def gen_spatial_sparse_selfsim_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 8, 16),
        block_size_kv=(1, 8, 16),
        fixed_diag_width=4,
        fixed_sink_width=4,
        inter_select_mode="topk",
        intra_select_mode="topp",
        q_inter_summarizer_mode=INTER_MAX_MIN,
        k_inter_summarizer_mode=INTER_MAX_MIN,
        q_intra_summarizer_mode="MeanSim",
        k_intra_summarizer_mode="MeanSim",
        symbol_inter_estimator=estimate_func_minmax,
    )


def gen_spatial_sparse_topp_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 8, 16),
        block_size_kv=(1, 8, 16),
        fixed_diag_width=4,
        fixed_sink_width=2,
        inter_select_mode="topk",
        intra_select_mode="topp",
        q_inter_summarizer_mode=INTER_MEAN,
        k_inter_summarizer_mode=INTER_MEAN,
        q_intra_summarizer_mode="MeanSim",
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_mean,
    )


def gen_sink_sparse_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 8, 16),
        block_size_kv=(1, 8, 16),
        fixed_diag_width=15,
        fixed_sink_width=10,
        inter_select_mode=None,
        intra_select_mode=None,
        q_inter_summarizer_mode=None,
        k_inter_summarizer_mode=None,
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=None,
    )


def gen_temporal_sparse_topp_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=(8, 4, 16),
        block_size_q=(8, 1, 16),
        block_size_kv=(8, 1, 16),
        fixed_diag_width=4,
        fixed_sink_width=10,
        inter_select_mode="topp",
        intra_select_mode="topk",
        q_inter_summarizer_mode=INTER_MEAN,
        k_inter_summarizer_mode=INTER_MEAN,
        q_intra_summarizer_mode="MeanSim",
        k_intra_summarizer_mode="MeanSim",
        symbol_inter_estimator=estimate_func_mean,
    )


def gen_temporal_sparse_topk_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=(8, 4, 16),
        block_size_q=(8, 1, 16),
        block_size_kv=(8, 1, 16),
        fixed_diag_width=4,
        fixed_sink_width=2,
        inter_select_mode="topk",
        intra_select_mode=None,
        q_inter_summarizer_mode=INTER_MAX_MIN,
        k_inter_summarizer_mode=INTER_MAX_MIN,
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
    )


def gen_spatial_sparse_config64() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 4, 16),
        block_size_kv=(1, 4, 16),
        fixed_diag_width=4,
        fixed_sink_width=4,
        inter_select_mode="topk",
        intra_select_mode=None,
        q_inter_summarizer_mode=INTER_MAX_MIN,
        k_inter_summarizer_mode=INTER_MAX_MIN,
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
    )


def gen_mixed_text_sparse_config() -> SparseHeadConfig:
    return _build_sparse_config(
        seqlen3d=DEFAULT_SEQLEN3D,
        block_size_q=(1, 8, 8),
        block_size_kv=(1, 8, 8),
        fixed_diag_width=4,
        fixed_sink_width=4,
        context_length=64,
        inter_select_mode="topk",
        intra_select_mode="topp",
        q_inter_summarizer_mode=INTER_MAX_MIN,
        k_inter_summarizer_mode=INTER_MAX_MIN,
        q_intra_summarizer_mode="MeanSim",
        k_intra_summarizer_mode="MeanSim",
        symbol_inter_estimator=estimate_func_minmax,
    )


__all__ = [
    "estimate_func_minmax",
    "estimate_func_mean",
    "gen_spatial_sparse_config",
    "gen_spatial_sparse_selfsim_config",
    "gen_spatial_sparse_topp_config",
    "gen_sink_sparse_config",
    "gen_temporal_sparse_topp_config",
    "gen_temporal_sparse_topk_config",
    "gen_spatial_sparse_config64",
    "gen_mixed_text_sparse_config",
]
