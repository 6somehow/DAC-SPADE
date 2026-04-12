from math import prod

import pytest

from tests.helpers.device import require_cuda_sm90a

require_cuda_sm90a("CUDA E2E path requires sm90a (H100/H800)")
pytest.importorskip("flash_attn_interface")

pytestmark = [pytest.mark.cuda, pytest.mark.sm90a, pytest.mark.fa3]

import torch

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.cuda.cuda_summarizer import CUDAE2EExecutor


def _dummy_estimator(block_q, block_k):
    return torch.zeros(
        block_q["Mean"].shape[:-1] + (block_k["Mean"].shape[-2],),
        device=block_q["Mean"].device,
        dtype=block_q["Mean"].dtype,
    )


def _build_config() -> SparseHeadConfig:
    return SparseHeadConfig(
        seqlen3d=(2, 2, 4),
        hidden_dim=128,
        block_size_q=(1, 1, 2),
        block_size_kv=(1, 1, 2),
        fixed_diag_width=1,
        fixed_sink_width=1,
        inter_select_mode="topk",
        intra_select_mode=None,
        q_inter_summarizer_mode={"Mean": "mean"},
        k_inter_summarizer_mode={"Mean": "mean"},
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=_dummy_estimator,
        softmax_scale=1.0 / (128**0.5),
        attn_dtype=torch.bfloat16,
        quant_dtype=None,
    )


def test_cuda_e2e_runs_with_current_project_components(monkeypatch):
    def fake_resolve_backend(_backend, requested_layout=None):
        layout = requested_layout or "bsnh"

        def fake_blocksparse_attn(q, _k, _v, _mask):
            return torch.zeros_like(q)

        return fake_blocksparse_attn, layout

    monkeypatch.setattr(
        "spade.core.lower.executor.AttentionExecutor._resolve_backend",
        staticmethod(fake_resolve_backend),
    )

    config = _build_config()
    e2e = CUDAE2EExecutor(config)
    seqlen = prod(config.seqlen3d)
    q = torch.randn(1, seqlen, 2, config.hidden_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_o = torch.empty_like(q)

    sparse_rate = e2e(q, k, v, res_o, [0, 1], 1, 1.0, config.fixed_diag_width, None)

    assert torch.isfinite(res_o).all()
    assert 0 <= sparse_rate.item() <= 1
