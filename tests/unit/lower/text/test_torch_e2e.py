from math import prod

import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from spade.core.lower.torch.torch_summarizer import TorchE2EExecutor
from tests.utils.gen_sparse import gen_mixed_text_sparse_config


def test_torch_text_e2e_runs_with_context(monkeypatch):
    def fake_resolve_backend(_backend, requested_layout=None):
        layout = requested_layout or "bsnh"

        def fake_blocksparse_attn(q, _k, _v, _mask):
            return torch.zeros_like(q)

        return fake_blocksparse_attn, layout

    monkeypatch.setattr(
        "spade.core.lower.executor.AttentionExecutor._resolve_backend",
        staticmethod(fake_resolve_backend),
    )

    sparse_config = gen_mixed_text_sparse_config()
    e2e = TorchE2EExecutor(sparse_config, layout="bsnh")
    prompt_length = 11
    seqlen = prod(sparse_config.seqlen3d) + sparse_config.context_length
    real_seqlen = prod(sparse_config.seqlen3d) + prompt_length
    q = torch.randn(1, seqlen, 2, sparse_config.hidden_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_o = torch.empty_like(q)

    sparse_rate = e2e(
        q,
        k,
        v,
        res_o,
        [0, 1],
        1,
        0.8,
        sparse_config.fixed_diag_width,
        real_seqlen,
    )

    assert torch.isfinite(res_o[:, :real_seqlen]).all()
    assert 0 <= sparse_rate.item() <= 1
