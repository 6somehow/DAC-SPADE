import pytest
import torch

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

from math import prod

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.triton.triton_summarizer import TritonE2EExecutor
from tests.utils.gen_sparse import gen_spatial_sparse_config64


def _try_executor(config: SparseHeadConfig, layout: str, backends):
    last_error = None
    for backend in backends:
        try:
            return TritonE2EExecutor(config, layout=layout, backend=backend)
        except Exception as exc:
            last_error = exc
    pytest.skip(
        f"No available block sparse attention backend for layout '{layout}': {last_error}"
    )


@pytest.mark.parametrize("layout,backends,dtype", [
    ("bsnh", ["cutlass", "flex"], torch.float16),
    ("bnsh", ["bnsh", "tk"], torch.bfloat16),
])
def test_triton_e2e_layouts(monkeypatch, layout, backends, dtype):
    def fake_resolve_backend(_backend, requested_layout=None):
        resolved_layout = requested_layout or layout

        def fake_blocksparse_attn(q, _k, _v, _mask):
            return torch.zeros_like(q)

        return fake_blocksparse_attn, resolved_layout

    monkeypatch.setattr(
        "spade.core.lower.executor.AttentionExecutor._resolve_backend",
        staticmethod(fake_resolve_backend),
    )

    config = gen_spatial_sparse_config64()
    e2e = _try_executor(config, layout, backends)

    bsz = 1
    num_heads = 4
    head_ids = list(range(num_heads))
    seqlen = prod(config.seqlen3d)
    q = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    config.hidden_dim,
                    device="cuda",
                    dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_o = torch.empty_like(q)

    inter_top_val = 1
    intra_top_val = 1.0
    diag_width = config.fixed_diag_width

    e2e(q, k, v, res_o, head_ids, inter_top_val, intra_top_val, diag_width,
        None)
    assert res_o.shape == q.shape
    assert torch.isfinite(res_o).all()
