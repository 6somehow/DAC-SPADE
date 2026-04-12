import importlib
from math import prod

import pytest
import torch

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.torch.torch_summarizer import TorchE2EExecutor
from tests.utils.gen_sparse import gen_spatial_sparse_selfsim_config


def _try_executor(config: SparseHeadConfig, layout: str, backends):
    last_error = None
    for backend in backends:
        try:
            return TorchE2EExecutor(config, layout=layout, backend=backend)
        except Exception as exc:
            last_error = exc
    pytest.skip(
        f"No available block sparse attention backend for layout '{layout}': {last_error}"
    )


@pytest.mark.parametrize("layout,backends,dtype", [
    ("bsnh", ["cutlass", "flex"], torch.float16),
    ("bnsh", ["bnsh", "tk"], torch.bfloat16),
])
def test_torch_e2e_layouts(monkeypatch, layout, backends, dtype):
    def fake_resolve_backend(_backend, requested_layout=None):
        resolved_layout = requested_layout or layout

        def fake_blocksparse_attn(q, _k, _v, _mask):
            return torch.zeros_like(q)

        return fake_blocksparse_attn, resolved_layout

    monkeypatch.setattr(
        "spade.core.lower.executor.AttentionExecutor._resolve_backend",
        staticmethod(fake_resolve_backend),
    )

    config = gen_spatial_sparse_selfsim_config()
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


def test_attention_executor_prefers_bnsh_on_sm8x(monkeypatch):
    import spade.core.lower.executor as executor_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9)
    )

    def fake_import(name):
        if name.endswith("block_sparse_attn_bnsh"):
            class Module:
                attention_layout = "bnsh"

                @staticmethod
                def blocksparse_flashattn(q, k, v, mask):
                    return q

            return Module()
        raise ImportError(name)

    monkeypatch.setattr(executor_module.importlib, "import_module", fake_import)

    fn, layout = executor_module.AttentionExecutor._resolve_backend(None)
    assert layout == "bnsh"
    assert fn is not None


def test_attention_executor_cuda_alias_respects_bnsh_layout_on_sm90(monkeypatch):
    import spade.core.lower.executor as executor_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0)
    )

    def fake_import(name):
        if name.endswith("block_sparse_attn_bnsh"):
            class Module:
                attention_layout = "bnsh"

                @staticmethod
                def blocksparse_flashattn(q, k, v, mask):
                    return q

            return Module()
        if name.endswith("block_sparse_attn_cutlass"):
            class Module:
                attention_layout = "bsnh"

                @staticmethod
                def blocksparse_flashattn(q, k, v, mask):
                    return q

            return Module()
        raise ImportError(name)

    monkeypatch.setattr(executor_module.importlib, "import_module", fake_import)

    fn, layout = executor_module.AttentionExecutor._resolve_backend(
        "cuda", requested_layout="bnsh"
    )
    assert layout == "bnsh"
    assert fn is not None


def test_block_sparse_attn_bnsh_validation_errors(monkeypatch):
    pytest.importorskip("block_sparse_attn")
    module = importlib.import_module(
        "spade.core.lower.block_sparse_attn.block_sparse_attn_bnsh"
    )

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0)
    )

    q = torch.randn(1, 2, 2, 64, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)

    with pytest.raises(AssertionError, match="torch.bfloat16"):
        module.blocksparse_flashattn(q, k, v, mask)

    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    with pytest.raises(AssertionError, match="head_dim in"):
        module.blocksparse_flashattn(
            q_bf16[..., :64],
            k_bf16[..., :64],
            v_bf16[..., :64],
            mask,
        )

    q_bad_block = torch.randn(1, 2, 2, 96, 128, device="cuda", dtype=torch.bfloat16)
    k_bad_block = torch.randn_like(q_bad_block)
    v_bad_block = torch.randn_like(q_bad_block)
    bad_mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)
    with pytest.raises(AssertionError, match="multiple of 64"):
        module.blocksparse_flashattn(q_bad_block, k_bad_block, v_bad_block, bad_mask)

    with pytest.raises(AssertionError, match="forward-only"):
        with torch.enable_grad():
            module.blocksparse_flashattn(
                q_bf16.requires_grad_(),
                k_bf16.requires_grad_(),
                v_bf16.requires_grad_(),
                mask,
            )
