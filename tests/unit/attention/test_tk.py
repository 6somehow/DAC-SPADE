import pytest

from tests.helpers.device import require_cuda_sm90a

require_cuda_sm90a("Hopper block-sparse fast path requires sm90a (H100/H800)")

pytestmark = [pytest.mark.cuda, pytest.mark.sm90a]

import torch

from block_sparse_attn import block_sparse_attn_func_bnsh


class _FakeExtension:

    def __init__(self):
        self.fwd_tk_calls = []
        self.fwd_bnsh_calls = []

    def fwd_tk(self, q, k, v, block_index, num_block):
        self.fwd_tk_calls.append((block_index.clone(), num_block.clone()))
        return (torch.full_like(q, 3), torch.empty(0, device=q.device))

    def fwd_bnsh(self, q, k, v, *_args):
        self.fwd_bnsh_calls.append(True)
        return (torch.full_like(q, 7), torch.empty(0, device=q.device))


def _make_inputs(seq_len=128, num_heads=2, head_dim=64):
    q = torch.randn(1, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    head_mask_type = torch.ones(num_heads, device="cuda", dtype=torch.int32)
    return q, k, v, head_mask_type


def test_hopper_fast_path_uses_unified_tk_kernel(monkeypatch):
    import block_sparse_attn.block_sparse_attn_interface_bnsh as module

    fake_ext = _FakeExtension()
    monkeypatch.setattr(module, "_get_block_sparse_extension", lambda required=False: fake_ext)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))

    q, k, v, head_mask_type = _make_inputs(seq_len=128, head_dim=64)
    mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)

    out = block_sparse_attn_func_bnsh(
        q,
        k,
        v,
        head_mask_type=head_mask_type,
        base_blockmask=mask,
        m_block_dim=64,
        n_block_dim=64,
    )

    assert fake_ext.fwd_tk_calls
    assert not fake_ext.fwd_bnsh_calls
    block_index, num_block = fake_ext.fwd_tk_calls[0]
    assert block_index.shape == (1, 2, 2, 2)
    assert torch.equal(num_block, torch.full((1, 2, 2), 2, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(out, torch.full_like(q, 3))


def test_hopper_fast_path_expands_128_token_masks(monkeypatch):
    import block_sparse_attn.block_sparse_attn_interface_bnsh as module

    fake_ext = _FakeExtension()
    monkeypatch.setattr(module, "_get_block_sparse_extension", lambda required=False: fake_ext)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))

    q, k, v, head_mask_type = _make_inputs(seq_len=256, head_dim=128)
    mask = torch.tensor([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                        device="cuda",
                        dtype=torch.bool)

    block_sparse_attn_func_bnsh(
        q,
        k,
        v,
        head_mask_type=head_mask_type,
        base_blockmask=mask,
        m_block_dim=128,
        n_block_dim=128,
    )

    block_index, num_block = fake_ext.fwd_tk_calls[0]
    assert block_index.shape == (1, 2, 4, 4)
    assert torch.equal(num_block, torch.full((1, 2, 4), 2, device="cuda", dtype=torch.int32))


def test_hopper_fast_path_requires_fwd_tk_symbol(monkeypatch):
    import block_sparse_attn.block_sparse_attn_interface_bnsh as module

    class ExtensionWithoutTk:
        pass

    monkeypatch.setattr(module, "_get_block_sparse_extension", lambda required=False: ExtensionWithoutTk())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))

    q, k, v, head_mask_type = _make_inputs(seq_len=128, head_dim=64)
    mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)

    with pytest.raises(RuntimeError, match="does not export"):
        block_sparse_attn_func_bnsh(
            q,
            k,
            v,
            head_mask_type=head_mask_type,
            base_blockmask=mask,
            m_block_dim=64,
            n_block_dim=64,
        )


def test_hopper_falls_back_to_native_bnsh_when_mask_types_are_mixed(monkeypatch):
    import block_sparse_attn.block_sparse_attn_interface_bnsh as module

    fake_ext = _FakeExtension()
    monkeypatch.setattr(module, "_get_block_sparse_extension", lambda required=False: fake_ext)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))

    q, k, v, _ = _make_inputs(seq_len=128, head_dim=64)
    head_mask_type = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)

    out = block_sparse_attn_func_bnsh(
        q,
        k,
        v,
        head_mask_type=head_mask_type,
        base_blockmask=mask,
        m_block_dim=64,
        n_block_dim=64,
    )

    assert fake_ext.fwd_bnsh_calls
    assert not fake_ext.fwd_tk_calls
    torch.testing.assert_close(out, torch.full_like(q, 7))
