"""Shared tensor factories for tests."""

import torch

from tests.helpers.layouts import apply_layout


def make_qkv(bsz, seqlen, num_heads, head_dim, device, dtype, layout):
    q = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    return (
        apply_layout(q, layout),
        apply_layout(k, layout),
        apply_layout(v, layout),
    )
