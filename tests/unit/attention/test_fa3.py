import pytest

from tests.helpers.device import require_cuda_sm90a

require_cuda_sm90a("FA3 requires sm90a (H100/H800)")
pytest.importorskip("flash_attn")
pytest.importorskip("flash_attn_interface")

pytestmark = [pytest.mark.cuda, pytest.mark.sm90a, pytest.mark.fa3]

import torch
from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v2
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3


def _make_varlen_qkv(seq_lens, num_heads=2, head_dim=64):
    q_chunks = []
    k_chunks = []
    v_chunks = []
    for seq_len in seq_lens:
        q_chunks.append(
            torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        )
        k_chunks.append(torch.randn_like(q_chunks[-1]))
        v_chunks.append(torch.randn_like(q_chunks[-1]))

    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    return torch.cat(q_chunks), torch.cat(k_chunks), torch.cat(v_chunks), cu_seqlens


def test_flash_attention_v3_varlen_matches_v2():
    q, k, v, cu_seqlens = _make_varlen_qkv([64, 48])
    max_seqlen = 64

    out_v2 = flash_attn_varlen_func_v2(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=False
    )
    out_v3 = flash_attn_varlen_func_v3(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=False
    )

    torch.testing.assert_close(out_v3, out_v2, rtol=1e-2, atol=1e-2)
