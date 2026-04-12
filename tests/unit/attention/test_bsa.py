import pytest

from tests.helpers.device import require_cuda_sm90a

require_cuda_sm90a("Block sparse attention requires sm90a (H100/H800)")
pytest.importorskip("flash_attn_interface")

pytestmark = [pytest.mark.cuda, pytest.mark.sm90a, pytest.mark.fa3]

import torch

from block_sparse_attn._extension import load_extension

if load_extension(required=False) is None:
    pytest.skip("block_sparse_attn native extension not built", allow_module_level=True)

from spade.core.lower.block_sparse_attn.block_sparse_attn_cutlass import (
    blocksparse_flashattn as blocksparse_flashattn_cutlass,
)
from spade.core.lower.block_sparse_attn.block_sparse_attn_flexattention import (
    blocksparse_flashattn as blocksparse_flashattn_flex,
)
from spade.core.lower.block_sparse_attn.block_sparse_attn_triton import (
    blocksparse_flashattn as blocksparse_flashattn_triton,
)


def test_block_sparse_attention_backends_match_flexattention():
    q_block_size = 128
    k_block_size = 128
    q_num_blocks = 2
    k_num_blocks = 2
    num_heads = 2
    bsz = 1
    hidden = 128

    q = torch.randn(
        bsz,
        num_heads,
        q_num_blocks,
        q_block_size,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        bsz,
        num_heads,
        k_num_blocks,
        k_block_size,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    block_sparse_mask = torch.tensor(
        [[[[1, 0], [1, 1]], [[1, 1], [0, 1]]]],
        dtype=torch.int,
        device="cuda",
    )

    out_flex = blocksparse_flashattn_flex(q, k, v, block_sparse_mask)
    out_triton = blocksparse_flashattn_triton(q, k, v, block_sparse_mask)
    out_cutlass = blocksparse_flashattn_cutlass(
        q.transpose(1, 3).transpose(1, 2),
        k.transpose(1, 3).transpose(1, 2),
        v.transpose(1, 3).transpose(1, 2),
        block_sparse_mask,
    ).transpose(1, 3).transpose(2, 3)

    torch.testing.assert_close(out_triton, out_flex, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_cutlass, out_flex, rtol=1e-2, atol=1e-2)
