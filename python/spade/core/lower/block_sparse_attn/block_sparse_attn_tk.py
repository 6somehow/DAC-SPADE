import torch

from spade.core.lower.block_sparse_attn.block_sparse_attn_bnsh import (
    blocksparse_flashattn as _blocksparse_flashattn_bnsh,
)

attention_layout = "bnsh"

def blocksparse_flashattn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_sparse_mask: torch.Tensor):
    return _blocksparse_flashattn_bnsh(q, k, v, block_sparse_mask)
