import math
import torch
from torch.nn.attention.flex_attention import flex_attention, BlockMask, _create_sparse_block_from_block_mask

import triton
import triton.language as tl
import torch.nn.functional as F

flex_attention = torch.compile(flex_attention)

attention_layout = "bsnh"

def blocksparse_flashattn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_sparse_mask: torch.Tensor):
    q = q.permute(0, 3, 1, 2, 4).contiguous()
    k = k.permute(0, 3, 1, 2, 4).contiguous()
    v = v.permute(0, 3, 1, 2, 4).contiguous()
    bsz, num_heads, q_num_blocks, q_block_size, head_dim = q.shape
    _, _, k_num_blocks, k_block_size, _ = k.shape
    sm_scale = 1 / math.sqrt(q.shape[-1])
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    q = q.flatten(2, 3)
    k = k.flatten(2, 3)
    v = v.flatten(2, 3)

    bmask = _create_sparse_block_from_block_mask(
        (block_sparse_mask, None), None, q_block_size, k_block_size)

    o = flex_attention(q, k, v, block_mask=bmask)

    q = torch.unflatten(q, 2, (q_num_blocks, q_block_size))
    o = torch.unflatten(o, 2, (q_num_blocks, q_block_size))

    k = torch.unflatten(k, 2, (k_num_blocks, k_block_size))
    v = torch.unflatten(v, 2, (k_num_blocks, k_block_size))
    return o.permute(0, 2, 3, 1, 4).contiguous()
