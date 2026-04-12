from typing import Tuple

import torch
import triton
import triton.language as tl
from spade_utils import mask_to_bsr


@triton.jit
def attn_mask_half_bigbird_kernel(attn_scores, q_block_seqlen: tl.constexpr,
                                  v_block_seqlen: tl.constexpr,
                                  block_block_size: tl.constexpr,
                                  sink_mask_width: tl.constexpr,
                                  diag_mask_width: tl.constexpr,
                                  val: tl.constexpr):
    DTYPE = attn_scores.dtype.element_ty
    head_idx = tl.program_id(0).to(tl.int64)
    tid = tl.program_id(1).to(tl.int64)
    block_seqlen_st = tid * block_block_size

    offset_row = tl.arange(0, block_block_size).to(tl.int64) + block_seqlen_st
    mask_row = offset_row < q_block_seqlen

    block_sink_mask: tl.constexpr = triton.next_power_of_2(sink_mask_width)
    offset_col_sink = tl.arange(0, block_sink_mask)
    mask_col_sink = offset_col_sink < sink_mask_width

    tl.store(attn_scores + head_idx * q_block_seqlen * v_block_seqlen +
             offset_row[:, None] * v_block_seqlen + offset_col_sink[None, :],
             val,
             mask=mask_row[:, None] & mask_col_sink[None, :])

    # the diagonal part of the mask
    block_diag_mask0: tl.constexpr = triton.next_power_of_2(
        block_block_size * v_block_seqlen // q_block_seqlen + diag_mask_width)
    block_diag_mask1: tl.constexpr = triton.next_power_of_2(block_block_size +
                                                            diag_mask_width)
    offset_row = tl.arange(0, block_diag_mask1)[:, None] + block_seqlen_st
    mask_row = offset_row < q_block_seqlen
    offset_col = tl.arange(0, block_diag_mask0)[
        None, :] + block_seqlen_st * v_block_seqlen // q_block_seqlen
    mask_col = offset_col < v_block_seqlen
    mask = (mask_row & mask_col) & (
        tl.abs(offset_row * v_block_seqlen // q_block_seqlen - offset_col)
        <= diag_mask_width)
    tl.store(attn_scores + head_idx * q_block_seqlen * v_block_seqlen +
             offset_row * v_block_seqlen + offset_col,
             val,
             mask=mask)


def StaticMaskSet(
    attn_scores: torch.Tensor,
    val: int,
    sink_mask_width=4,
    diag_mask_width=4,
):
    bsz, num_heads, q_block_seqlen, v_block_seqlen = attn_scores.shape

    if isinstance(diag_mask_width, int):
        bblockSize = 8

        grid = lambda META: (bsz * num_heads,
                             triton.cdiv(q_block_seqlen, bblockSize))
        attn_mask_half_bigbird_kernel[grid](
            attn_scores,
            q_block_seqlen,
            v_block_seqlen,
            bblockSize,
            sink_mask_width,
            diag_mask_width,
            val=val,
        )
    else:
        raise ValueError("diag_mask_width must be int")

@triton.jit
def Mask2BSRKernel(
    mask_ptr,
    bsr_ptr,
    bsr_num_ptr,
    mask_bs_stride,
    mask_h_stride,
    mask_q_stride,
    mask_kv_stride,
    bsr_bs_stride,
    bsr_h_stride,
    bsr_q_stride,
    bsr_kv_stride,
    bsr_num_bs_stride,
    bsr_num_h_stride,
    bsr_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0).to(tl.int64), tl.program_id(1).to(
        tl.int64), tl.program_id(2).to(tl.int64)
    bsr_ptr_base = bsr_ptr + b * bsr_bs_stride + h * bsr_h_stride + q * bsr_q_stride
    mask_ptr_base = mask_ptr + b * mask_bs_stride + h * mask_h_stride + q * mask_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        mask = tl.load(mask_ptr_base + i * mask_kv_stride)
        if mask:
            tl.store(bsr_ptr_base + num * bsr_kv_stride, i)
            num += 1

    tl.store(
        bsr_num_ptr + b * bsr_num_bs_stride + h * bsr_num_h_stride +
        q * bsr_num_q_stride, num)


def Mask2BSRTriton(sparse_mask: torch.Tensor) -> torch.Tensor:
    """
    sparse_mask: [bsz, num_heads, q_num_blocks, k_num_blocks] dtype: torch.bool
    return:
        bsr_mask: [bsz, num_heads, q_num_blocks, k_num_blocks] dtype: torch.int32
        num_blocks: [bsz, num_heads, q_num_blocks] dtype: torch.int32
    """
    bsz, num_heads, q_num_blocks, k_num_blocks = sparse_mask.size()
    bsr = torch.full((bsz, num_heads, q_num_blocks, k_num_blocks),
                     -1,
                     dtype=torch.int32,
                     device=sparse_mask.device)
    num_blocks = torch.empty((bsz, num_heads, q_num_blocks),
                             dtype=torch.int32,
                             device=sparse_mask.device)
    grid = (bsz, num_heads, q_num_blocks)
    Mask2BSRKernel[grid](sparse_mask, bsr, num_blocks, *sparse_mask.stride(),
                         *bsr.stride(), *num_blocks.stride(), k_num_blocks)

    return bsr, num_blocks


def Mask2BSRCUDA(
        sparse_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sparse_mask: [bsz, num_heads, q_num_blocks, k_num_blocks] dtype: torch.bool
    return:
        bsr_mask: [bsz, num_heads, q_num_blocks, k_num_blocks] dtype: torch.int32
        num_blocks: [bsz, num_heads, q_num_blocks] dtype: torch.int32
    """
    bsz, num_heads, q_num_blocks, k_num_blocks = sparse_mask.size()

    bsr = torch.full((bsz, num_heads, q_num_blocks, k_num_blocks),
                     -1,
                     dtype=torch.int32,
                     device=sparse_mask.device)
    num_blocks = torch.empty((bsz, num_heads, q_num_blocks),
                             dtype=torch.int32,
                             device=sparse_mask.device)
    mask_to_bsr(sparse_mask, bsr, num_blocks)
    return bsr, num_blocks
