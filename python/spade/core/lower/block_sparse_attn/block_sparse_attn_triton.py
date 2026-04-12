import math
import torch

import triton
import triton.language as tl
import torch.nn.functional as F

attention_layout = "bnsh"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _fwd_kernel_inner(acc, l_i, m_i, q, k_block_col_idx, block_mask_ptr,
                      k_ptrs, v_ptrs, offs_m, offs_n, stride_ktb, stride_vtb,
                      stride_bmask_n, sm_scale, seqlen_k,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

    mask_val = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n)

    if mask_val == True:
        start_n = k_block_col_idx
        # -- compute qk ----

        k = tl.load(k_ptrs + start_n * stride_ktb)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        qk *= sm_scale
        # qk = tl.where(qk_mask, qk, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(v_ptrs + start_n * stride_vtb)

        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    block_mask_ptr,
    Out,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qmb: tl.constexpr,
    stride_qms: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_knb: tl.constexpr,
    stride_kns: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vnb: tl.constexpr,
    stride_vns: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bmz: tl.constexpr,
    stride_bmh: tl.constexpr,
    stride_bmm: tl.constexpr,
    stride_bmn: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_omb: tl.constexpr,
    stride_oms: tl.constexpr,
    stride_od: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    off_hz = tl.program_id(0)
    start_m = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh + start_m * stride_qmb
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # decode mask

    # initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qms + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kns + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vns + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V

    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs)

    k_block_start = 0
    k_block_end = NUM_BLOCK_N

    # loop over k, v and update accumulator
    for col_idx in range(k_block_start, k_block_end):
        acc, l_i, m_i = _fwd_kernel_inner(acc, l_i, m_i, q, col_idx, mask_ptrs,
                                          k_ptrs, v_ptrs, offs_m, offs_n,
                                          stride_knb, stride_vnb, stride_bmn,
                                          sm_scale, N_CTX, BLOCK_M, BLOCK_N)

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty)

    off_o = off_z * stride_oz + off_h * stride_oh + start_m * stride_omb + offs_m[:, None] * stride_oms + offs_d[
        None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def blocksparse_flashattn(q, k, v, block_sparse_mask):

    BLOCK_M = q.shape[3]
    BLOCK_N = k.shape[3]
    o = torch.empty_like(q)
    bsz, num_heads, q_num_blocks, q_block_size, head_dim = q.shape
    _, _, k_num_blocks, k_block_size, _ = k.shape
    sm_scale = 1 / math.sqrt(q.shape[-1])
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]

    grid = (bsz * num_heads, q_num_blocks)

    assert q.shape[-1] in [64, 128]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        block_sparse_mask,
        o,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_sparse_mask.stride(),
        *o.stride(),
        num_heads,
        q_num_blocks * q_block_size,
        q_num_blocks,
        q_block_size,
        k_num_blocks,
        k_block_size,
        head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o
