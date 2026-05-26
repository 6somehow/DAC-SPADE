# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import torch

from block_sparse_attn.block_sparse_attn_interface import (
    _get_block_sparse_extension,
    _check_exact_streaming_inputs,
    _check_mask_inputs,
    _check_qkv_dtype_device,
    _normalize_softmax_scale,
    _to_row_blockmask,
    maybe_contiguous,
    replace_ones_with_count,
)


def _is_hopper(device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0)


def _is_pure_blocksparse(head_mask_type: torch.Tensor) -> bool:
    return bool(torch.all(head_mask_type == 1).item())


def _expand_blockmask_for_tk(
    blockmask: torch.Tensor,
    m_block_dim: int,
    n_block_dim: int,
) -> torch.Tensor:
    if m_block_dim != 64:
        blockmask = blockmask.repeat_interleave(m_block_dim // 64, dim=-2)
    if n_block_dim != 64:
        blockmask = blockmask.repeat_interleave(n_block_dim // 64, dim=-1)
    return blockmask


def _mask_to_bsr(mask: torch.Tensor):
    mask = mask.to(dtype=torch.bool)
    bsz, num_heads, q_blocks, k_blocks = mask.shape
    indices = torch.arange(k_blocks, device=mask.device, dtype=torch.int32)
    indices = indices.view(1, 1, 1, k_blocks).expand(bsz, num_heads, q_blocks, k_blocks)
    padded = torch.where(mask, indices, torch.full_like(indices, k_blocks))
    bsr = torch.sort(padded, dim=-1).values
    bsr = torch.where(bsr == k_blocks, torch.full_like(bsr, -1), bsr)
    num_blocks = mask.sum(dim=-1, dtype=torch.int32)
    return bsr.contiguous(), num_blocks.contiguous()


def _maybe_hopper_fast_path(
    q,
    k,
    v,
    head_mask_type,
    base_blockmask,
    streaming_info,
    p_dropout,
    is_causal,
    exact_streaming,
    return_attn_probs,
    m_block_dim,
    n_block_dim,
):
    if base_blockmask is None:
        return None
    if not _is_hopper(q.device):
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return None
    if q.shape[-1] not in (64, 128):
        return None
    if p_dropout != 0.0 or is_causal or exact_streaming or return_attn_probs:
        return None
    if streaming_info is not None:
        return None
    if not _is_pure_blocksparse(head_mask_type):
        return None

    ext = _get_block_sparse_extension(required=True)
    if not hasattr(ext, "fwd_tk"):
        raise RuntimeError(
            "Hopper block-sparse fast path requires block_sparse_attn_cuda.fwd_tk, "
            "but the loaded extension does not export it. Rebuild the native "
            "extension on the H100 node with ThunderKittens available and "
            "BLOCK_SPARSE_ATTN_CUDA_ARCHS including 90a."
        )
    expanded_mask = _expand_blockmask_for_tk(
        base_blockmask.contiguous(), m_block_dim=m_block_dim, n_block_dim=n_block_dim
    )
    q2k_block_sparse_index, q2k_block_sparse_num = _mask_to_bsr(expanded_mask)
    kernel_ret = ext.fwd_tk(
        maybe_contiguous(q),
        maybe_contiguous(k),
        maybe_contiguous(v),
        q2k_block_sparse_index,
        q2k_block_sparse_num,
    )
    return _extract_out_from_bnsh_return(kernel_ret)


def _extract_out_from_bnsh_return(kernel_ret):
    if isinstance(kernel_ret, (tuple, list)):
        if len(kernel_ret) == 0:
            raise RuntimeError("BNSH forward kernel returned an empty sequence")
        return kernel_ret[0]
    return kernel_ret


def _check_bnsh_inputs(q, k, v, head_mask_type):
    _check_qkv_dtype_device(q, k, v)
    if not q.is_cuda:
        raise AssertionError("BNSH tensors must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise AssertionError("BNSH tensors must be 4D")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise AssertionError("Native BNSH forward only supports fp16 and bf16 inputs")
    if q.shape[-1] not in (32, 64, 128) or k.shape[-1] not in (32, 64, 128) or v.shape[-1] not in (32, 64, 128):
        raise AssertionError("Native BNSH forward only supports head_dim in {32, 64, 128}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AssertionError("Batch size mismatch in q/k/v")
    if k.shape[1] != v.shape[1]:
        raise AssertionError("k/v head dimension mismatch")
    if k.shape[2] != v.shape[2]:
        raise AssertionError("k/v sequence length mismatch")
    if head_mask_type.ndim != 1:
        raise AssertionError("head_mask_type must be a 1D tensor")
    if q.shape[1] != head_mask_type.numel():
        raise AssertionError("BNSH layout requires q.shape[1] to match head_mask_type length")


def _check_bnsh_block_dims(m_block_dim, n_block_dim):
    if m_block_dim % 64 != 0:
        raise AssertionError(
            "Native BNSH forward requires m_block_dim to be a multiple of 64 (for example 64 or 128)"
        )
    if n_block_dim % 64 != 0:
        raise AssertionError(
            "Native BNSH forward requires n_block_dim to be a multiple of 64 (for example 64 or 128)"
        )


<<<<<<< HEAD
=======
def _arch_mismatch_error_message(device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = f"{capability[0]}{capability[1]}"
    if capability == (9, 0):
        arch = "90a"
    return (
        "The loaded block_sparse_attn native extension does not contain a usable "
        f"kernel image for the active GPU (sm{arch}). Rebuild "
        "block_sparse_attn_cuda with BLOCK_SPARSE_ATTN_CUDA_ARCHS including this "
        "architecture. For mixed A100/4090/H100 nodes, rebuild with something like "
        '`BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89;90a"`.'
    )


>>>>>>> dev
def block_sparse_attn_func_bnsh(
    q,
    k,
    v,
    head_mask_type,
    base_blockmask=None,
    streaming_info=None,
    max_seqlen_q_=None,
    max_seqlen_k_=None,
    p_dropout=0.0,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
    m_block_dim=64,
    n_block_dim=64,
    window_size_left=-1,
    window_size_right=-1,
):
    del deterministic

    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise NotImplementedError("Native BNSH forward does not support backward")
    if p_dropout != 0.0:
        raise NotImplementedError("Native BNSH forward does not support dropout")
    if return_attn_probs:
        raise NotImplementedError("Native BNSH forward does not support return_attn_probs")

    _check_bnsh_inputs(q, k, v, head_mask_type)
    _check_bnsh_block_dims(m_block_dim, n_block_dim)

    if max_seqlen_q_ is None:
        max_seqlen_q_ = q.shape[2]
    if max_seqlen_k_ is None:
        max_seqlen_k_ = k.shape[2]

    hopper_out = _maybe_hopper_fast_path(
        q=q,
        k=k,
        v=v,
        head_mask_type=head_mask_type,
        base_blockmask=base_blockmask,
        streaming_info=streaming_info,
        p_dropout=p_dropout,
        is_causal=is_causal,
        exact_streaming=exact_streaming,
        return_attn_probs=return_attn_probs,
        m_block_dim=m_block_dim,
        n_block_dim=n_block_dim,
    )
    if hopper_out is not None:
        return hopper_out

    softmax_scale = _normalize_softmax_scale(softmax_scale, q)
    head_mask_type = maybe_contiguous(head_mask_type.to(dtype=torch.int32))
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    _check_mask_inputs(base_blockmask, blocksparse_head_num)
    _check_exact_streaming_inputs(exact_streaming, streaming_info, is_causal)

    row_blockmask = _to_row_blockmask(base_blockmask, is_causal)
    if streaming_info is not None:
        streaming_info = maybe_contiguous(streaming_info.to(dtype=torch.int32))

    ext = _get_block_sparse_extension(required=True)
<<<<<<< HEAD
    kernel_ret = ext.fwd_bnsh(
        maybe_contiguous(q),
        maybe_contiguous(k),
        maybe_contiguous(v),
        head_mask_type,
        streaming_info,
        row_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        m_block_dim,
        n_block_dim,
        exact_streaming,
        False,
        None,
    )
=======
    try:
        kernel_ret = ext.fwd_bnsh(
            maybe_contiguous(q),
            maybe_contiguous(k),
            maybe_contiguous(v),
            head_mask_type,
            streaming_info,
            row_blockmask,
            max_seqlen_q_,
            max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            window_size_left,
            window_size_right,
            m_block_dim,
            n_block_dim,
            exact_streaming,
            False,
            None,
        )
    except Exception as exc:
        if "no kernel image is available for execution on the device" in str(exc):
            raise RuntimeError(_arch_mismatch_error_message(q.device)) from exc
        raise
>>>>>>> dev
    return _extract_out_from_bnsh_return(kernel_ret)
