import torch

from block_sparse_attn import block_sparse_attn_func_bnsh

attention_layout = "bnsh"

def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise AssertionError("BNSH block-sparse attention requires CUDA tensors")
    capability = torch.cuda.get_device_capability(q.device)
    if capability[0] not in (8, 9):
        raise AssertionError(
            "BNSH block-sparse attention only supports Ampere, Ada, and Hopper, "
            f"got sm{capability[0]}{capability[1]}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise AssertionError(
            "BNSH block-sparse attention requires q/k/v dtype torch.bfloat16"
        )
    if q.ndim != 5 or k.ndim != 5 or v.ndim != 5:
        raise AssertionError("BNSH block-sparse attention expects 5D block tensors")

    bsz, num_heads, q_num_blocks, q_block_size, head_dim = q.shape
    kb, kh, k_num_blocks, k_block_size, k_head_dim = k.shape
    vb, vh, v_num_blocks, v_block_size, v_head_dim = v.shape
    if (kb, kh) != (bsz, num_heads) or (vb, vh) != (bsz, num_heads):
        raise AssertionError("BNSH block-sparse attention requires matching batch/head dims")
    if (k_num_blocks, k_block_size, k_head_dim) != (v_num_blocks, v_block_size, v_head_dim):
        raise AssertionError("BNSH block-sparse attention requires matching k/v shapes")
    allowed_head_dims = {64, 128} if capability == (9, 0) else {128}
    if head_dim not in allowed_head_dims or k_head_dim not in allowed_head_dims or v_head_dim not in allowed_head_dims:
        allowed = ", ".join(str(dim) for dim in sorted(allowed_head_dims))
        raise AssertionError(
            f"BNSH block-sparse attention only supports head_dim in {{{allowed}}}"
        )
    if q_block_size != k_block_size:
        raise AssertionError(
            "BNSH block-sparse attention requires q and k block sizes to match, "
            f"got q_block_size={q_block_size}, k_block_size={k_block_size}"
        )
    if q_block_size % 64 != 0:
        raise AssertionError(
            "BNSH block-sparse attention requires block size to be a multiple of 64, "
            f"got block_size={q_block_size}"
        )
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise AssertionError(
            "BNSH block-sparse attention is forward-only and does not support backward"
        )


def blocksparse_flashattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_sparse_mask: torch.Tensor,
):
    _check_inputs(q, k, v)

    bsz, num_heads, q_num_blocks, q_block_size, head_dim = q.shape
    _, _, k_num_blocks, k_block_size, _ = k.shape
    if block_sparse_mask.shape != (bsz, num_heads, q_num_blocks, k_num_blocks):
        raise AssertionError(
            "BNSH block-sparse attention requires block_sparse_mask shape "
            f"({bsz}, {num_heads}, {q_num_blocks}, {k_num_blocks}), "
            f"got {tuple(block_sparse_mask.shape)}"
        )

    tq = q.reshape(bsz, num_heads, q_num_blocks * q_block_size, head_dim)
    tk = k.reshape(bsz, num_heads, k_num_blocks * k_block_size, head_dim)
    tv = v.reshape(bsz, num_heads, k_num_blocks * k_block_size, head_dim)
    head_mask_type = torch.ones(num_heads, device=q.device, dtype=torch.int32)

    o_res = block_sparse_attn_func_bnsh(
        tq,
        tk,
        tv,
        head_mask_type=head_mask_type,
        base_blockmask=block_sparse_mask.contiguous(),
        p_dropout=0.0,
        deterministic=False,
        is_causal=False,
        return_attn_probs=False,
        m_block_dim=q_block_size,
        n_block_dim=k_block_size,
    )
    return o_res.reshape(bsz, num_heads, q_num_blocks, q_block_size, head_dim)
