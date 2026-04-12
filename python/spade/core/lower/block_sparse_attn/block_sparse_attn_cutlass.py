import torch

from block_sparse_attn import block_sparse_attn_func

attention_layout = "bsnh"

def blocksparse_flashattn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_sparse_mask: torch.Tensor):
    bsz, q_num_blocks, q_block_size, num_heads, head_dim = q.shape
    _, k_num_blocks, k_block_size, _, _ = k.shape
    # The block sparse kernel is implemented for block size 128
    assert q_block_size == 128 and k_block_size == 128, "Block-sparse attention only supports block size of 128"

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3]

    q_flat = q.flatten(1, 2).squeeze_(0)
    k_flat = k.flatten(1, 2).squeeze_(0)
    v_flat = v.flatten(1, 2).squeeze_(0)

    q_len = q_flat.size(0)
    k_len = k_flat.size(0)

    q_cu_seq_lens = torch.tensor([0, q_len],
                                 dtype=torch.int32,
                                 device=q.device)
    k_cu_seq_lens = torch.tensor([0, k_len],
                                 dtype=torch.int32,
                                 device=q.device)
    head_mask_type = torch.tensor([1 for _ in range(num_heads)],
                                  device=q.device,
                                  dtype=torch.int32)

    attn_output = block_sparse_attn_func(
        q_flat,
        k_flat,
        v_flat,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        block_sparse_mask.contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=False,
    )
    attn_output = attn_output.view(bsz, q_num_blocks, q_block_size, num_heads,
                                   head_dim)

    return attn_output
