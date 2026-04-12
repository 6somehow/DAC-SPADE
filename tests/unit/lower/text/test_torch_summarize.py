from math import prod

import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from spade.core.lower.executor import EstimatorExecutor
from spade.core.lower.torch.torch_summarizer import TorchSummarizerExecutor
from tests.utils.gen_sparse import gen_mixed_text_sparse_config


def test_torch_text_estimator_marks_prompt_context_blocks():
    sparse_config = gen_mixed_text_sparse_config()
    topk_size = 1
    prompt_length = 11
    summarizer = TorchSummarizerExecutor(sparse_config)
    estimator = EstimatorExecutor(sparse_config)
    seqlen = prod(sparse_config.seqlen3d) + sparse_config.context_length
    real_seqlen = prod(sparse_config.seqlen3d) + prompt_length
    num_heads = 2
    q = torch.randn(1, seqlen, num_heads, sparse_config.hidden_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_blocks = sparse_config.num_blocks_q_int + sparse_config.context_q_num_block
    kv_blocks = sparse_config.num_blocks_kv_int + sparse_config.context_kv_num_block
    block_q = torch.empty(
        1,
        q_blocks,
        sparse_config.block_size_q_int,
        num_heads,
        sparse_config.hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_k = torch.empty(
        1,
        kv_blocks,
        sparse_config.block_size_kv_int,
        num_heads,
        sparse_config.hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_v = torch.empty_like(block_k)
    mask = torch.zeros(1, num_heads, q_blocks, kv_blocks, dtype=torch.bool, device="cuda")
    head_ids = torch.arange(num_heads, device="cuda")

    res_dict = summarizer(q, k, v, head_ids, block_q, block_k, block_v, real_seqlen)
    estimator(
        **res_dict,
        mask=mask,
        inter_top_val=topk_size,
        intra_top_val=0.8,
        diag_width=sparse_config.fixed_diag_width,
        realSeqlen=real_seqlen,
    )

    prompt_q_blocks = (prompt_length + sparse_config.block_size_q_int - 1) // sparse_config.block_size_q_int
    prompt_kv_blocks = (prompt_length + sparse_config.block_size_kv_int - 1) // sparse_config.block_size_kv_int
    video_q_blocks = sparse_config.num_blocks_q_int
    video_kv_blocks = sparse_config.num_blocks_kv_int

    valid_kv_end = video_kv_blocks + prompt_kv_blocks
    assert mask[:, :, video_q_blocks : video_q_blocks + prompt_q_blocks, :valid_kv_end].all()
    assert mask[:, :, :, video_kv_blocks:valid_kv_end].all()
    assert not mask[:, :, :, video_kv_blocks + prompt_kv_blocks :].any()
