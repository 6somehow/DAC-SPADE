from math import prod

import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from spade.core.lower.torch.torch_summarizer import TorchSummarizerExecutor
from spade.core.lower.cuda.cuda_summarizer_hy import CudaHYSummarizerExecutor
from tests.utils.gen_sparse import gen_mixed_text_sparse_config


def test_summarize():
    sparse_config = gen_mixed_text_sparse_config()
    prompt_length = 11
    torch_summarizer = TorchSummarizerExecutor(sparse_config)
    cuda_summarizer = CudaHYSummarizerExecutor(sparse_config)
    seqlen = prod(sparse_config.seqlen3d) + sparse_config.context_length
    real_seqlen = prod(sparse_config.seqlen3d) + prompt_length
    num_heads = 2
    bsz = 1
    hidden = sparse_config.hidden_dim
    q = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    hidden,
                    device='cuda',
                    dtype=torch.bfloat16)
    k = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    hidden,
                    device='cuda',
                    dtype=torch.bfloat16)
    v = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    hidden,
                    device='cuda',
                    dtype=torch.bfloat16)

    block_q = torch.empty(
        bsz,
        sparse_config.num_blocks_q_int + sparse_config.context_q_num_block,
        sparse_config.block_size_q_int,
        num_heads,
        hidden,
        device='cuda',
        dtype=torch.bfloat16)
    block_k = torch.empty(
        bsz,
        sparse_config.num_blocks_kv_int + sparse_config.context_kv_num_block,
        sparse_config.block_size_kv_int,
        num_heads,
        hidden,
        device='cuda',
        dtype=torch.bfloat16)
    block_v = torch.empty_like(block_k)

    head_ids = torch.arange(num_heads, device='cuda')
    torch_res_dict = torch_summarizer(q, k, v, head_ids, block_q, block_k,
                                      block_v, real_seqlen)

    cuda_block_q = torch.empty_like(block_q)
    cuda_block_k = torch.empty_like(block_k)
    cuda_block_v = torch.empty_like(block_v)
    cuda_res_dict = cuda_summarizer(q, k, v, head_ids, cuda_block_q,
                                    cuda_block_k, cuda_block_v, real_seqlen)

    torch.testing.assert_close(cuda_block_q, block_q)
    torch.testing.assert_close(cuda_block_k, block_k)
    torch.testing.assert_close(cuda_block_v, block_v)

    torch.testing.assert_close(
        cuda_res_dict['block_inter_q']['Max']
        [:, :, :-sparse_config.context_q_num_block],
        torch_res_dict['block_inter_q']['Max']
        [:, :, :-sparse_config.context_q_num_block])
    torch.testing.assert_close(
        cuda_res_dict['block_inter_k']['Max']
        [:, :, :-sparse_config.context_kv_num_block],
        torch_res_dict['block_inter_k']['Max']
        [:, :, :-sparse_config.context_kv_num_block])
    torch.testing.assert_close(cuda_res_dict['block_intra_q']
                               [:, :, :-sparse_config.context_q_num_block],
                               torch_res_dict['block_intra_q']
                               [:, :, :-sparse_config.context_q_num_block],
                               atol=1e-2,
                               rtol=1e-2)
    torch.testing.assert_close(cuda_res_dict['block_intra_k']
                               [:, :, :-sparse_config.context_kv_num_block],
                               torch_res_dict['block_intra_k']
                               [:, :, :-sparse_config.context_kv_num_block],
                               atol=1e-2,
                               rtol=1e-2)
