from math import prod

import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from spade.core.lower.torch.torch_summarizer import TorchSummarizerExecutor
from spade.core.lower.cuda.cuda_summarizer import CUDASummarizerExecutor
from tests.utils.gen_sparse import gen_spatial_sparse_config


@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_summarize(layout):
    sparse_config = gen_spatial_sparse_config()
    summarizer = TorchSummarizerExecutor(sparse_config, layout=layout)
    seqlen = prod(sparse_config.seqlen3d)
    num_heads = 4
    bsz = 1
    hidden = sparse_config.hidden_dim

    k = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    hidden,
                    device='cuda',
                    dtype=torch.bfloat16)
    q = torch.randn(bsz,
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
    head_ids = torch.tensor([0, 3], device='cuda', dtype=torch.int64)
    head_ids_len = head_ids.numel()

    if layout == "bsnh":
        torch_block_q = torch.empty(bsz,
                                    sparse_config.num_blocks_q_int,
                                    sparse_config.block_size_q_int,
                                    head_ids_len,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)
        torch_block_k = torch.empty(bsz,
                                    sparse_config.num_blocks_kv_int,
                                    sparse_config.block_size_kv_int,
                                    head_ids_len,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)
        torch_block_v = torch.empty(bsz,
                                    sparse_config.num_blocks_kv_int,
                                    sparse_config.block_size_kv_int,
                                    head_ids_len,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)
    else:
        torch_block_q = torch.empty(bsz,
                                    head_ids_len,
                                    sparse_config.num_blocks_q_int,
                                    sparse_config.block_size_q_int,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)
        torch_block_k = torch.empty(bsz,
                                    head_ids_len,
                                    sparse_config.num_blocks_kv_int,
                                    sparse_config.block_size_kv_int,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)
        torch_block_v = torch.empty(bsz,
                                    head_ids_len,
                                    sparse_config.num_blocks_kv_int,
                                    sparse_config.block_size_kv_int,
                                    hidden,
                                    device='cuda',
                                    dtype=torch.bfloat16)

    torch_res_dict = summarizer(q, k, v, head_ids, torch_block_q,
                                torch_block_k, torch_block_v, None)

    cuda_block_q = torch.empty_like(torch_block_q)
    cuda_block_k = torch.empty_like(torch_block_k)
    cuda_block_v = torch.empty_like(torch_block_v)

    cuda_summarizer = CUDASummarizerExecutor(sparse_config, layout=layout)
    cuda_res_dict = cuda_summarizer(q, k, v, head_ids, cuda_block_q,
                                    cuda_block_k, cuda_block_v, None)

    for key in torch_res_dict.keys():
        torch.testing.assert_close(torch_res_dict[key],
                                   cuda_res_dict[key],
                                   rtol=1e-2,
                                   atol=1e-2)
