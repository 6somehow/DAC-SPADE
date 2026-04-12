import pytest
import torch

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

from math import prod

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.torch.torch_summarizer import TorchSummarizerExecutor
from tests.utils.gen_sparse import gen_spatial_sparse_selfsim_config



def _run_layout(layout: str, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, head_ids: torch.Tensor):
    config = gen_spatial_sparse_selfsim_config()
    summarizer = TorchSummarizerExecutor(config, layout=layout)

    bsz, seqlen, num_heads, hidden_dim = q.shape
    num_head_ids = head_ids.numel()

    q_num_blocks = config.num_blocks_q_int
    kv_num_blocks = config.num_blocks_kv_int
    q_block = config.block_size_q_int
    kv_block = config.block_size_kv_int

    if layout == "bsnh":
        block_q = torch.empty(bsz,
                              q_num_blocks,
                              q_block,
                              num_head_ids,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)
        block_k = torch.empty(bsz,
                              kv_num_blocks,
                              kv_block,
                              num_head_ids,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)
        block_v = torch.empty(bsz,
                              kv_num_blocks,
                              kv_block,
                              num_head_ids,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)
    else:
        block_q = torch.empty(bsz,
                              num_head_ids,
                              q_num_blocks,
                              q_block,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)
        block_k = torch.empty(bsz,
                              num_head_ids,
                              kv_num_blocks,
                              kv_block,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)
        block_v = torch.empty(bsz,
                              num_head_ids,
                              kv_num_blocks,
                              kv_block,
                              hidden_dim,
                              device="cuda",
                              dtype=torch.float16)

    res = summarizer(q, k, v, head_ids, block_q, block_k, block_v, None)
    return res, q_num_blocks, kv_num_blocks, num_head_ids, config.hidden_dim


@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_torch_summarizer_layout_shapes(layout):
    config =  gen_spatial_sparse_selfsim_config()
    bsz = 1
    num_heads = 4
    num_head_ids = 2
    head_ids = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    seqlen = prod(config.seqlen3d)
    q = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    config.hidden_dim,
                    device="cuda",
                    dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    res, q_num_blocks, kv_num_blocks, num_head_ids, hidden_dim = _run_layout(
        layout, q, k, v, head_ids)

    for name, tensor in res["block_inter_q"].items():
        assert tensor.shape == (1, num_head_ids, q_num_blocks, hidden_dim)
    for name, tensor in res["block_inter_k"].items():
        assert tensor.shape == (1, num_head_ids, kv_num_blocks, hidden_dim)

    assert res["block_intra_q"].shape == (1, num_head_ids, q_num_blocks)
    assert res["block_intra_k"].shape == (1, num_head_ids, kv_num_blocks)


def test_torch_summarizer_layout_values_match():
    config = gen_spatial_sparse_selfsim_config()
    bsz = 1
    num_heads = 4
    head_ids = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    seqlen = prod(config.seqlen3d)
    q = torch.randn(bsz,
                    seqlen,
                    num_heads,
                    config.hidden_dim,
                    device="cuda",
                    dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    res_bsnh, _, _, _, _ = _run_layout("bsnh", q, k, v, head_ids)
    res_bnsh, _, _, _, _ = _run_layout("bnsh", q, k, v, head_ids)

    for key in res_bsnh["block_inter_q"]:
        torch.testing.assert_close(res_bsnh["block_inter_q"][key],
                                   res_bnsh["block_inter_q"][key],
                                   rtol=1e-3,
                                   atol=1e-3)
    for key in res_bsnh["block_inter_k"]:
        torch.testing.assert_close(res_bsnh["block_inter_k"][key],
                                   res_bnsh["block_inter_k"][key],
                                   rtol=1e-3,
                                   atol=1e-3)

    torch.testing.assert_close(res_bsnh["block_intra_q"],
                               res_bnsh["block_intra_q"],
                               rtol=1e-3,
                               atol=1e-3)
    torch.testing.assert_close(res_bsnh["block_intra_k"],
                               res_bnsh["block_intra_k"],
                               rtol=1e-3,
                               atol=1e-3)
