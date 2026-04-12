import pytest
import torch

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

from math import prod

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.triton.triton_summarizer import TritonSummarizerExecutor
from spade.core.lower.torch.torch_summarizer import TorchSummarizerExecutor


def _dummy_estimator(block_q, block_k):
    return torch.zeros(1, device=next(iter(block_q.values())).device)


def _build_config() -> SparseHeadConfig:
    seqlen3d = (2, 2, 4)
    hidden_dim = 16
    block_size_q = (1, 1, 2)
    block_size_kv = (1, 1, 2)
    return SparseHeadConfig(
        seqlen3d=seqlen3d,
        hidden_dim=hidden_dim,
        block_size_q=block_size_q,
        block_size_kv=block_size_kv,
        fixed_diag_width=1,
        fixed_sink_width=1,
        inter_select_mode="topk",
        intra_select_mode=None,
        q_inter_summarizer_mode={"Mean": "mean"},
        k_inter_summarizer_mode={"Mean": "mean"},
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=_dummy_estimator,
        softmax_scale=1.0 / (hidden_dim**0.5),
        attn_dtype=torch.float16,
        quant_dtype=None,
    )


def _run_layout(layout: str, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, head_ids: torch.Tensor):
    config = _build_config()
    summarizer = TritonSummarizerExecutor(config, layout=layout)

    bsz, _, _, hidden_dim = q.shape
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

    res = summarizer(q, k, v, head_ids, block_q, block_k, block_v)
    return res, q_num_blocks, kv_num_blocks, num_head_ids, hidden_dim


@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_triton_summarizer_layout_shapes(layout):
    torch.manual_seed(0)
    config = _build_config()
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

    res, q_num_blocks, kv_num_blocks, num_head_ids, hidden_dim = _run_layout(
        layout, q, k, v, head_ids)

    for name, tensor in res["block_inter_q"].items():
        assert tensor.shape == (1, num_head_ids, q_num_blocks, hidden_dim)
    for name, tensor in res["block_inter_k"].items():
        assert tensor.shape == (1, num_head_ids, kv_num_blocks, hidden_dim)


@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_triton_summarizer_matches_torch(layout):
    torch.manual_seed(0)
    config = _build_config()
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

    triton_res, _, _, _, _ = _run_layout(layout, q, k, v, head_ids)
    # build torch blocks explicitly to avoid extra kernel work
    bsz, _, _, hidden_dim = q.shape
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

    torch_res = TorchSummarizerExecutor(config, layout=layout)(
        q, k, v, head_ids, block_q, block_k, block_v, None)

    for key in torch_res["block_inter_q"]:
        torch.testing.assert_close(triton_res["block_inter_q"][key],
                                   torch_res["block_inter_q"][key],
                                   rtol=1e-2,
                                   atol=1e-2)
    for key in torch_res["block_inter_k"]:
        torch.testing.assert_close(triton_res["block_inter_k"][key],
                                   torch_res["block_inter_k"][key],
                                   rtol=1e-2,
                                   atol=1e-2)
