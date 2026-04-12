import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from math import prod
from spade.core.lower.utils.reorder import (
    ReorderTensorBSNH,
    ReorderBackTensorBSNH,
    ReorderTensorBNSH,
    ReorderBackTensorBNSH,
)
from spade.core.lower.utils.utils import getNumBlocks


@pytest.mark.parametrize("seqlenTuple,qBlockSizeTuple", [
    ((2, 8, 16), (1, 8, 16)),
    ((4, 8, 16), (1, 4, 16)),
    ((8, 4, 16), (4, 1, 16)),
    ((4, 8, 8), (1, 2, 4)),
    ((5, 8, 8), (2, 4, 4)),
])

@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_order(seqlenTuple, qBlockSizeTuple, layout):
    bsz = 1
    head_dim = 128
    num_heads = 8
    numBlockTuple = getNumBlocks(seqlenTuple, qBlockSizeTuple)

    numHeadList = [0, 1, 3, 5, 7]

    seqlen = prod(seqlenTuple)
    numBlocks = prod(numBlockTuple)
    blockSize = prod(qBlockSizeTuple)

    k = torch.randn((bsz, seqlen, num_heads, head_dim),
                    dtype=torch.bfloat16,
                    device='cuda')
    num_head_idx = len(numHeadList)
    if layout == "bsnh":
        k_reordered = torch.empty(
            (bsz, numBlocks, blockSize, num_head_idx, head_dim),
            dtype=k.dtype,
            device=k.device)
    else:
        k_reordered = torch.empty(
            (bsz, num_head_idx, numBlocks, blockSize, head_dim),
            dtype=k.dtype,
            device=k.device)
    numHeadListt = torch.tensor(numHeadList,
                                dtype=torch.int64,
                                device=k.device)
    k_reordered_back = torch.empty_like(k)

    if layout == "bsnh":
        ReorderTensorBSNH(k_reordered, k, numHeadListt, 0, seqlenTuple,
                          qBlockSizeTuple, numBlockTuple)
        ReorderBackTensorBSNH(k_reordered_back, k_reordered, numHeadListt,
                              seqlenTuple, qBlockSizeTuple, numBlockTuple)
    else:
        ReorderTensorBNSH(k_reordered, k, numHeadListt, 0, seqlenTuple,
                          qBlockSizeTuple, numBlockTuple)
        ReorderBackTensorBNSH(k_reordered_back, k_reordered, numHeadListt,
                              seqlenTuple, qBlockSizeTuple, numBlockTuple)
    for i in numHeadList:
        try:
            torch.testing.assert_close(k[:, :, i, :], k_reordered_back[:, :,
                                                                       i, :])
        except AssertionError as e:
            print(f'head {i} failed')
            raise e
