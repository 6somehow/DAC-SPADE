import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch

from math import prod
from spade.core.lower.utils.reorder import (
    ReorderBackTensorTextBSNH,
    ReorderTensorTextBSNH,
    ReorderBackTensorTextBNSH,
    ReorderTensorTextBNSH,
)
from spade.core.lower.utils.utils import getNumBlocks


@pytest.mark.parametrize("seqlenTuple,qBlockSizeTuple,cxtSeqlen", [
    ((4, 8, 8), (1, 4, 4), 16),
    ((8, 4, 8), (4, 1, 4), 16),
])
@pytest.mark.parametrize("layout", ["bsnh", "bnsh"])
def test_order(seqlenTuple, qBlockSizeTuple, cxtSeqlen, layout):
    bsz = 1
    head_dim = 128
    num_heads = 8
    numBlockTuple = getNumBlocks(seqlenTuple, qBlockSizeTuple)

    numHeadList = [0, 1, 3, 5, 7]

    vseqlen = prod(seqlenTuple)
    seqlen = vseqlen + cxtSeqlen
    blockSize = prod(qBlockSizeTuple)
    prompt_length = 11
    realSeqlen = prompt_length + vseqlen

    assert cxtSeqlen % blockSize == 0, f"text_length {cxtSeqlen} must be divisible by blockSize {blockSize}"

    cxtBlocks = cxtSeqlen // blockSize
    numBlocks = prod(numBlockTuple)
    k = torch.randn((bsz, seqlen, num_heads, head_dim),
                    dtype=torch.bfloat16,
                    device='cuda')
    num_head_idx = len(numHeadList)
    if layout == "bsnh":
        k_reordered = torch.zeros(
            (bsz, numBlocks + cxtBlocks, blockSize, num_head_idx, head_dim),
            dtype=k.dtype,
            device=k.device)
    else:
        k_reordered = torch.zeros(
            (bsz, num_head_idx, numBlocks + cxtBlocks, blockSize, head_dim),
            dtype=k.dtype,
            device=k.device)
    numHeadListt = torch.tensor(numHeadList,
                                dtype=torch.int64,
                                device=k.device)
    k_reordered_back = torch.empty_like(k)

    if layout == "bsnh":
        ReorderTensorTextBSNH(k_reordered, k, numHeadListt, 0, seqlenTuple,
                              qBlockSizeTuple, numBlockTuple, realSeqlen)
        ReorderBackTensorTextBSNH(k_reordered_back, k_reordered, numHeadListt,
                                  seqlenTuple, qBlockSizeTuple, numBlockTuple)
        k_reordered = k_reordered.permute(0, 3, 1, 2, 4).flatten(-3, -2)
    else:
        ReorderTensorTextBNSH(k_reordered, k, numHeadListt, 0, seqlenTuple,
                              qBlockSizeTuple, numBlockTuple, realSeqlen)
        ReorderBackTensorTextBNSH(k_reordered_back, k_reordered, numHeadListt,
                                  seqlenTuple, qBlockSizeTuple, numBlockTuple)
        k_reordered = k_reordered.flatten(-3, -2)
    for i in numHeadList:
        try:
            assert (k_reordered[:, :, numBlocks * blockSize +
                                prompt_length:] == 0).all()
            torch.testing.assert_close(k[:, :realSeqlen, i, :],
                                       k_reordered_back[:, :realSeqlen, i, :])
        except AssertionError as e:
            print(f'head {i} failed')
            raise e
