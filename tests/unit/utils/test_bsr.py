import pytest
import torch

from tests.helpers.device import require_cuda
from spade.core.lower.utils.common import Mask2BSRTriton, Mask2BSRCUDA

require_cuda()

pytestmark = pytest.mark.cuda


def test_mask_to_bsr():
    bsz = 1
    num_heads = 4
    q_num_blocks = 16
    k_num_blocks = 16
    sparsity = 0.3
    device = "cuda"

    sparse_mask = torch.rand(bsz,
                             num_heads,
                             q_num_blocks,
                             k_num_blocks,
                             device=device) > sparsity

    bsr_triton, num_triton = Mask2BSRTriton(sparse_mask)
    bsr_torch, num_torch = Mask2BSRCUDA(sparse_mask)

    assert torch.equal(num_triton, num_torch)
    assert torch.equal(bsr_triton, bsr_torch)
