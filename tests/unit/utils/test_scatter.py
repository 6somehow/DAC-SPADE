import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch
from spade_utils import scatter_mask


def inter_topk(estimates: torch.Tensor, mask: torch.Tensor, topk_size: int):
    topk_size = min(topk_size, estimates.shape[-1])
    index = torch.topk(estimates, topk_size, dim=-1).indices
    mask.scatter_(-1, index, True)


def inter_topk_tensor(estimates: torch.Tensor, mask: torch.Tensor,
                      topk_size: torch.Tensor):
    topk_size_val = min(topk_size.max().item(), estimates.shape[-1])
    index = torch.topk(estimates, topk_size_val,
                       dim=-1).indices.to(torch.int32)
    scatter_mask(mask, index, topk_size)


def test_scatter_tensor():
    bsz, num_heads, q_bseqlen, k_bseqlen = 1, 4, 16, 16
    estimates = torch.randn(bsz,
                            num_heads,
                            q_bseqlen,
                            k_bseqlen,
                            dtype=torch.float16,
                            device='cuda')
    mask = torch.zeros_like(estimates, dtype=torch.bool, device='cuda')
    mask_std = torch.zeros_like(estimates, dtype=torch.bool, device='cuda')

    topk_size = 4
    topk_t = torch.full((bsz, num_heads),
                        topk_size,
                        dtype=torch.int32,
                        device='cuda')

    inter_topk_tensor(estimates, mask, topk_t)
    inter_topk(estimates, mask_std, topk_size)
    torch.testing.assert_close(mask, mask_std)
