import pytest

from tests.helpers.device import require_cuda

require_cuda()

pytestmark = pytest.mark.cuda

import torch
from spade_utils import static_sink_diag_set
from spade.core.lower.utils.common import StaticMaskSet


def test_attn_mask_half_bigbird():
    bsz = 1
    num_heads = 10
    q_block_seqlen = 128
    v_block_seqlen = 128
    diag_width = 4
    attn_scores = torch.zeros((bsz, num_heads, q_block_seqlen, v_block_seqlen),
                              dtype=torch.bool,
                              device='cuda')
    attn_scores_long = torch.zeros(
        (bsz, num_heads, q_block_seqlen, v_block_seqlen),
        dtype=torch.bool,
        device='cuda')
    diag = torch.full((bsz, num_heads),
                      diag_width,
                      dtype=torch.int32,
                      device='cuda')
    StaticMaskSet(attn_scores, 1, 1, diag_width)
    static_sink_diag_set(attn_scores_long, 1, diag)

    torch.testing.assert_close(attn_scores, attn_scores_long)

