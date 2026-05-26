

### Spade start
import torch
import os
from functools import partial
from math import prod
from spade.core.lower.config import SparseHeadConfig
from spade.engine.engine import SparseAttnEngine, SparseExecutorTable
from spade.utils import _set_cuda_arch

def estimate_func_minmax(block_q, block_k):
    q_max_min = block_q['Max'] + block_q['Min']
    attn_weights_max = torch.matmul(q_max_min,
                                    block_k['Max'].transpose(-1, -2))
    attn_weights_min = torch.matmul(q_max_min,
                                    block_k['Min'].transpose(-1, -2))
    return torch.max(attn_weights_max, attn_weights_min)


def dync_policy_func(start_timestep: int, total_timestep: int,
                     total_block: int, sim_val: torch.Tensor, timestep: int,
                     pipeline: int, layer: int):
    factor = 0.5 if pipeline == 0 else 0.4
    base_block = (1 - (timestep - start_timestep) /
                  (total_timestep - start_timestep)) * factor * total_block
    inter_top_val = (1 - sim_val) * base_block
    diag_width = total_block * 0.05
    intra_top_val = 0.9
    return inter_top_val.to(torch.int32), intra_top_val, int(diag_width)


def select_sparse_engine(backend='torch'):
    """
    An example of how to instantiate the SparseAttnEngine.
    """

    _set_cuda_arch()
    # 1. Initialize the executor table
    exec_table = SparseExecutorTable()

    hidden_dim = 128
    softmax_scale = 1.0 / (hidden_dim**0.5)
    # Form (Frame, Height, Width)
    seqlen3d = (16, 45, 80)

    spatial_config0 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(1, 4, 16),
        block_size_kv=(1, 4, 16),
        fixed_diag_width=40,
        fixed_sink_width=5,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        context_length=None,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    spatial_config1 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(1, 8, 8),
        block_size_kv=(1, 8, 8),
        fixed_diag_width=40,
        fixed_sink_width=5,
        context_length=None,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    temporal_config0 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(16, 1, 4),
        block_size_kv=(16, 1, 4),
        fixed_diag_width=40,
        fixed_sink_width=5,
        context_length=None,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    temporal_config1 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(8, 1, 8),
        block_size_kv=(8, 1, 8),
        fixed_diag_width=40,
        fixed_sink_width=5,
        context_length=None,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    mixed_config0 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(4, 2, 8),
        block_size_kv=(4, 2, 8),
        fixed_diag_width=40,
        fixed_sink_width=5,
        context_length=None,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    mixed_config1 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=(2, 4, 8),
        block_size_kv=(2, 4, 8),
        fixed_diag_width=40,
        fixed_sink_width=5,
        context_length=None,
        inter_select_mode='topk',
        intra_select_mode='topp',
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode='MeanSim',
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    spatial_config_id0 = exec_table.register_config(spatial_config0)
    spatial_config_id1 = exec_table.register_config(spatial_config1)
    temporal_config_id0 = exec_table.register_config(temporal_config0)
    temporal_config_id1 = exec_table.register_config(temporal_config1)
<<<<<<< HEAD
    # mixed_config_id0 = exec_table.register_config(mixed_config0)
    # mixed_config_id1 = exec_table.register_config(mixed_config1)

    timesteps = 40
=======
    mixed_config_id0 = exec_table.register_config(mixed_config0)
    mixed_config_id1 = exec_table.register_config(mixed_config1)

    timesteps = 50
>>>>>>> dev
    num_pipelines = 2
    num_layers = 40
    num_heads = 40

<<<<<<< HEAD
    start_timestep = 0
=======
    start_timestep = 1
>>>>>>> dev
    start_layer = 1

    layout = 'bnsh'

    sparse_engine = SparseAttnEngine(exec_table=exec_table,
                                     seqlen3d=seqlen3d,
                                     dync_policy_func=partial(
                                         dync_policy_func,
                                         start_timestep=start_timestep,
                                         total_timestep=timesteps,
                                         total_block=prod(seqlen3d) // 64),
                                     timesteps=timesteps,
                                     num_pipelines=num_pipelines,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     sparse_start_timestep=start_timestep,
                                     sparse_start_layer=start_layer,
                                     layout=layout,
                                     backend=backend,
<<<<<<< HEAD
                                     is_record_sparse_rate=True)
=======
                                     is_record_sparse_rate=False)
>>>>>>> dev

    print("SparseAttnEngine instantiated successfully.")
    print(f"Executor table: {exec_table}")
    exec_table._log()

    return sparse_engine


### Spade end
spade_engine = select_sparse_engine(backend='cuda')
is_spade_engine = True
