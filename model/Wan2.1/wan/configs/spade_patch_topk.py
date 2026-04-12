

### Spade start
import torch
import os
from spade.core.lower.config import SparseHeadConfig
from spade.engine.engine import SparseAttnEngine, SparseExecutorTable
from spade.utils import _set_cuda_arch

@torch.compile
def estimate_func_minmax(block_q, block_k):
    q_max_min = block_q['Max'] + block_q['Min']
    attn_weights_max = torch.matmul(q_max_min,
                                    block_k['Max'].transpose(-1, -2))
    attn_weights_min = torch.matmul(q_max_min,
                                    block_k['Min'].transpose(-1, -2))
    return torch.max(attn_weights_max, attn_weights_min)


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

    spatial_block_size_q0 = (1, 4, 16)
    spatial_block_size_kv0 = (1, 4, 16)

    # 2. Generate and register sparse configurations.
    # The order of registration determines the config ID.
    # Index 0 is reserved for dense attention.
    spatial_config0 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=spatial_block_size_q0,
        block_size_kv=spatial_block_size_kv0,
        fixed_diag_width=8,
        fixed_sink_width=8,
        inter_select_mode='topk',
        intra_select_mode=None,
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    spatial_block_size_q1 = (1, 8, 8)
    spatial_block_size_kv1 = (1, 8, 8)

    # 2. Generate and register sparse configurations.
    # The order of registration determines the config ID.
    # Index 0 is reserved for dense attention.
    spatial_config1 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=spatial_block_size_q1,
        block_size_kv=spatial_block_size_kv1,
        fixed_diag_width=8,
        fixed_sink_width=8,
        inter_select_mode='topk',
        intra_select_mode=None,
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    temporal_block_size_q0 = (16, 1, 4)
    temporal_block_size_kv0 = (16, 1, 4)

    temporal_config0 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=temporal_block_size_q0,
        block_size_kv=temporal_block_size_kv0,
        fixed_diag_width=8,
        fixed_sink_width=8,
        inter_select_mode='topk',
        intra_select_mode=None,
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode=None,
        k_intra_summarizer_mode=None,
        symbol_inter_estimator=estimate_func_minmax,
        softmax_scale=softmax_scale,
        quant_dtype=None,
        attn_dtype=torch.bfloat16,
    )

    temporal_block_size_q1 = (8, 1, 8)
    temporal_block_size_kv1 = (8, 1, 8)

    temporal_config1 = SparseHeadConfig(
        hidden_dim=hidden_dim,
        seqlen3d=seqlen3d,
        block_size_q=temporal_block_size_q1,
        block_size_kv=temporal_block_size_kv1,
        fixed_diag_width=8,
        fixed_sink_width=8,
        inter_select_mode='topk',
        intra_select_mode=None,
        q_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        k_inter_summarizer_mode={
            "Max": 'max',
            "Min": 'min'
        },
        q_intra_summarizer_mode=None,
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

    # 3. Define the pattern table dimensions
    timesteps = 50
    num_pipelines = 2
    num_layers = 40
    num_heads = 40

    # 4. Create the pattern table
    # Shape: [timesteps, num_pipelines, num_layer, num_heads, 2]
    # The last dimension holds (executor_id, sparse_value)
    pattern_table = torch.zeros(timesteps,
                                num_pipelines,
                                num_layers,
                                num_heads,
                                dtype=torch.int8,
                                device='cpu')
    inter_top_val = torch.ones_like(pattern_table,
                                    dtype=torch.float32,
                                    device='cpu')
    intra_top_val = torch.ones_like(pattern_table,
                                    dtype=torch.float32,
                                    device='cpu')

    start_timestep = 10
    start_layer = 1

    pattern_table[start_timestep:, :, start_layer:, :] = -1
    inter_top_val[
        start_timestep:, 0, start_layer:, :] = 600 - 14 * torch.arange(
            0, timesteps - start_timestep, dtype=torch.float32).view(-1, 1, 1)
    intra_top_val[start_timestep:, 0, start_layer:, :] = 10

    inter_top_val[
        start_timestep:, 1, start_layer:, :] = 300 - 6 * torch.arange(
            0, timesteps - start_timestep, dtype=torch.float32).view(-1, 1, 1)
    intra_top_val[start_timestep:, 1, start_layer:, :] = 10

    # 5. Instantiate the SparseAttnEngine
    # The engine will compile the registered configs internally.
    sparse_engine = SparseAttnEngine(exec_table=exec_table,
                                     pattern_table=pattern_table,
                                     inter_top_val=inter_top_val,
                                     intra_top_val=intra_top_val,
                                     backend=backend)

    print("SparseAttnEngine instantiated successfully.")
    print(f"Executor table: {exec_table}")
    exec_table._log()
    print(f"Pattern table shape: {pattern_table.shape}")

    return sparse_engine


### Spade end
spade_engine = select_sparse_engine(backend='torch')
