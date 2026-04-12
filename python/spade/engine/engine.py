import torch
from typing import Optional, Callable, Union, Tuple, Dict, List

from spade.engine.dense_attention import flash_attention
from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.executor import E2EExecutor
from spade.core.transform.compile import compile
from spade_utils import cossim


class SparseExecutorTable:

    def __init__(self) -> None:
        self.sparse_executors: List[Union[Callable, E2EExecutor]] = [
            flash_attention,
        ]
        self.sparse_configs: List[Optional[SparseHeadConfig]] = [
            None,
        ]
        self.attn_dtype: Optional[torch.dtype] = None

    def register_config(self, config: SparseHeadConfig) -> int:
        if not isinstance(config, SparseHeadConfig):
            raise TypeError("Only SparseHeadConfig objects can be registered.")
        self.sparse_configs.append(config)
        if self.attn_dtype is None:
            self.attn_dtype = config.attn_dtype
        else:
            if self.attn_dtype != config.attn_dtype:
                raise TypeError(
                    "The attn_dtype of SparseHeadConfig must be the same.")
        return len(self.sparse_configs) - 1

    def compile(self, backend: str, layout: str) -> None:
        for i, config in enumerate(self.sparse_configs):
            if i != 0:
                self.sparse_executors.append(compile(config, backend, layout))

    def get_config(self, index: int) -> Optional[SparseHeadConfig]:
        if 0 < index < len(self.sparse_configs):
            return self.sparse_configs[index]
        return None

    def __getitem__(self, index: int) -> Optional[Callable]:
        if 0 <= index < len(self.sparse_executors):
            return self.sparse_executors[index]
        else:
            raise IndexError(f'index {index} out of range')

    def __len__(self) -> int:
        return len(self.sparse_configs)

    def __str__(self):
        return f"SparseExecutorTable({len(self.sparse_configs)} sparse configs)"

    def _log(self):
        for i, config in enumerate(self.sparse_configs):
            if i != 0:
                print(f"sparse config {i}: {config}")


# def example_dync_policy_func(bssim: torch.Tensor, timestep: int, pipeline: int,
#                              layer: int) -> torch.Tensor:
#     return bssim[timestep, pipeline, layer, 0]


class SparseAttnEngine:
    def __init__(
        self,
        exec_table: SparseExecutorTable,
        dync_policy_func: Callable,
        timesteps: int,
        num_pipelines: int,
        num_layers: int,
        num_heads: int,
        seqlen3d: Tuple[int, int, int],
        layout: str,
        sparse_start_timestep: Union[int, list[int]] = 0,
        sparse_start_layer: int = 0,
        context_length: Optional[int] = None,
        backend: str = 'torch',
        is_record_sparse_rate: bool = False,
    ):
        """
        sparse_rt_table: shape [timesteps, num_pipelines, num_layers, num_heads, 2]
        last dim contains sparse executor index and sparse value.
        """
        self.seqlen3d = seqlen3d
        self.exec_table = exec_table
        self.backend = backend
        self.dync_policy_func = dync_policy_func
        self.timesteps, self.num_pipelines, self.num_layers, self.num_heads = timesteps, num_pipelines, num_layers, num_heads
        self.exec_table.compile(self.backend, layout)
        self.max_num_patterns = len(self.exec_table)
        self.now_timestep, self.now_pipeline, self.now_layer = 0, 0, 0
        self.sparse_start_layer = sparse_start_layer
        if isinstance(sparse_start_timestep, int):
            self.sparse_timestep = tuple(
                range(sparse_start_timestep, timesteps))
        else:
            self.sparse_timestep = tuple(sparse_start_timestep)
        self.context_length = context_length
        self.eval_block_cache: Dict[Tuple[str, Tuple[int, int, int]],
                                    torch.Tensor] = {}
        self.record_sparse_rate = is_record_sparse_rate
        if self.record_sparse_rate:
            self.sparse_rate = torch.zeros(self.timesteps,
                                           self.num_pipelines,
                                           self.num_layers,
                                           self.num_heads,
                                           dtype=torch.float16,
                                           device='cpu')
            self.sparse_pattern = torch.zeros(self.timesteps,
                                              self.num_pipelines,
                                              self.num_layers,
                                              self.num_heads,
                                              dtype=torch.int8,
                                              device='cpu')

            self.record_sim = torch.zeros(self.timesteps,
                                          self.num_pipelines,
                                          self.num_layers,
                                          self.num_heads,
                                          dtype=torch.float16,
                                          device='cpu')

        for i in range(1, len(self.exec_table)):
            assert self.exec_table.get_config(
                i
            ).context_length == self.context_length, f"the text length of sparse pattern {i} must equal to {self.context_length}"

    def reset(self) -> None:
        self.now_timestep, self.now_pipeline, self.now_layer = 0, 0, 0
        if self.record_sparse_rate:
            self.sparse_rate.zero_()
            self.sparse_pattern.zero_()
            self.record_sim.zero_()

    def eval_self_block(self, q: torch.Tensor, sparse_id: int):
        sparse_config = self.exec_table.get_config(sparse_id)

        seqlen3d = sparse_config.seqlen3d

        qNumBlocks3d = sparse_config.num_blocks_q
        qBlockSize3d = sparse_config.block_size_q

        text_seqlen = 0 if self.context_length is None else self.context_length

        block_cossim_q = self.eval_block_cache.get(('q', qBlockSize3d))
        if block_cossim_q is None:
            block_cossim_q = cossim(q, seqlen3d, qBlockSize3d, qNumBlocks3d,
                                    text_seqlen)
            self.eval_block_cache[('q', qBlockSize3d)] = block_cossim_q

        return block_cossim_q

    def _attn_head_select(self, q: torch.Tensor, k: torch.Tensor):
        attn_head_lists: Dict[int, Tuple[List[int], List[float]]] = {}
        sim: List[torch.Tensor] = [0] * (self.max_num_patterns - 1)
        for sparse_id in range(1, self.max_num_patterns):
            sim_val = self.eval_self_block(q, sparse_id).squeeze(0)
            sim[sparse_id - 1] = sim_val

        sim = torch.stack(sim, dim=0)
        sim_selection = sim.argmax(dim=0) + 1
        sim_max_val = sim.max(dim=0).values

        for head_id in range(self.num_heads):
            sparse_id = sim_selection[head_id].item()
            sim_val = sim_max_val[head_id].item()
            if sparse_id not in attn_head_lists:
                attn_head_lists[int(sparse_id)] = (
                    [],
                    [],
                )
            attn_head_lists[sparse_id][0].append(head_id)
            attn_head_lists[sparse_id][1].append(sim_val)

        exec_parameters_calls = {}
        for sparse_id in attn_head_lists.keys():
            sim_val = torch.tensor(attn_head_lists[sparse_id][1],
                                   dtype=torch.float32,
                                   device=q.device)
            head_ids = attn_head_lists[sparse_id][0]
            inter_top_val, intra_top_val, diag_width = self.dync_policy_func(
                sim_val=sim_val,
                timestep=self.now_timestep,
                pipeline=self.now_pipeline,
                layer=self.now_layer)
            if isinstance(inter_top_val, torch.Tensor):
                inter_top_val = inter_top_val.unsqueeze(0).to(q.device)
            if isinstance(diag_width, torch.Tensor):
                diag_width = diag_width.unsqueeze(0).to(q.device)
            exec_parameters_calls[sparse_id] = {
                'head_ids': head_ids,
                'inter_top_val': inter_top_val,
                'intra_top_val': intra_top_val,
                'diag_width': diag_width,
            }

        if self.record_sparse_rate:
            self.sparse_pattern[self.now_timestep, self.now_pipeline,
                                self.now_layer] = sim_selection
            self.record_sim[self.now_timestep, self.now_pipeline,
                            self.now_layer] = sim_max_val

        return exec_parameters_calls

    def _generate_flash_attn_args(self, q: torch.Tensor, k: torch.Tensor):
        bsz, seqlen_q, _, _ = q.shape
        _, seqlen_kv, _, _ = k.shape

        cu_seqlens_q = torch.arange(0, (bsz + 1) * seqlen_q,
                                    step=seqlen_q,
                                    dtype=torch.int32,
                                    device=q.device)
        max_seqlen_q = seqlen_q

        cu_seqlens_kv = torch.arange(0, (bsz + 1) * seqlen_kv,
                                     step=seqlen_kv,
                                     dtype=torch.int32,
                                     device=k.device)
        max_seqlen_kv = seqlen_kv

        return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    def _maybe_cast_inputs(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor,
                                                     torch.dtype]:
        q_dtype = q.dtype
        if q_dtype != self.exec_table.attn_dtype:
            q = q.to(self.exec_table.attn_dtype)
            k = k.to(self.exec_table.attn_dtype)
            v = v.to(self.exec_table.attn_dtype)
        return q, k, v, q_dtype

    def _advance_step(self) -> None:
        self.now_layer += 1
        if self.now_layer == self.num_layers:
            self.now_layer = 0
            self.now_pipeline += 1
            if self.now_pipeline == self.num_pipelines:
                self.now_pipeline = 0
                self.now_timestep += 1

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
    ) -> torch.Tensor:
        if self.context_length is not None:
            real_length = cu_seqlens_q[1].item()
        else:
            real_length = None

        q, k, v, qdtype = self._maybe_cast_inputs(q, k, v)

        self.eval_block_cache = {}
        res_o = None

        if self.now_timestep in self.sparse_timestep and self.now_layer >= self.sparse_start_layer:
            res_o = torch.empty_like(q)
            exec_parameters_calls = self._attn_head_select(q, k)

            for sparse_id, parameters in exec_parameters_calls.items():

                sparse_rate = self.exec_table[sparse_id](
                    q=q,
                    k=k,
                    v=v,
                    res_o=res_o,
                    realSeqlen=real_length,
                    **parameters)

                if self.record_sparse_rate:
                    for head_id in parameters['head_ids']:
                        self.sparse_rate[self.now_timestep, self.now_pipeline,
                                         self.now_layer, head_id] = sparse_rate

        else:
            if cu_seqlens_q is None:
                cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = self._generate_flash_attn_args(
                    q, k)


            res_o = flash_attention(q,k,v)

        self._advance_step()
        return res_o.to(qdtype)
