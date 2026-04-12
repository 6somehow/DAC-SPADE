from spade.engine.engine import SparseAttnEngine


class _DummyExecTable:

    def __init__(self):
        self.compile_backends = []

    def compile(self, backend: str, layout: str) -> None:
        self.compile_backends.append((backend, layout))

    def __len__(self) -> int:
        return 1


def _dummy_policy(sim_val, timestep, pipeline, layer):
    return 1, 0.99, 1


def test_sparse_engine_instances_are_isolated():
    cuda_table = _DummyExecTable()
    torch_table = _DummyExecTable()

    engine_cuda = SparseAttnEngine(
        exec_table=cuda_table,
        dync_policy_func=_dummy_policy,
        timesteps=4,
        num_pipelines=1,
        num_layers=2,
        num_heads=2,
        seqlen3d=(1, 1, 1),
        layout="bnsh",
        backend="cuda",
    )
    engine_torch = SparseAttnEngine(
        exec_table=torch_table,
        dync_policy_func=_dummy_policy,
        timesteps=4,
        num_pipelines=1,
        num_layers=2,
        num_heads=2,
        seqlen3d=(1, 1, 1),
        layout="bnsh",
        backend="torch",
    )

    assert engine_cuda is not engine_torch
    assert engine_cuda.backend == "cuda"
    assert engine_torch.backend == "torch"
    assert cuda_table.compile_backends == [("cuda", "bnsh")]
    assert torch_table.compile_backends == [("torch", "bnsh")]

    engine_cuda._advance_step()
    assert (engine_cuda.now_timestep, engine_cuda.now_pipeline,
            engine_cuda.now_layer) == (0, 0, 1)
    assert (engine_torch.now_timestep, engine_torch.now_pipeline,
            engine_torch.now_layer) == (0, 0, 0)
