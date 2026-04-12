from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.executor import E2EExecutor, SummarizerExecutor, EstimatorExecutor, AttentionExecutor
from spade.core.lower.torch import TorchSummarizerExecutor
from spade.core.lower.triton import TritonSummarizerExecutor
from spade.core.lower.cuda import CUDASummarizerExecutor


def compile(config: SparseHeadConfig, backend: str, layout:str) -> E2EExecutor:
    assert layout in ('bsnh', 'bnsh'), "Layout must be either 'bsnh' or 'bshn'"
    summarizer = None
    if backend == "torch":
        summarizer = TorchSummarizerExecutor(config, layout)
    elif backend == "cuda":
        if config.context_length == None:
            summarizer = CUDASummarizerExecutor(config, layout)
        else:
            summarizer = CUDASummarizerExecutor(config, layout)

            # summarizer = CudaHYSummarizerExecutor(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    estimator = EstimatorExecutor(config,layout)
    attention = AttentionExecutor(config, layout, backend=backend)
    return E2EExecutor(config, summarizer, estimator, attention)
