__all__ = ["compile", "E2EExecutor", "SparseHeadConfig"]


def __getattr__(name):
    if name == "compile":
        from spade.core.transform.compile import compile as _compile
        return _compile
    if name == "E2EExecutor":
        from spade.core.lower.executor import E2EExecutor as _executor
        return _executor
    if name == "SparseHeadConfig":
        from spade.core.lower.config import SparseHeadConfig as _config
        return _config
    raise AttributeError(name)
