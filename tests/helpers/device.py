import pytest


def require_cuda_sm90a(reason_prefix="Requires sm90a (H100/H800)"):
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        pytest.skip(f"{reason_prefix}; found sm{major}{minor}",
                    allow_module_level=True)


def require_cuda(reason_prefix="CUDA not available"):
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip(reason_prefix, allow_module_level=True)
