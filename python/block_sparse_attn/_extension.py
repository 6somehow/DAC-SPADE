from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from types import ModuleType

_EXTENSION_NAME = "block_sparse_attn_cuda"
_extension_module: ModuleType | None = None
_extension_error: Exception | None = None
_cuda_driver_preloaded = False


def _iter_cuda_driver_candidates():
    seen: set[str] = set()

    def add(candidate):
        if not candidate:
            return
        path = str(candidate)
        if path in seen:
            return
        seen.add(path)
        yield path

    for candidate in (
        ctypes.util.find_library("cuda"),
        "libcuda.so.1",
        "libcuda.so",
    ):
        yield from add(candidate)

    cuda_home = os.getenv("CUDA_HOME")
    candidate_dirs = []
    if cuda_home:
        cuda_home_path = Path(cuda_home)
        candidate_dirs.extend([
            cuda_home_path / "compat" / "lib.real",
            cuda_home_path / "compat",
            cuda_home_path / "lib64",
            cuda_home_path / "targets" / "x86_64-linux" / "lib",
        ])
    candidate_dirs.extend([
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda/compat/lib.real"),
        Path("/usr/local/cuda/compat"),
        Path("/usr/local/nvidia/lib64"),
    ])
    for candidate_dir in candidate_dirs:
        for lib_name in ("libcuda.so.1", "libcuda.so"):
            candidate = candidate_dir / lib_name
            if candidate.exists():
                yield from add(candidate)


def _preload_cuda_driver():
    global _cuda_driver_preloaded
    if _cuda_driver_preloaded:
        return
    for lib_name in _iter_cuda_driver_candidates():
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            _cuda_driver_preloaded = True
            return
        except OSError:
            continue


def _should_retry_with_cuda_driver(exc: ImportError) -> bool:
    return "cuGetErrorString" in str(exc)


def _import_extension():
    try:
        return importlib.import_module(_EXTENSION_NAME)
    except ImportError as exc:
        if not _should_retry_with_cuda_driver(exc):
            raise
        _preload_cuda_driver()
        sys.modules.pop(_EXTENSION_NAME, None)
        return importlib.import_module(_EXTENSION_NAME)


def _candidate_paths():
    repo_root = Path(__file__).resolve().parents[2]
    build_roots = [
        repo_root,
        repo_root / "build",
        repo_root / "csrc" / "block_sparse_attn",
        repo_root / "csrc" / "block_sparse_attn" / "build",
    ]
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for build_root in build_roots:
            direct = build_root / f"{_EXTENSION_NAME}{suffix}"
            if direct.exists():
                yield direct
            yield from build_root.glob(f"**/{_EXTENSION_NAME}{suffix}")


def load_extension(required: bool = False) -> ModuleType | None:
    global _extension_module, _extension_error
    if _extension_module is not None:
        return _extension_module

    _preload_cuda_driver()
    try:
        _extension_module = _import_extension()
        return _extension_module
    except ModuleNotFoundError as exc:
        if exc.name != _EXTENSION_NAME:
            raise
        _extension_error = exc
    except ImportError as exc:
        _extension_error = exc

    for candidate in _candidate_paths():
        spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_EXTENSION_NAME] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            if not _should_retry_with_cuda_driver(exc):
                _extension_error = exc
                sys.modules.pop(_EXTENSION_NAME, None)
                continue
            _preload_cuda_driver()
            sys.modules.pop(_EXTENSION_NAME, None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_EXTENSION_NAME] = module
            spec.loader.exec_module(module)
        _extension_module = module
        return module

    if required:
        if _extension_error is not None and not isinstance(_extension_error, ModuleNotFoundError):
            raise ImportError(
                f"Failed to load {_EXTENSION_NAME}: {_extension_error}"
            ) from _extension_error
        raise ModuleNotFoundError(
            "The block_sparse_attn native extension is not built. "
            "Run `python csrc/block_sparse_attn/setup.py build_ext --inplace` "
            "from the repository root, or install the extension package."
        ) from _extension_error
    return None
