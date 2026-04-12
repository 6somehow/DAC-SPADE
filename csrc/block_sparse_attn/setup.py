import ast
import functools
import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SOURCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_ROOT.parent.parent
PYTHON_ROOT = REPO_ROOT / "python"
PACKAGE_ROOT = PYTHON_ROOT / "block_sparse_attn"
THIRDPARTY_ROOT = REPO_ROOT / "3rdparty"
CUTLASS_ROOT = Path(os.getenv(
    "CUTLASS_ROOT",
    str(THIRDPARTY_ROOT / "flash-attention" / "csrc" / "cutlass"),
)).resolve()
PACKAGE_NAME = "block_sparse_attn"
SKIP_CUDA_BUILD = os.getenv("BLOCK_SPARSE_ATTN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

# setuptools resolves package_dir and --inplace extension paths relative to the
# process cwd, so normalize it for both direct and repo-root invocations.
os.chdir(SOURCE_ROOT)

with open(SOURCE_ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


@functools.lru_cache(maxsize=None)
def cuda_archs():
    archs = os.getenv("BLOCK_SPARSE_ATTN_CUDA_ARCHS")
    if archs is not None:
        return [arch.strip() for arch in archs.split(";") if arch.strip()]

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) == (9, 0):
                return ["90a"]
            if (major, minor) == (8, 9):
                return ["89"]
            if (major, minor) == (8, 6):
                return ["86"]
            if (major, minor) == (8, 0):
                return ["80"]
            return [f"{major}{minor}"]
    except Exception:
        pass

    return ["80", "89", "90a"]


def get_generated_cuda_sources():
    return [str(path)
            for path in sorted((SOURCE_ROOT / "src").glob("flash_*block*.cu"))]


def get_package_version():
    with open(PACKAGE_ROOT / "__init__.py", "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return str(ast.literal_eval(version_match.group(1)))


def get_env_positive_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


def get_nvcc_threads() -> int:
    return get_env_positive_int("NVCC_THREADS", default=1) or 1


def configure_parallel_build():
    nvcc_threads = get_nvcc_threads()
    max_jobs = get_env_positive_int("MAX_JOBS")
    if max_jobs is None:
        import psutil

        cpu_count = os.cpu_count() or 1
        max_jobs = max(1, cpu_count // max(1, nvcc_threads))
        free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        max_jobs = min(max_jobs, max(1, int(free_memory_gb / max(3.0, 2.25 * nvcc_threads))))
        max_jobs = min(max_jobs, get_env_positive_int("BLOCK_SPARSE_ATTN_MAX_JOBS_CAP", default=8) or 8)
    os.environ["MAX_JOBS"] = str(max_jobs)
    return max_jobs, nvcc_threads


def append_nvcc_threads(nvcc_args):
    return nvcc_args + ["--threads", str(get_nvcc_threads())]


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [str(Path(cuda_dir) / "bin" / "nvcc"), "-V"],
        universal_newlines=True,
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    return raw_output, parse(output[release_idx].split(",")[0])


def add_cuda_gencodes(cc_flag, bare_metal_version):
    archs = set(cuda_archs())
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
    if "86" in archs:
        cc_flag += ["-gencode", "arch=compute_86,code=sm_86"]
    if "89" in archs and bare_metal_version >= Version("11.8"):
        cc_flag += ["-gencode", "arch=compute_89,code=sm_89"]
    if "90" in archs and bare_metal_version >= Version("11.8"):
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
    if "90a" in archs:
        if bare_metal_version < Version("12.0"):
            raise RuntimeError("sm90a builds require CUDA 12.0 or newer")
        cc_flag += ["-gencode", "arch=compute_90a,code=sm_90a"]

    ptx_arch = None
    if "90a" in archs or "90" in archs:
        ptx_arch = "90"
    elif "89" in archs:
        ptx_arch = "89"
    elif "86" in archs:
        ptx_arch = "86"
    elif "80" in archs:
        ptx_arch = "80"
    if ptx_arch is not None:
        cc_flag += ["-gencode", f"arch=compute_{ptx_arch},code=compute_{ptx_arch}"]
    return cc_flag


def check_required_headers():
    required_headers = [
        CUTLASS_ROOT / "include" / "cutlass" / "numeric_types.h",
        CUTLASS_ROOT / "include" / "cute" / "tensor.hpp",
    ]
    missing = [str(path) for path in required_headers if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing required CUTLASS headers:\n  - "
            + "\n  - ".join(missing)
            + "\nInitialize flash-attention recursively or set CUTLASS_ROOT to a "
            "CUTLASS checkout."
        )


def cuda_driver_library_dirs():
    candidates = []
    if CUDA_HOME is not None:
        cuda_home = Path(CUDA_HOME)
        candidates.extend([
            cuda_home / "lib64" / "stubs",
            cuda_home / "targets" / "x86_64-linux" / "lib" / "stubs",
            cuda_home / "lib64",
            cuda_home / "targets" / "x86_64-linux" / "lib",
            cuda_home / "compat",
            cuda_home / "compat" / "lib.real",
        ])
    candidates.extend([
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda/compat"),
        Path("/usr/local/cuda/compat/lib.real"),
        Path("/usr/local/nvidia/lib64"),
    ])
    return [
        str(path) for path in candidates
        if (path / "libcuda.so").exists() or (path / "libcuda.so.1").exists()
    ]


def cuda_driver_link_args():
    library_dirs = cuda_driver_library_dirs()
    for library_dir in library_dirs:
        path = Path(library_dir)
        if (path / "libcuda.so").exists():
            return [
                *(f"-L{driver_dir}" for driver_dir in library_dirs),
                "-Wl,--no-as-needed",
                "-lcuda",
                "-Wl,--as-needed",
            ]
        if (path / "libcuda.so.1").exists():
            return [
                *(f"-L{driver_dir}" for driver_dir in library_dirs),
                "-Wl,--no-as-needed",
                "-l:libcuda.so.1",
                "-Wl,--as-needed",
            ]
    raise RuntimeError(
        "Hopper ThunderKittens build requires the CUDA driver library "
        "(libcuda.so or libcuda.so.1), but it was not found. Make sure the "
        "NVIDIA driver libraries are visible, for example through "
        "LD_LIBRARY_PATH or /usr/local/nvidia/lib64."
    )


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        max_jobs, nvcc_threads = configure_parallel_build()
        self.parallel = max_jobs
        print(f"Configuring parallel build: MAX_JOBS={max_jobs}, NVCC_THREADS={nvcc_threads}")
        super().__init__(*args, **kwargs)


ext_modules = []
if not SKIP_CUDA_BUILD:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set and nvcc was not found.")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.7"):
        raise RuntimeError("Block Sparse Attention requires CUDA 11.7 or newer.")
    check_required_headers()
    print(f"Building CUDA architectures: {';'.join(cuda_archs())}")

    thunderkittens_root = Path(os.getenv(
        "THUNDERKITTENS_ROOT",
        str(REPO_ROOT / "3rdparty" / "ThunderKittens"),
    )).resolve()
    thunderkittens_include = thunderkittens_root / "include"
    thunderkittens_prototype = thunderkittens_root / "prototype"
    kittens_header = thunderkittens_include / "kittens.cuh"
    enable_tk = kittens_header.exists()
    build_tk = enable_tk and "90a" in set(cuda_archs())

    nvcc_flags = [
        "-O3",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DTORCH_COMPILE",
    ]
    if enable_tk:
        nvcc_flags.extend([
            f"-I{thunderkittens_include}",
            f"-I{thunderkittens_prototype}",
        ])
    if build_tk:
        nvcc_flags.append("-DKITTENS_HOPPER")
        nvcc_flags.append("-DBLOCK_SPARSE_ATTN_ENABLE_TK")
    cc_flag = add_cuda_gencodes([], bare_metal_version)

    if "90a" in set(cuda_archs()) and not enable_tk:
        print(
            "warning: THUNDERKITTENS_ROOT missing kittens.cuh; "
            "skipping Hopper tk_block_sparse_sm90a.cu build."
        )

    extra_sources = [
        str(SOURCE_ROOT / "flash_api.cpp"),
    ]
    if build_tk:
        extra_sources.append(str(SOURCE_ROOT / "tk_block_sparse_sm90a.cu"))

    ext_modules.append(
        CUDAExtension(
            name="block_sparse_attn_cuda",
            sources=extra_sources + get_generated_cuda_sources(),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"] + (["-DBLOCK_SPARSE_ATTN_ENABLE_TK"] if build_tk else []),
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                SOURCE_ROOT,
                SOURCE_ROOT / "src",
                CUTLASS_ROOT / "include",
                CUTLASS_ROOT / "tools" / "util" / "include",
            ] + ([thunderkittens_include, thunderkittens_prototype] if enable_tk else []),
            library_dirs=(cuda_driver_library_dirs() if build_tk else []),
            extra_link_args=(cuda_driver_link_args() if build_tk else []),
        )
    )


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    description="In-tree Block Sparse Attention kernels for sparseDiTEngine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        where=os.path.relpath(PYTHON_ROOT, SOURCE_ROOT),
        include=["block_sparse_attn", "block_sparse_attn.*"],
    ),
    package_dir={"": os.path.relpath(PYTHON_ROOT, SOURCE_ROOT)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.9",
    install_requires=["torch", "einops"],
    setup_requires=["packaging", "psutil", "ninja"],
)
