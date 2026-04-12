# Install Guide

This guide covers a clean source install of `sparseDiTEngine` for development
and local experimentation.

## Prerequisites

- Linux
- NVIDIA GPU and CUDA toolkit with `nvcc` available on `PATH`
- Python 3.9 to 3.12 recommended
- Git

Notes:

- `block_sparse_attn` native kernels require CUDA and a working PyTorch CUDA
  install.
- Hopper pure block-sparse fast paths additionally depend on
  `3rdparty/ThunderKittens`.
- Model subdirectories under `model/` may have extra requirements beyond the
  core engine.

## 1. Clone the repository

```bash
git clone https://github.com/6somehow/sparseDiTEngine.git
cd sparseDiTEngine
```

Initialize the remaining external dependencies used by this repo:

```bash
git submodule update --init --recursive 3rdparty/ThunderKittens 3rdparty/flash-attention
```

The recursive update is required because `block_sparse_attn` builds against the
CUTLASS checkout nested under `3rdparty/flash-attention/csrc/cutlass`.

## 2. Create a Python environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Or use conda/uv/your preferred environment manager if your team already has a
standard setup.

## 3. Install Python dependencies

Install the common Python build/runtime dependencies first:

```bash
python -m pip install packaging ninja psutil pytest
```

Install PyTorch with CUDA support using the method appropriate for your system.
The editable package does not ask pip to install PyTorch, which avoids
overriding NVIDIA or cluster-provided PyTorch wheels such as
`torch==2.7.0a0+*.nv*`. Then install Triton and the core Python package:

```bash
python -m pip install triton
python -m pip install -e python
```

If you prefer source-tree imports without an editable install, you can skip the
last command and instead use:

```bash
export PYTHONPATH="$PWD/python"
```

## 4. Build native extensions

### 4a. Build `spade_utils`

```bash
cd csrc/spade_utils
python setup.py build_ext --inplace
cd ../..
```

### 4b. Build `block_sparse_attn`

By default the build uses CUTLASS from
`3rdparty/flash-attention/csrc/cutlass` and looks for ThunderKittens at
`3rdparty/ThunderKittens`. The Hopper fast path is only built when
`kittens.cuh` is present, otherwise the build skips the Hopper kernel. Override
the paths if needed:

```bash
export CUTLASS_ROOT="${CUTLASS_ROOT:-$PWD/3rdparty/flash-attention/csrc/cutlass}"
export THUNDERKITTENS_ROOT="${THUNDERKITTENS_ROOT:-$PWD/3rdparty/ThunderKittens}"
```

Optional: select target architectures explicitly. If you do not have
ThunderKittens available, omit `90a` so the Hopper kernel is not attempted.

Examples:

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89;90a"
```

For an H100-only build that should use `tk_block_sparse_sm90a.cu`, make sure
ThunderKittens is populated and build for `90a`:

```bash
git submodule update --init --recursive 3rdparty/ThunderKittens
export THUNDERKITTENS_ROOT="$PWD/3rdparty/ThunderKittens"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="90a"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
```

If importing `block_sparse_attn_cuda` reports an undefined `cuGetErrorString`
symbol, remove the old `.so` and rebuild after this step; the Hopper TK build
links the CUDA driver library.

You can verify the rebuilt extension records the CUDA driver dependency with:

```bash
readelf -d python/block_sparse_attn_cuda*.so | grep -E 'libcuda|NEEDED'
```

If the build log still shows `-gencode arch=compute_80` or
`arch=compute_89` on an H100-only build, `BLOCK_SPARSE_ATTN_CUDA_ARCHS` was not
set in the shell running `setup.py`.

Then build:

```bash
cd csrc/block_sparse_attn
python setup.py build_ext --inplace
cd ../..
```

### 4c. Optional: build `summarizer_hy`

Only needed if you use the Hunyuan Video summarizer path:

```bash
cd csrc/summarizer_hy
python setup.py build_ext --inplace
cd ../..
```

## 5. Smoke test the install

If you used the editable install:

```bash
python -c "import block_sparse_attn; import spade; print('imports ok')"
```

If you are using source-tree imports:

```bash
PYTHONPATH="$PWD/python" python -c "import block_sparse_attn; import spade; print('imports ok')"
```

You can also run a small test target:

```bash
pytest -q tests/unit/lower/e2e/test_torch_e2e.py::test_attention_executor_prefers_bnsh_on_sm8x
```

## 6. Optional model-specific setup

The root install only covers the core engine and native attention backends.
Model integrations may require extra packages or checkpoints.

See:

- `model/Wan2.1/INSTALL.md`
- `model/Wan2.2/INSTALL.md`
- `model/hyvideo/README.md`

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Install a CUDA-enabled PyTorch build into the same Python environment used for
building and testing.

### `nvcc` not found

Make sure the CUDA toolkit is installed and `CUDA_HOME` or `PATH` points to the
toolkit containing `bin/nvcc`.

### ThunderKittens include errors on Hopper builds

Set:

```bash
export THUNDERKITTENS_ROOT=/absolute/path/to/ThunderKittens
```

If you are not targeting Hopper, you can also avoid the tk kernel by using:

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89"
```

### Import works in one Python but not another

Make sure `python`, `pip`, and `pytest` all come from the same environment:

```bash
which python
which pip
which pytest
```
