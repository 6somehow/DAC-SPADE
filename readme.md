# sparseDiTEngine

`sparseDiTEngine` is a sparse attention runtime for diffusion-transformer style
video workloads. The repo ships:

- the core `spade` Python package under [`python/spade`](python/spade)
- an in-tree `block_sparse_attn` package under [`python/block_sparse_attn`](python/block_sparse_attn)
- native CUDA/CUTLASS/ThunderKittens kernels under [`csrc`](csrc)
- model integrations for Wan 2.1, Wan 2.2, and Hunyuan Video under [`model`](model)

For step-by-step environment setup, see [`INSTALL.md`](INSTALL.md).

## Repository layout

- `python/block_sparse_attn`: first-party block sparse attention Python API
- `csrc/block_sparse_attn`: native block sparse attention extension sources
- `python/spade`: sparse engine APIs and model-facing integration code
- `tests`: unit and integration tests

## Build the native block sparse extension

The native extension lives in-tree and can be built directly from this
repository.

```bash
export THUNDERKITTENS_ROOT="${THUNDERKITTENS_ROOT:-$PWD/3rdparty/ThunderKittens}"
python csrc/block_sparse_attn/setup.py build_ext --inplace
```

If you need to control compilation targets, set
`BLOCK_SPARSE_ATTN_CUDA_ARCHS`, for example:

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89;90a"
```

## Python package usage

For source-tree usage:

```bash
export PYTHONPATH="$PWD/python"
python -c "import block_sparse_attn; import spade"
```

The `block_sparse_attn` package preserves the historical import name while now
living directly inside this repository.

## Notes

- `block_sparse_attn_func_bnsh` is the single BNSH forward entrypoint.
- Hopper-only pure block-sparse forward kernels are dispatched through the same
  `block_sparse_attn` package; there is no separate legacy Hopper extension.
- Model subdirectories under `model/` keep their own upstream licenses and
  notices.
