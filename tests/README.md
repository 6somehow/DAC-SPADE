# Tests

The test suite is organized by project subsystem. It intentionally contains only
pytest tests and shared test helpers; benchmark scripts, historical reference
implementations, generated outputs, and exploratory notebooks/scripts live
outside this tree.

## Structure

- `tests/helpers/` – shared utilities (layouts, device guards, tensor factories)
- `tests/unit/`
  - `attention/` – block-sparse attention backends (tk/cutlass/flex/fa3)
  - `engine/` – engine orchestration and tables
  - `lower/`
    - `e2e/` – end-to-end lower pipeline tests
    - `summarizer/` – summarizer backends (torch/triton/cuda)
    - `text/` – text/variable seqlen paths
  - `utils/` – reorder/mask/scatter helpers
- `tests/utils/` – compact sparse-config factories used by tests

## Markers

- `cuda` – requires CUDA runtime
- `sm90a` – requires H100/H800 (sm90a)
- `fa3` – requires flash-attn 3
- `layout(name)` – layout-specific tests

## Layouts

Supported layouts:
- `bsnh` – (batch, seq, heads, head_dim)
- `bnsh` – (batch, heads, seq, head_dim)

Use `tests/helpers/layouts.py` helpers to generate or convert layouted tensors.

## Running

- All tests: `pytest tests`
- Unit tests only: `pytest tests/unit`
- CUDA-only: `pytest -m cuda tests/unit`

If a native optional extension is not built, the affected tests skip at runtime.
If PyTorch reports a missing `libc10`/`libtorch` dependency, activate the same
environment used to build PyTorch extensions or add the PyTorch library
directory to `LD_LIBRARY_PATH`.
