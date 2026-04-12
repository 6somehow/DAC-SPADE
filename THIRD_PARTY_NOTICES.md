# Third-Party Notices

This repository vendors or redistributes the following third-party components:

## Block-Sparse-Attention

- Provenance: absorbed into this repository from the standalone
  `Block-Sparse-Attention` project and adapted for in-tree use.
- Local source locations:
  - `python/block_sparse_attn`
  - `csrc/block_sparse_attn`
- Upstream license text is preserved at
  `csrc/block_sparse_attn/LICENSE.block-sparse-attention`.

## CUTLASS

- Used from the `flash-attention` nested dependency at
  `3rdparty/flash-attention/csrc/cutlass`, or from `CUTLASS_ROOT` if
  overridden at build time.
- License text is preserved in the corresponding CUTLASS checkout.

## ThunderKittens

- Used as the `3rdparty/ThunderKittens` submodule, or from
  `THUNDERKITTENS_ROOT` if overridden at build time.

## Model subtrees

The model integrations in `model/Wan2.1`, `model/Wan2.2`, and
`model/hyvideo` retain their own upstream licenses, notices, and distribution
constraints. The root `LICENSE` file does not replace or override those
subtree-specific terms.
