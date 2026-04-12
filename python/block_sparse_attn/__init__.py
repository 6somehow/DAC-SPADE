__version__ = "0.0.1"

try:
    from block_sparse_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ModuleNotFoundError as exc:
    if exc.name != "block_sparse_attn.flash_attn_interface":
        raise

from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    token_streaming_attn_func,
    block_streaming_attn_func,
)

from block_sparse_attn.block_sparse_attn_interface_bnsh import (
    block_sparse_attn_func_bnsh,
)
