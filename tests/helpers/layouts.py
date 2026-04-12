"""Layout helpers for tests.

Layouts:
- bsnh: (batch, seq, heads, head_dim)
- bnsh: (batch, heads, seq, head_dim)
"""

LAYOUTS = ("bsnh", "bnsh")


def apply_layout(tensor, layout: str):
    if layout == "bsnh":
        return tensor
    if layout == "bnsh":
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(f"Unsupported layout '{layout}'")


def infer_layout(shape):
    if len(shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape={shape}")
    b, d1, d2, d3 = shape
    return "bsnh" if d1 >= d2 else "bnsh"
