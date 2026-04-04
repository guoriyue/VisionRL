"""KV Cache append Triton kernel.

Writes new K/V tokens into a pre-allocated contiguous cache buffer
at specified position indices. One program per (batch, head, new_token) triple.
"""

import triton
import triton.language as tl


@triton.jit
def kv_cache_append_kernel(
    New_ptr,        # [B, H, S_new, D]
    Cache_ptr,      # [B, H, S_max, D]
    Positions_ptr,  # [S_new]
    H, S_new, S_max, D,
    stride_nb, stride_nh, stride_ns, stride_nd,
    stride_cb, stride_ch, stride_cs, stride_cd,
    BLOCK_D: tl.constexpr,
):
    """Write new K or V tokens into cache at given positions.

    Grid: (B * H * S_new,)
    """
    pid = tl.program_id(0)

    # Decompose pid -> (b, h, s)
    b = pid // (H * S_new)
    rem = pid % (H * S_new)
    h = rem // S_new
    s = rem % S_new

    # Position in cache for this token
    pos = tl.load(Positions_ptr + s)

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    # Load from new tokens
    src_offset = b * stride_nb + h * stride_nh + s * stride_ns
    vals = tl.load(New_ptr + src_offset + offs_d * stride_nd, mask=mask, other=0.0)

    # Store into cache at the correct position
    dst_offset = b * stride_cb + h * stride_ch + pos * stride_cs
    tl.store(Cache_ptr + dst_offset + offs_d * stride_cd, vals, mask=mask)
