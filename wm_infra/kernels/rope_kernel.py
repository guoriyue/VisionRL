"""Rotary Position Embedding (RoPE) Triton kernel.

Applies rotation: for each pair (x0, x1) at position p:
    y0 = x0 * cos(p*theta) - x1 * sin(p*theta)
    y1 = x0 * sin(p*theta) + x1 * cos(p*theta)

The cos/sin tables are precomputed and passed in.
One program handles one (token, head) pair.
"""

import triton
import triton.language as tl


@triton.jit
def rope_fwd_kernel(
    X_ptr,          # [B, S, H, D]
    COS_ptr,        # [S, D//2] — precomputed cos(pos * freq)
    SIN_ptr,        # [S, D//2] — precomputed sin(pos * freq)
    POS_ptr,        # [S] — position indices (int64)
    Y_ptr,          # [B, S, H, D] — output
    S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_cos_s, stride_cos_d,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per (batch, seq, head)
    pid = tl.program_id(0)
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # Get actual position for this token
    pos = tl.load(POS_ptr + s)

    cols = tl.arange(0, BLOCK_D)
    mask = cols < HALF_D

    # Load paired elements: x[..., :D//2] and x[..., D//2:]
    base = b * stride_xb + s * stride_xs + h * stride_xh
    x0 = tl.load(X_ptr + base + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X_ptr + base + (cols + HALF_D) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin for this position
    cos_base = pos * stride_cos_s
    cos_val = tl.load(COS_ptr + cos_base + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin_val = tl.load(SIN_ptr + cos_base + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    y0 = x0 * cos_val - x1 * sin_val
    y1 = x0 * sin_val + x1 * cos_val

    # Store
    out_base = b * stride_yb + s * stride_ys + h * stride_yh
    out_dtype = tl.load(X_ptr + base).dtype
    tl.store(Y_ptr + out_base + cols * stride_yd, y0.to(out_dtype), mask=mask)
    tl.store(Y_ptr + out_base + (cols + HALF_D) * stride_yd, y1.to(out_dtype), mask=mask)


@triton.jit
def rope_bwd_kernel(
    DY_ptr,         # [B, S, H, D]
    COS_ptr,        # [S, D//2]
    SIN_ptr,        # [S, D//2]
    POS_ptr,        # [S]
    DX_ptr,         # [B, S, H, D]
    S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_cos_s, stride_cos_d,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Backward is inverse rotation: negate sin
    pid = tl.program_id(0)
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    pos = tl.load(POS_ptr + s)

    cols = tl.arange(0, BLOCK_D)
    mask = cols < HALF_D

    base = b * stride_xb + s * stride_xs + h * stride_xh
    dy0 = tl.load(DY_ptr + base + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)
    dy1 = tl.load(DY_ptr + base + (cols + HALF_D) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    cos_base = pos * stride_cos_s
    cos_val = tl.load(COS_ptr + cos_base + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin_val = tl.load(SIN_ptr + cos_base + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

    # Inverse rotation: negate sin
    dx0 = dy0 * cos_val + dy1 * sin_val
    dx1 = -dy0 * sin_val + dy1 * cos_val

    out_base = b * stride_yb + s * stride_ys + h * stride_yh
    out_dtype = tl.load(DY_ptr + base).dtype
    tl.store(DX_ptr + out_base + cols * stride_yd, dx0.to(out_dtype), mask=mask)
    tl.store(DX_ptr + out_base + (cols + HALF_D) * stride_yd, dx1.to(out_dtype), mask=mask)
