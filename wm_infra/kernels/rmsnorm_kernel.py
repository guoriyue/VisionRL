"""RMSNorm Triton kernels (forward + backward).

RMSNorm(x) = x * (1 / sqrt(mean(x^2) + eps)) * weight

One program per row. Each program:
  1. Loads the entire row of x
  2. Computes RMS = sqrt(mean(x^2) + eps)
  3. Normalizes: y = x / RMS * weight
  4. Stores y (and rstd for backward)
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,          # [M, N]
    W_ptr,          # [N]
    Y_ptr,          # [M, N]
    Rstd_ptr,       # [M] — 1/sqrt(mean(x^2)+eps), saved for backward
    N,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load row
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    y = x * rstd * w

    # Store
    tl.store(Y_ptr + row * N + cols, y.to(tl.load(X_ptr + row * N).dtype), mask=mask)
    tl.store(Rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(
    DY_ptr,         # [M, N] — grad of output
    X_ptr,          # [M, N]
    W_ptr,          # [N]
    Rstd_ptr,       # [M]
    DX_ptr,         # [M, N] — grad of input
    DW_ptr,         # [M, N] — per-row partial dweight (reduced outside kernel)
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load
    dy = tl.load(DY_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    # x_hat = x * rstd
    x_hat = x * rstd

    # dweight partial: dw_row = dy * x_hat
    dw = dy * x_hat
    tl.store(DW_ptr + row * N + cols, dw.to(tl.load(DY_ptr + row * N).dtype), mask=mask)

    # dx = rstd * (dy * w - x_hat * mean(dy * w * x_hat))
    dy_w = dy * w
    inner = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * inner)

    tl.store(DX_ptr + row * N + cols, dx.to(tl.load(X_ptr + row * N).dtype), mask=mask)
