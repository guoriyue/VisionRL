"""Triton kernel for single-position RoPE (decode path).

At decode time, _apply_rope_inline uses 4-5 separate PyTorch elementwise
kernels per call (2 muls + 2 mul-add + 1 cat). With 2 calls per layer
(Q and K), that's ~10 kernels per layer. This Triton kernel does the
entire rotation in a single launch.

Input layout: [num_rows, D] contiguous, where num_rows = B*H (all batch
and head dims collapsed). The kernel processes each row independently.
"""

import triton
import triton.language as tl


@triton.jit
def rope_single_pos_kernel(
    # Input tensor (read-only)
    x_ptr,
    # Output tensor (write)
    out_ptr,
    # Precomputed cos/sin for this position: [D//2] each
    cos_ptr,
    sin_ptr,
    # Number of rows to process = B * H
    num_rows,
    # Head dimension
    D: tl.constexpr,
    # Half of head dimension (rounded up to power of 2 for tl.arange)
    HALF_D: tl.constexpr,
):
    """Apply RoPE rotation to a single-position tensor.

    Reads from x_ptr, writes to out_ptr (can be same or different tensor).

    For each row r and column index i in [0, D//2):
        x0 = x[r, i]
        x1 = x[r, i + D//2]
        out[r, i]        = x0 * cos[i] - x1 * sin[i]
        out[r, i + D//2] = x0 * sin[i] + x1 * cos[i]

    Grid: (num_rows,)
    Each program handles one row of D elements.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    half_d = D // 2
    col_range = tl.arange(0, HALF_D)
    col_mask = col_range < half_d

    # Load cos/sin for this position
    cos_vals = tl.load(cos_ptr + col_range, mask=col_mask).to(tl.float32)
    sin_vals = tl.load(sin_ptr + col_range, mask=col_mask).to(tl.float32)

    # Load x0 (first half) and x1 (second half)
    base = row_idx * D
    x0_offsets = base + col_range
    x1_offsets = base + col_range + half_d

    x0 = tl.load(x_ptr + x0_offsets, mask=col_mask).to(tl.float32)
    x1 = tl.load(x_ptr + x1_offsets, mask=col_mask).to(tl.float32)

    # Rotation
    y0 = x0 * cos_vals - x1 * sin_vals
    y1 = x0 * sin_vals + x1 * cos_vals

    # Store to output
    tl.store(out_ptr + x0_offsets, y0.to(out_ptr.dtype.element_ty), mask=col_mask)
    tl.store(out_ptr + x1_offsets, y1.to(out_ptr.dtype.element_ty), mask=col_mask)
