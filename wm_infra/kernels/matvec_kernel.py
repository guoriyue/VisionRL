"""Decode-specific GEMV kernels for M=1 (single-token) matrix-vector multiplication.

At decode time (M=1), the standard grouped GEMM kernel wastes >93% of compute
on zero-padded rows (BLOCK_M=16, but only 1 real row). These GEMV kernels are
designed for memory-bandwidth-bound M=1 workloads:

  - Each thread block computes a chunk of the N output dimension
  - The single input vector is loaded into registers/shared memory once
  - Weight rows are streamed through and dotted with the input

Variants:
  - batched_matvec_kernel: fp16, shared input across experts (gate/up proj)
  - batched_matvec_varying_kernel: fp16, per-expert input (down proj)
  - batched_matvec_int4_kernel: INT4, shared input (gate/up proj)
  - batched_matvec_int4_varying_kernel: INT4, per-expert input (down proj)
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_kernel(
    # ── Data pointers ──
    x_ptr,                  # [K] — single input vector (shared across experts)
    W_ptr,                  # [num_experts_active, K, N] — selected expert weights
    out_ptr,                # [num_experts_active, N] — output per expert
    # ── Dimensions ──
    N,                      # output dim
    K,                      # input dim (reduction)
    num_experts_active,     # number of active experts (top_k)
    # ── Layout: W strides ──
    stride_we,              # stride for expert dim
    stride_wk,              # stride for K dim
    stride_wn,              # stride for N dim
    # ── Layout: out strides ──
    stride_oe,              # stride for expert dim in output
    stride_on,              # stride for N dim in output
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched GEMV: out[e, :] = x @ W[e, :, :] for all active experts.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Each program handles one expert and one chunk of N output elements.

    This is a pure GEMV: M=1, so no tl.dot needed. We use vector loads
    and element-wise multiply-accumulate, which maximizes memory bandwidth
    utilization for the memory-bound M=1 regime.
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles, loading x and W
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load x vector chunk: [BLOCK_K]
        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_offsets = (expert_idx * stride_we +
                     k_range[:, None] * stride_wk +
                     n_range[None, :] * stride_wn)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        # Dot: acc += sum_k(x[k] * W[k, n]) — broadcast x over N
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # Store output
    out_offsets = expert_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_int4_kernel(
    # ── Data pointers ──
    x_ptr,                  # [K] — single input vector
    W_packed_ptr,           # [num_experts_active, K, N//2] — uint8 packed INT4
    out_ptr,                # [num_experts_active, N] — output
    # ── Quantization parameters ──
    scales_ptr,             # [num_experts_active, K//group_size, N] — fp16
    zeros_ptr,              # [num_experts_active, K//group_size, N] — fp16
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    group_size,             # INT4 quantization group size
    # ── Layout: W_packed strides ──
    stride_wpe,
    stride_wpk,
    stride_wpn,
    # ── Layout: scales/zeros strides ──
    stride_se,
    stride_sg,
    stride_sn,
    # ── Layout: out strides ──
    stride_oe,
    stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched INT4 GEMV: out[e, :] = x @ dequant(W_packed[e, :, :]).

    Grid: (num_experts_active, ceil(N / BLOCK_N))

    Same structure as batched_matvec_kernel but with inline INT4 dequantization.
    Loads uint8 packed weights, unpacks even/odd columns, applies scale/zero,
    then accumulates.
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    # Even/odd output column indices for INT4 unpacking
    HALF_N: tl.constexpr = BLOCK_N // 2
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N

    # Packed column indices
    n_packed = n_start // 2 + tl.arange(0, HALF_N)
    n_packed_mask = n_packed < (N + 1) // 2

    acc_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load x vector chunk: [BLOCK_K]
        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        # Load packed weights: [BLOCK_K, HALF_N] uint8
        wp_offsets = (expert_idx * stride_wpe +
                      k_range[:, None] * stride_wpk +
                      n_packed[None, :] * stride_wpn)
        wp_mask = k_mask[:, None] & n_packed_mask[None, :]
        w_packed = tl.load(W_packed_ptr + wp_offsets, mask=wp_mask, other=0)

        # Unpack: low nibble = even, high nibble = odd
        b_low = (w_packed & 0xF).to(tl.float32)
        b_high = ((w_packed >> 4) & 0xF).to(tl.float32)

        # Load scales/zeros
        g_range = k_range // group_size
        sz_even_offsets = (expert_idx * stride_se +
                           g_range[:, None] * stride_sg +
                           n_even[None, :] * stride_sn)
        sz_odd_offsets = (expert_idx * stride_se +
                          g_range[:, None] * stride_sg +
                          n_odd[None, :] * stride_sn)
        sz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        s_even = tl.load(scales_ptr + sz_even_offsets, mask=sz_even_mask, other=1.0).to(tl.float32)
        z_even = tl.load(zeros_ptr + sz_even_offsets, mask=sz_even_mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scales_ptr + sz_odd_offsets, mask=sz_odd_mask, other=1.0).to(tl.float32)
        z_odd = tl.load(zeros_ptr + sz_odd_offsets, mask=sz_odd_mask, other=0.0).to(tl.float32)

        # Dequantize: w_fp = (w_int4 - zero) * scale
        w_even = (b_low - z_even) * s_even   # [BLOCK_K, HALF_N]
        w_odd = (b_high - z_odd) * s_odd     # [BLOCK_K, HALF_N]

        # GEMV: acc += x[k] * w[k, n]
        acc_even += tl.sum(x_vals[:, None] * w_even, axis=0)
        acc_odd += tl.sum(x_vals[:, None] * w_odd, axis=0)

    # Store interleaved output
    out_even_offsets = expert_idx * stride_oe + n_even * stride_on
    out_odd_offsets = expert_idx * stride_oe + n_odd * stride_on
    tl.store(out_ptr + out_even_offsets, acc_even.to(out_ptr.dtype.element_ty), mask=n_even_mask)
    tl.store(out_ptr + out_odd_offsets, acc_odd.to(out_ptr.dtype.element_ty), mask=n_odd_mask)


# ─── Per-expert input variants (for down projection) ───
# These are like the shared-x variants above, but each expert has its own
# input vector x[e, :] instead of a single shared x[:].

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_varying_kernel(
    # ── Data pointers ──
    x_ptr,                  # [num_experts_active, K] — per-expert input vectors
    W_ptr,                  # [num_experts_active, K, N] — selected expert weights
    out_ptr,                # [num_experts_active, N] — output per expert
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    # ── Layout: x strides ──
    stride_xe,
    stride_xk,
    # ── Layout: W strides ──
    stride_we,
    stride_wk,
    stride_wn,
    # ── Layout: out strides ──
    stride_oe,
    stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched GEMV with per-expert input: out[e, :] = x[e, :] @ W[e, :, :].

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Used for down projection where each expert has a different activated input.
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load per-expert x vector chunk: [BLOCK_K]
        x_offsets = expert_idx * stride_xe + k_range * stride_xk
        x_vals = tl.load(x_ptr + x_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_offsets = (expert_idx * stride_we +
                     k_range[:, None] * stride_wk +
                     n_range[None, :] * stride_wn)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    out_offsets = expert_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_int4_varying_kernel(
    # ── Data pointers ──
    x_ptr,                  # [num_experts_active, K] — per-expert input vectors
    W_packed_ptr,           # [num_experts_active, K, N//2] — uint8 packed INT4
    out_ptr,                # [num_experts_active, N] — output
    # ── Quantization parameters ──
    scales_ptr,             # [num_experts_active, K//group_size, N]
    zeros_ptr,              # [num_experts_active, K//group_size, N]
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    group_size,
    # ── Layout: x strides ──
    stride_xe,
    stride_xk,
    # ── Layout: W_packed strides ──
    stride_wpe,
    stride_wpk,
    stride_wpn,
    # ── Layout: scales/zeros strides ──
    stride_se,
    stride_sg,
    stride_sn,
    # ── Layout: out strides ──
    stride_oe,
    stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched INT4 GEMV with per-expert input: out[e] = x[e] @ dequant(W[e]).

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    HALF_N: tl.constexpr = BLOCK_N // 2
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N

    n_packed = n_start // 2 + tl.arange(0, HALF_N)
    n_packed_mask = n_packed < (N + 1) // 2

    acc_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load per-expert x vector chunk: [BLOCK_K]
        x_offsets = expert_idx * stride_xe + k_range * stride_xk
        x_vals = tl.load(x_ptr + x_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Load packed weights: [BLOCK_K, HALF_N] uint8
        wp_offsets = (expert_idx * stride_wpe +
                      k_range[:, None] * stride_wpk +
                      n_packed[None, :] * stride_wpn)
        wp_mask = k_mask[:, None] & n_packed_mask[None, :]
        w_packed = tl.load(W_packed_ptr + wp_offsets, mask=wp_mask, other=0)

        b_low = (w_packed & 0xF).to(tl.float32)
        b_high = ((w_packed >> 4) & 0xF).to(tl.float32)

        g_range = k_range // group_size
        sz_even_offsets = (expert_idx * stride_se +
                           g_range[:, None] * stride_sg +
                           n_even[None, :] * stride_sn)
        sz_odd_offsets = (expert_idx * stride_se +
                          g_range[:, None] * stride_sg +
                          n_odd[None, :] * stride_sn)
        sz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        s_even = tl.load(scales_ptr + sz_even_offsets, mask=sz_even_mask, other=1.0).to(tl.float32)
        z_even = tl.load(zeros_ptr + sz_even_offsets, mask=sz_even_mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scales_ptr + sz_odd_offsets, mask=sz_odd_mask, other=1.0).to(tl.float32)
        z_odd = tl.load(zeros_ptr + sz_odd_offsets, mask=sz_odd_mask, other=0.0).to(tl.float32)

        w_even = (b_low - z_even) * s_even
        w_odd = (b_high - z_odd) * s_odd

        acc_even += tl.sum(x_vals[:, None] * w_even, axis=0)
        acc_odd += tl.sum(x_vals[:, None] * w_odd, axis=0)

    out_even_offsets = expert_idx * stride_oe + n_even * stride_on
    out_odd_offsets = expert_idx * stride_oe + n_odd * stride_on
    tl.store(out_ptr + out_even_offsets, acc_even.to(out_ptr.dtype.element_ty), mask=n_even_mask)
    tl.store(out_ptr + out_odd_offsets, acc_odd.to(out_ptr.dtype.element_ty), mask=n_odd_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_dual_matvec_kernel(
    x_ptr,                  # [K] — shared input vector
    W_gate_ptr,             # [num_experts, K, N]
    W_up_ptr,               # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_gate_ptr,           # [num_experts_active, N]
    out_up_ptr,             # [num_experts_active, N]
    N,
    K,
    num_experts_active,
    stride_wge,
    stride_wgk,
    stride_wgn,
    stride_wue,
    stride_wuk,
    stride_wun,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Shared-input dual GEMV for gate/up projections using expert id indirection.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Avoids per-step weight gathering by reading directly from the source expert tensors.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        gate_offsets = (
            expert_idx * stride_wge
            + k_range[:, None] * stride_wgk
            + n_range[None, :] * stride_wgn
        )
        up_offsets = (
            expert_idx * stride_wue
            + k_range[:, None] * stride_wuk
            + n_range[None, :] * stride_wun
        )
        w_mask = k_mask[:, None] & n_mask[None, :]

        gate_vals = tl.load(W_gate_ptr + gate_offsets, mask=w_mask, other=0.0).to(tl.float32)
        up_vals = tl.load(W_up_ptr + up_offsets, mask=w_mask, other=0.0).to(tl.float32)

        prod = x_vals[:, None]
        acc_gate += tl.sum(prod * gate_vals, axis=0)
        acc_up += tl.sum(prod * up_vals, axis=0)

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_gate_ptr + out_offsets, acc_gate.to(out_gate_ptr.dtype.element_ty), mask=n_mask)
    tl.store(out_up_ptr + out_offsets, acc_up.to(out_up_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_matvec_varying_kernel(
    x_ptr,                  # [num_experts_active, K]
    W_ptr,                  # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_ptr,                # [num_experts_active, N]
    N,
    K,
    num_experts_active,
    stride_xe,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Per-expert-input GEMV using expert id indirection for down projection."""
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(
            x_ptr + row_idx * stride_xe + k_range * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_offsets = (
            expert_idx * stride_we
            + k_range[:, None] * stride_wk
            + n_range[None, :] * stride_wn
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


# ─── Fused kernels for reduced kernel count ───

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_dual_matvec_swiglu_kernel(
    x_ptr,                  # [K] — shared input vector
    W_gate_ptr,             # [num_experts, K, N]
    W_up_ptr,               # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_ptr,                # [num_experts_active, N] — SwiGLU output
    N,
    K,
    num_experts_active,
    stride_wge,
    stride_wgk,
    stride_wgn,
    stride_wue,
    stride_wuk,
    stride_wun,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Fused gate+up GEMV with inline SwiGLU activation.

    Computes: out[e, :] = SiLU(x @ W_gate[expert_ids[e]]) * (x @ W_up[expert_ids[e]])

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Saves one kernel launch per MoE layer by fusing the SwiGLU activation
    into the matvec output stage.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        gate_offsets = (
            expert_idx * stride_wge
            + k_range[:, None] * stride_wgk
            + n_range[None, :] * stride_wgn
        )
        up_offsets = (
            expert_idx * stride_wue
            + k_range[:, None] * stride_wuk
            + n_range[None, :] * stride_wun
        )
        w_mask = k_mask[:, None] & n_mask[None, :]

        gate_vals = tl.load(W_gate_ptr + gate_offsets, mask=w_mask, other=0.0).to(tl.float32)
        up_vals = tl.load(W_up_ptr + up_offsets, mask=w_mask, other=0.0).to(tl.float32)

        prod = x_vals[:, None]
        acc_gate += tl.sum(prod * gate_vals, axis=0)
        acc_up += tl.sum(prod * up_vals, axis=0)

    # Fused SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up
    # Already in fp32 — sigmoid needs fp32 anyway
    sigmoid_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sigmoid_gate
    result = silu_gate * acc_up

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, result.to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def indexed_matvec_varying_weighted_kernel(
    x_ptr,                  # [num_experts_active, K]
    W_ptr,                  # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    weights_ptr,            # [num_experts_active] — routing weights
    out_ptr,                # [1, N] — weighted sum output (accumulated via atomic_add)
    N,
    K,
    num_experts_active,
    stride_xe,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Per-expert GEMV with routing weight applied + atomic accumulation.

    Computes: out[0, :] = sum_e(weights[e] * x[e] @ W[expert_ids[e]])

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Fuses the down projection, weight multiplication, and expert reduction
    into a single kernel, eliminating 3 separate kernel launches (matvec,
    unsqueeze+mul, sum) per MoE layer.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    weight = tl.load(weights_ptr + row_idx).to(tl.float32)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(
            x_ptr + row_idx * stride_xe + k_range * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_offsets = (
            expert_idx * stride_we
            + k_range[:, None] * stride_wk
            + n_range[None, :] * stride_wn
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # Apply routing weight and accumulate into shared output via atomic_add
    weighted = acc * weight
    out_offsets = n_range * stride_on
    tl.atomic_add(out_ptr + out_offsets, weighted.to(out_ptr.dtype.element_ty), mask=n_mask)


# ─── Indexed INT4 variants (no weight gathering) ───
# These combine expert-id indirection (like indexed_dual_matvec_kernel)
# with INT4 dequantization (like batched_matvec_int4_kernel).
# Key benefit: avoids index_select weight copies at decode time.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_dual_matvec_int4_kernel(
    # ── Data pointers ──
    x_ptr,                  # [K] — shared input vector
    W_gate_packed_ptr,      # [num_experts, K, N//2] — uint8 packed INT4
    W_up_packed_ptr,        # [num_experts, K, N//2] — uint8 packed INT4
    # ── Quantization params ──
    scales_gate_ptr,        # [num_experts, K//group_size, N] — fp16
    zeros_gate_ptr,         # [num_experts, K//group_size, N] — fp16
    scales_up_ptr,          # [num_experts, K//group_size, N] — fp16
    zeros_up_ptr,           # [num_experts, K//group_size, N] — fp16
    # ── Expert mapping ──
    expert_ids_ptr,         # [num_experts_active]
    # ── Outputs ──
    out_gate_ptr,           # [num_experts_active, N]
    out_up_ptr,             # [num_experts_active, N]
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    group_size,
    # ── Layout: W_packed strides [num_experts, K, N//2] ──
    stride_wge, stride_wgk, stride_wgn,
    stride_wue, stride_wuk, stride_wun,
    # ── Layout: scales/zeros strides [num_experts, num_groups, N] ──
    stride_sge, stride_sgg, stride_sgn,
    stride_sue, stride_sug, stride_sun,
    # ── Layout: output strides [num_experts_active, N] ──
    stride_oe, stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Shared-input dual INT4 GEMV for gate/up projections with expert indirection.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Each program handles one expert and BLOCK_N output columns.
    Reads weights directly from [num_experts, ...] tensors using expert_ids lookup.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N

    # INT4 column setup
    HALF_N: tl.constexpr = BLOCK_N // 2
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N
    n_packed = n_start // 2 + tl.arange(0, HALF_N)
    n_packed_mask = n_packed < (N + 1) // 2

    acc_gate_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_gate_odd = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_up_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_up_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load shared input vector
        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        # ── Gate weights ──
        gp_offsets = (expert_idx * stride_wge +
                      k_range[:, None] * stride_wgk +
                      n_packed[None, :] * stride_wgn)
        gp_mask = k_mask[:, None] & n_packed_mask[None, :]
        g_packed = tl.load(W_gate_packed_ptr + gp_offsets, mask=gp_mask, other=0)

        g_low = (g_packed & 0xF).to(tl.float32)
        g_high = ((g_packed >> 4) & 0xF).to(tl.float32)

        g_range = k_range // group_size
        sgz_even_offsets = (expert_idx * stride_sge +
                            g_range[:, None] * stride_sgg +
                            n_even[None, :] * stride_sgn)
        sgz_odd_offsets = (expert_idx * stride_sge +
                           g_range[:, None] * stride_sgg +
                           n_odd[None, :] * stride_sgn)
        sgz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sgz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        sg_even = tl.load(scales_gate_ptr + sgz_even_offsets, mask=sgz_even_mask, other=1.0).to(tl.float32)
        zg_even = tl.load(zeros_gate_ptr + sgz_even_offsets, mask=sgz_even_mask, other=0.0).to(tl.float32)
        sg_odd = tl.load(scales_gate_ptr + sgz_odd_offsets, mask=sgz_odd_mask, other=1.0).to(tl.float32)
        zg_odd = tl.load(zeros_gate_ptr + sgz_odd_offsets, mask=sgz_odd_mask, other=0.0).to(tl.float32)

        gw_even = (g_low - zg_even) * sg_even
        gw_odd = (g_high - zg_odd) * sg_odd

        x_bcast = x_vals[:, None]
        acc_gate_even += tl.sum(x_bcast * gw_even, axis=0)
        acc_gate_odd += tl.sum(x_bcast * gw_odd, axis=0)

        # ── Up weights ──
        up_offsets = (expert_idx * stride_wue +
                      k_range[:, None] * stride_wuk +
                      n_packed[None, :] * stride_wun)
        u_packed = tl.load(W_up_packed_ptr + up_offsets, mask=gp_mask, other=0)

        u_low = (u_packed & 0xF).to(tl.float32)
        u_high = ((u_packed >> 4) & 0xF).to(tl.float32)

        suz_even_offsets = (expert_idx * stride_sue +
                            g_range[:, None] * stride_sug +
                            n_even[None, :] * stride_sun)
        suz_odd_offsets = (expert_idx * stride_sue +
                           g_range[:, None] * stride_sug +
                           n_odd[None, :] * stride_sun)

        su_even = tl.load(scales_up_ptr + suz_even_offsets, mask=sgz_even_mask, other=1.0).to(tl.float32)
        zu_even = tl.load(zeros_up_ptr + suz_even_offsets, mask=sgz_even_mask, other=0.0).to(tl.float32)
        su_odd = tl.load(scales_up_ptr + suz_odd_offsets, mask=sgz_odd_mask, other=1.0).to(tl.float32)
        zu_odd = tl.load(zeros_up_ptr + suz_odd_offsets, mask=sgz_odd_mask, other=0.0).to(tl.float32)

        uw_even = (u_low - zu_even) * su_even
        uw_odd = (u_high - zu_odd) * su_odd

        acc_up_even += tl.sum(x_bcast * uw_even, axis=0)
        acc_up_odd += tl.sum(x_bcast * uw_odd, axis=0)

    # Store interleaved gate output
    out_offsets_even = row_idx * stride_oe + n_even * stride_on
    out_offsets_odd = row_idx * stride_oe + n_odd * stride_on
    tl.store(out_gate_ptr + out_offsets_even, acc_gate_even.to(out_gate_ptr.dtype.element_ty), mask=n_even_mask)
    tl.store(out_gate_ptr + out_offsets_odd, acc_gate_odd.to(out_gate_ptr.dtype.element_ty), mask=n_odd_mask)
    tl.store(out_up_ptr + out_offsets_even, acc_up_even.to(out_up_ptr.dtype.element_ty), mask=n_even_mask)
    tl.store(out_up_ptr + out_offsets_odd, acc_up_odd.to(out_up_ptr.dtype.element_ty), mask=n_odd_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_matvec_int4_varying_kernel(
    # ── Data pointers ──
    x_ptr,                  # [num_experts_active, K] — per-expert input vectors
    W_packed_ptr,           # [num_experts, K, N//2] — uint8 packed INT4
    # ── Quantization params ──
    scales_ptr,             # [num_experts, K//group_size, N] — fp16
    zeros_ptr,              # [num_experts, K//group_size, N] — fp16
    # ── Expert mapping ──
    expert_ids_ptr,         # [num_experts_active]
    # ── Output ──
    out_ptr,                # [num_experts_active, N]
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    group_size,
    # ── Layout: x strides [num_experts_active, K] ──
    stride_xe, stride_xk,
    # ── Layout: W_packed strides [num_experts, K, N//2] ──
    stride_wpe, stride_wpk, stride_wpn,
    # ── Layout: scales/zeros strides [num_experts, num_groups, N] ──
    stride_se, stride_sg, stride_sn,
    # ── Layout: output strides [num_experts_active, N] ──
    stride_oe, stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Per-expert-input INT4 GEMV with expert indirection for down projection.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N

    HALF_N: tl.constexpr = BLOCK_N // 2
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N
    n_packed = n_start // 2 + tl.arange(0, HALF_N)
    n_packed_mask = n_packed < (N + 1) // 2

    acc_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load per-expert x vector chunk
        x_vals = tl.load(
            x_ptr + row_idx * stride_xe + k_range * stride_xk,
            mask=k_mask, other=0.0,
        ).to(tl.float32)

        # Load packed weights for this expert
        wp_offsets = (expert_idx * stride_wpe +
                      k_range[:, None] * stride_wpk +
                      n_packed[None, :] * stride_wpn)
        wp_mask = k_mask[:, None] & n_packed_mask[None, :]
        w_packed = tl.load(W_packed_ptr + wp_offsets, mask=wp_mask, other=0)

        b_low = (w_packed & 0xF).to(tl.float32)
        b_high = ((w_packed >> 4) & 0xF).to(tl.float32)

        # Scales/zeros
        g_range = k_range // group_size
        sz_even_offsets = (expert_idx * stride_se +
                           g_range[:, None] * stride_sg +
                           n_even[None, :] * stride_sn)
        sz_odd_offsets = (expert_idx * stride_se +
                          g_range[:, None] * stride_sg +
                          n_odd[None, :] * stride_sn)
        sz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        s_even = tl.load(scales_ptr + sz_even_offsets, mask=sz_even_mask, other=1.0).to(tl.float32)
        z_even = tl.load(zeros_ptr + sz_even_offsets, mask=sz_even_mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scales_ptr + sz_odd_offsets, mask=sz_odd_mask, other=1.0).to(tl.float32)
        z_odd = tl.load(zeros_ptr + sz_odd_offsets, mask=sz_odd_mask, other=0.0).to(tl.float32)

        w_even = (b_low - z_even) * s_even
        w_odd = (b_high - z_odd) * s_odd

        x_bcast = x_vals[:, None]
        acc_even += tl.sum(x_bcast * w_even, axis=0)
        acc_odd += tl.sum(x_bcast * w_odd, axis=0)

    out_offsets_even = row_idx * stride_oe + n_even * stride_on
    out_offsets_odd = row_idx * stride_oe + n_odd * stride_on
    tl.store(out_ptr + out_offsets_even, acc_even.to(out_ptr.dtype.element_ty), mask=n_even_mask)
    tl.store(out_ptr + out_offsets_odd, acc_odd.to(out_ptr.dtype.element_ty), mask=n_odd_mask)
