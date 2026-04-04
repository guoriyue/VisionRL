"""INT4 GEMV (Matrix-Vector) kernel optimized for M=1 decode.

Instead of using the grouped GEMM kernel with BLOCK_M tiling (which wastes
>90% compute padding M=1 to BLOCK_M=16), this kernel is designed specifically
for the M=1 case with maximum memory bandwidth utilization.

Key optimizations vs grouped_gemm_int4_kernel:
1. No BLOCK_M dimension -- single row, eliminates 15/16 wasted rows
2. No even/odd split -- unpacks both nibbles inline, avoids stride-2 access
3. Larger BLOCK_K for better streaming (process more K per thread block)
4. Sequential accumulation pattern optimized for memory bandwidth
5. Shared activation vector across all N-tiles via L1 cache

Weight format (same as existing):
    B_packed: [K, N//2] uint8 -- same layout, no conversion needed
    scales:   [K//group_size, N] fp16
    zeros:    [K//group_size, N] fp16
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def matvec_int4_kernel(
    # Pointers
    A_ptr,           # [K] fp16 activation vector (single row)
    B_packed_ptr,    # [K, N//2] uint8 packed INT4 weights
    C_ptr,           # [N] fp16 output vector
    scales_ptr,      # [K//group_size, N] fp16
    zeros_ptr,       # [K//group_size, N] fp16
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    # Strides for B_packed (row-major [K, N//2])
    stride_bk,
    stride_bn,
    # Strides for scales/zeros [num_groups, N]
    stride_sg,
    stride_sn,
    # Tuning parameters
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT4 GEMV: c[n] = sum_k a[k] * dequant(B[k, n]) for M=1.

    Each program instance handles BLOCK_N output elements, iterating over K.
    Within each K-tile, we load BLOCK_K activation values and BLOCK_K x BLOCK_N
    packed weights, unpack both nibbles to full columns, dequantize, and
    accumulate the dot product.
    """
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # Accumulator for output columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Packed N indices (each uint8 holds 2 consecutive N values)
    # n_range covers [n_start, n_start+BLOCK_N) in the output space
    # In packed space, column j holds output columns 2*j and 2*j+1
    # We need packed columns for n_range//2, but since n_range is contiguous,
    # we process BLOCK_N//2 packed columns covering BLOCK_N output columns.
    HALF_N: tl.constexpr = BLOCK_N // 2
    packed_n_start = n_start // 2
    packed_n_range = packed_n_start + tl.arange(0, HALF_N)
    packed_n_mask = packed_n_range < (N + 1) // 2

    # Even and odd output column indices within this N-block
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N

    # We accumulate even and odd separately, then interleave
    acc_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load activation vector tile: [BLOCK_K]
        a = tl.load(A_ptr + k_range, mask=k_mask, other=0.0)  # fp16

        # Load packed B: [BLOCK_K, HALF_N] uint8
        bp_offsets = k_range[:, None] * stride_bk + packed_n_range[None, :] * stride_bn
        bp_mask = k_mask[:, None] & packed_n_mask[None, :]
        b_packed = tl.load(B_packed_ptr + bp_offsets, mask=bp_mask, other=0)

        # Unpack nibbles
        b_low = (b_packed & 0xF).to(tl.float32)    # even columns [BLOCK_K, HALF_N]
        b_high = ((b_packed >> 4) & 0xF).to(tl.float32)  # odd columns [BLOCK_K, HALF_N]

        # Load per-group scales and zeros
        g_range = k_range // group_size
        # Even column scales/zeros
        sz_even_offsets = g_range[:, None] * stride_sg + n_even[None, :] * stride_sn
        sz_odd_offsets = g_range[:, None] * stride_sg + n_odd[None, :] * stride_sn
        sz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        s_even = tl.load(scales_ptr + sz_even_offsets, mask=sz_even_mask, other=1.0).to(tl.float32)
        z_even = tl.load(zeros_ptr + sz_even_offsets, mask=sz_even_mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scales_ptr + sz_odd_offsets, mask=sz_odd_mask, other=1.0).to(tl.float32)
        z_odd = tl.load(zeros_ptr + sz_odd_offsets, mask=sz_odd_mask, other=0.0).to(tl.float32)

        # Dequantize
        b_even_deq = (b_low - z_even) * s_even   # [BLOCK_K, HALF_N]
        b_odd_deq = (b_high - z_odd) * s_odd     # [BLOCK_K, HALF_N]

        # Dot product: a[k] * B_deq[k, n], summed over k
        a_f32 = a.to(tl.float32)
        acc_even += tl.sum(a_f32[:, None] * b_even_deq, axis=0)
        acc_odd += tl.sum(a_f32[:, None] * b_odd_deq, axis=0)

    # Store interleaved output
    tl.store(C_ptr + n_even, acc_even.to(tl.float16), mask=n_even_mask)
    tl.store(C_ptr + n_odd, acc_odd.to(tl.float16), mask=n_odd_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def grouped_matvec_int4_kernel(
    # Pointers
    A_ptr,           # [top_k, K] fp16 activations (one row per expert)
    B_packed_ptr,    # [num_experts, K, N//2] uint8 packed INT4 weights
    C_ptr,           # [top_k, N] fp16 output
    scales_ptr,      # [num_experts, K//group_size, N] fp16
    zeros_ptr,       # [num_experts, K//group_size, N] fp16
    # Expert mapping
    expert_ids_ptr,  # [top_k] int32 -- which expert for each row
    # Dimensions
    top_k,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    # Strides for B_packed [num_experts, K, N//2]
    stride_be, stride_bk, stride_bn,
    # Strides for scales/zeros [num_experts, num_groups, N]
    stride_se, stride_sg, stride_sn,
    # Strides for A [top_k, K]
    stride_am, stride_ak,
    # Strides for C [top_k, N]
    stride_cm, stride_cn,
    # Tuning
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped INT4 GEMV for top-k decode: each row uses a different expert.

    Grid: [num_n_tiles * top_k]. Each program handles one (expert_row, N_block).
    """
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    row_idx = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    # Bounds check on row
    if row_idx >= top_k:
        return

    expert_id = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N

    HALF_N: tl.constexpr = BLOCK_N // 2
    packed_n_start = n_start // 2
    packed_n_range = packed_n_start + tl.arange(0, HALF_N)
    packed_n_mask = packed_n_range < (N + 1) // 2

    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N

    acc_even = tl.zeros((HALF_N,), dtype=tl.float32)
    acc_odd = tl.zeros((HALF_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load activation for this row
        a = tl.load(A_ptr + row_idx * stride_am + k_range * stride_ak,
                     mask=k_mask, other=0.0)

        # Load packed B for this expert
        bp_offsets = (expert_id * stride_be +
                      k_range[:, None] * stride_bk +
                      packed_n_range[None, :] * stride_bn)
        bp_mask = k_mask[:, None] & packed_n_mask[None, :]
        b_packed = tl.load(B_packed_ptr + bp_offsets, mask=bp_mask, other=0)

        b_low = (b_packed & 0xF).to(tl.float32)
        b_high = ((b_packed >> 4) & 0xF).to(tl.float32)

        g_range = k_range // group_size
        sz_even_offsets = (expert_id * stride_se +
                           g_range[:, None] * stride_sg +
                           n_even[None, :] * stride_sn)
        sz_odd_offsets = (expert_id * stride_se +
                          g_range[:, None] * stride_sg +
                          n_odd[None, :] * stride_sn)
        sz_even_mask = k_mask[:, None] & n_even_mask[None, :]
        sz_odd_mask = k_mask[:, None] & n_odd_mask[None, :]

        s_even = tl.load(scales_ptr + sz_even_offsets, mask=sz_even_mask, other=1.0).to(tl.float32)
        z_even = tl.load(zeros_ptr + sz_even_offsets, mask=sz_even_mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scales_ptr + sz_odd_offsets, mask=sz_odd_mask, other=1.0).to(tl.float32)
        z_odd = tl.load(zeros_ptr + sz_odd_offsets, mask=sz_odd_mask, other=0.0).to(tl.float32)

        b_even_deq = (b_low - z_even) * s_even
        b_odd_deq = (b_high - z_odd) * s_odd

        a_f32 = a.to(tl.float32)
        acc_even += tl.sum(a_f32[:, None] * b_even_deq, axis=0)
        acc_odd += tl.sum(a_f32[:, None] * b_odd_deq, axis=0)

    # Store
    c_even_offsets = row_idx * stride_cm + n_even * stride_cn
    c_odd_offsets = row_idx * stride_cm + n_odd * stride_cn
    tl.store(C_ptr + c_even_offsets, acc_even.to(tl.float16), mask=n_even_mask)
    tl.store(C_ptr + c_odd_offsets, acc_odd.to(tl.float16), mask=n_odd_mask)
