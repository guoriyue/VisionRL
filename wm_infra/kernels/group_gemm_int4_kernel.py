"""INT4 Grouped GEMM kernel: weight-only INT4 quantization with per-group scales/zeros.

Loads INT4 weights packed as uint8 (2 values per byte), dequantizes to fp16/bf16 on the fly,
and computes grouped matrix multiplication using the same tile mapping as the fp16 kernel.

Packing convention (same as GPTQ/AWQ):
    packed[k, n] = weights[k, 2*n] | (weights[k, 2*n+1] << 4)
    i.e., low nibble = even column, high nibble = odd column.

Weight format:
    B_packed: [num_experts, K, N//2] uint8
    scales:   [num_experts, num_groups, N] fp16 (num_groups = K // group_size)
    zeros:    [num_experts, num_groups, N] fp16
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def grouped_gemm_int4_kernel(
    # ── Data pointers ──
    A_ptr,                      # [total_tokens, K] fp16 activations
    B_packed_ptr,               # [num_experts, K, N//2] uint8 packed INT4 weights
    C_ptr,                      # [total_tokens, N] fp16 output
    # ── Quantization parameters ──
    scales_ptr,                 # [num_experts, K//group_size, N] fp16
    zeros_ptr,                  # [num_experts, K//group_size, N] fp16
    # ── Tile mapping ──
    tile_expert_ids_ptr,
    tile_m_offsets_ptr,
    tile_m_ends_ptr,
    # ── Dimensions ──
    N,
    K,
    total_m_tiles,
    group_size,                 # quantization group size (e.g. 128)
    # ── Layout: A [total_tokens, K] ──
    stride_ak, stride_am,
    # ── Layout: B_packed [expert, K, N//2] ──
    stride_bpe, stride_bpk, stride_bpn,
    # ── Layout: scales/zeros [expert, num_groups, N] ──
    stride_se, stride_sg, stride_sn,
    # ── Layout: C ──
    stride_cm, stride_cn,
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 8,
):
    """INT4 Grouped GEMM: C[e] = A_e @ dequant(B_packed[e]).

    For each K-tile, loads uint8 packed weights, unpacks to two INT4 values,
    dequantizes with per-group scales/zeros, then accumulates via tl.dot.

    The inner loop processes BLOCK_K rows of K at a time. For the B matrix,
    we load [BLOCK_K, BLOCK_N//2] uint8 values, unpack into even/odd columns,
    dequantize each half separately, then compute two half-width dot products.
    """
    pid = tl.program_id(0)

    # ── Map pid to (pid_m, pid_n) with GROUP_M swizzle ──
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(total_m_tiles - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── O(1) tile→expert lookup ──
    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # We split BLOCK_N into two halves: even columns [0,2,4,...] and odd [1,3,5,...]
    # Each half has BLOCK_N//2 elements.
    # acc_even accumulates A @ B_even_cols, acc_odd accumulates A @ B_odd_cols
    HALF_N: tl.constexpr = BLOCK_N // 2
    acc_even = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    # Even/odd output column indices
    n_even = n_start + tl.arange(0, HALF_N) * 2
    n_odd = n_even + 1
    n_even_mask = n_even < N
    n_odd_mask = n_odd < N

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_offsets = m_range[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask_2d = m_mask[:, None] & k_mask[None, :]
        a = tl.load(A_ptr + a_offsets, mask=a_mask_2d, other=0.0)

        # Load packed B: [BLOCK_K, HALF_N] uint8
        # B_packed[expert, k, n//2] where n = even output column index
        n_packed = n_start // 2 + tl.arange(0, HALF_N)
        n_packed_mask = n_packed < (N + 1) // 2

        bp_offsets = (expert_id * stride_bpe +
                      k_range[:, None] * stride_bpk +
                      n_packed[None, :] * stride_bpn)
        bp_mask = k_mask[:, None] & n_packed_mask[None, :]
        b_packed = tl.load(B_packed_ptr + bp_offsets, mask=bp_mask, other=0)

        # Unpack INT4: low nibble = even column, high nibble = odd column
        b_low = (b_packed & 0xF).to(tl.float32)     # [BLOCK_K, HALF_N]
        b_high = ((b_packed >> 4) & 0xF).to(tl.float32)  # [BLOCK_K, HALF_N]

        # Load scales/zeros for this K-block
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

        # Dequantize
        b_even_deq = ((b_low - z_even) * s_even).to(A_ptr.dtype.element_ty)
        b_odd_deq = ((b_high - z_odd) * s_odd).to(A_ptr.dtype.element_ty)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, HALF_N]
        acc_even += tl.dot(a, b_even_deq)
        acc_odd += tl.dot(a, b_odd_deq)

    # ── Store C: interleave even/odd columns back ──
    # Even columns: C[m, n_start + 0], C[m, n_start + 2], ...
    c_even_offsets = m_range[:, None] * stride_cm + n_even[None, :] * stride_cn
    c_even_mask = m_mask[:, None] & n_even_mask[None, :]
    tl.store(C_ptr + c_even_offsets, acc_even.to(C_ptr.dtype.element_ty), mask=c_even_mask)

    # Odd columns: C[m, n_start + 1], C[m, n_start + 3], ...
    c_odd_offsets = m_range[:, None] * stride_cm + n_odd[None, :] * stride_cn
    c_odd_mask = m_mask[:, None] & n_odd_mask[None, :]
    tl.store(C_ptr + c_odd_offsets, acc_odd.to(C_ptr.dtype.element_ty), mask=c_odd_mask)
