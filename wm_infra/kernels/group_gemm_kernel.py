"""Grouped GEMM kernel: single-launch multi-expert matrix multiplication.

This is THE critical kernel. It computes:
    C[e] = A[expert_offsets[e]:expert_offsets[e+1]] @ B[e]
for all experts e in a single kernel launch.

Each expert has a different number of tokens (M_e varies), but all experts
share the same N and K dimensions (expert weight matrices are same shape).

Key design decisions:
  - 2D grid: (num_tiles_total, 1) where tiles span all experts
  - Precomputed tile_expert_ids/tile_m_offsets for O(1) pid→expert mapping
  - Standard tiled GEMM inner loop per tile
  - No padding: only compute real tokens per expert

Includes FP8 variant (grouped_gemm_fp8_kernel) that loads FP8 tiles,
upcasts to FP16, and applies per-tensor/per-expert scale factors.

Restructured for clarity and backward support.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def grouped_gemm_kernel(
    # ── Data pointers ──
    A_ptr,                  # [total_tokens, K] — contiguous, sorted by expert
    B_ptr,                  # [num_experts, K, N]
    C_ptr,                  # [total_tokens, N] — output
    # ── Tile mapping (precomputed on CPU) ──
    tile_expert_ids_ptr,    # [total_m_tiles] — expert id for each M-tile
    tile_m_offsets_ptr,     # [total_m_tiles] — token start offset for each M-tile
    tile_m_ends_ptr,        # [total_m_tiles] — token end offset for each M-tile
    # ── Dimensions ──
    N,
    K,
    total_m_tiles,          # number of M-tiles (for bounds check with GROUP_M)
    # ── Layout ──
    stride_ak, stride_am,  # A strides
    stride_be, stride_bk, stride_bn,  # B strides [expert, K, N]
    stride_cm, stride_cn,  # C strides
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,   # number of M-tiles to group for L2 locality
):
    """Grouped GEMM: C[e] = A_e @ B[e] for variable-size A_e.

    Grid: (total_m_tiles * ceil(N/BLOCK_N), 1)

    Each program computes one (BLOCK_M, BLOCK_N) tile of output C.
    Tile→expert mapping is precomputed on CPU for O(1) lookup.
    """
    pid = tl.program_id(0)

    # ── Map pid to (pid_m, pid_n) ──
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # GROUP_M for L2 cache locality (standard Triton swizzle pattern)
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(total_m_tiles - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    # ── O(1) tile→expert lookup ──
    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    # ── Compute tile boundaries ──
    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end  # don't read past this expert's tokens

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # ── Tiled matmul: A[m_range, :] @ B[expert_id, :, n_range] ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_offsets = m_range[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N] from B[expert_id]
        b_offsets = (expert_id * stride_be +
                     k_range[:, None] * stride_bk +
                     n_range[None, :] * stride_bn)
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # ── Store C tile ──
    c_offsets = m_range[:, None] * stride_cm + n_range[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptr + c_offsets, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def grouped_gemm_bwd_dA_kernel(
    # dC @ B^T → dA
    dC_ptr,                 # [total_tokens, N] — gradient from downstream
    B_ptr,                  # [num_experts, K, N] — weights
    dA_ptr,                 # [total_tokens, K] — output gradient
    # ── Tile mapping ──
    tile_expert_ids_ptr,
    tile_m_offsets_ptr,
    tile_m_ends_ptr,
    # ── Dimensions ──
    N,
    K,
    # ── Layout ──
    stride_dcm, stride_dcn,
    stride_be, stride_bk, stride_bn,
    stride_dam, stride_dak,
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    """Backward: dA = dC @ B^T for each expert.

    dA[M, K] = dC[M, N] @ B^T[N, K]
    Grid: (total_m_tiles * num_k_tiles,)
    """
    pid = tl.program_id(0)

    num_k_tiles = tl.cdiv(K, BLOCK_N)  # output tiles over K dimension
    pid_m = pid // num_k_tiles
    pid_k = pid % num_k_tiles

    # ── O(1) tile→expert lookup ──
    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start_val = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    # ── Tile boundaries ──
    m_range = m_start_val + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end

    k_start = pid_k * BLOCK_N
    k_range = k_start + tl.arange(0, BLOCK_N)
    k_mask = k_range < K

    # ── Tiled matmul: dC[m, :] @ B[expert, :, :]^T ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_K):
        n_range = n_start + tl.arange(0, BLOCK_K)
        n_mask_inner = n_range < N

        # Load dC tile: [BLOCK_M, BLOCK_K] from dC[m_range, n_range]
        dc_offsets = m_range[:, None] * stride_dcm + n_range[None, :] * stride_dcn
        dc_mask = m_mask[:, None] & n_mask_inner[None, :]
        dc = tl.load(dC_ptr + dc_offsets, mask=dc_mask, other=0.0)

        # Load B^T tile: [BLOCK_K, BLOCK_N]
        bt_offsets = (expert_id * stride_be +
                      k_range[None, :] * stride_bk +
                      n_range[:, None] * stride_bn)
        bt_mask = n_mask_inner[:, None] & k_mask[None, :]
        bt = tl.load(B_ptr + bt_offsets, mask=bt_mask, other=0.0)

        acc += tl.dot(dc, bt)

    # ── Store dA tile ──
    da_offsets = m_range[:, None] * stride_dam + k_range[None, :] * stride_dak
    da_mask = m_mask[:, None] & k_mask[None, :]
    tl.store(dA_ptr + da_offsets, acc.to(dA_ptr.dtype.element_ty), mask=da_mask)


@triton.jit
def grouped_gemm_bwd_dB_kernel(
    # A^T @ dC → dB (weight gradient)
    A_ptr,                  # [total_tokens, K]
    dC_ptr,                 # [total_tokens, N]
    dB_ptr,                 # [num_experts, K, N] — output weight gradient
    expert_offsets_ptr,
    N,
    K,
    num_experts,
    stride_am, stride_ak,
    stride_dcm, stride_dcn,
    stride_dbe, stride_dbk, stride_dbn,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
):
    """Backward: dB[e] = A_e^T @ dC_e for each expert.

    Different tiling than forward: we tile over (K, N) for the output dB,
    and reduce over M (token dimension) for each expert.
    """
    pid = tl.program_id(0)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_expert = num_k_tiles * num_n_tiles

    expert_id = pid // tiles_per_expert
    local_pid = pid % tiles_per_expert
    pid_k = local_pid // num_n_tiles
    pid_n = local_pid % num_n_tiles

    # Expert's token range
    e_start = tl.load(expert_offsets_ptr + expert_id)
    e_end = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens_e = e_end - e_start

    # Early return for empty experts (dB pre-zeroed in ops layer)
    if num_tokens_e == 0:
        return

    # ── Output tile boundaries ──
    k_start = pid_k * BLOCK_K
    k_range = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_range < K

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # ── Reduction over M (token dimension) ──
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, num_tokens_e, BLOCK_M):
        m_range = e_start + m_start + tl.arange(0, BLOCK_M)
        m_mask_inner = m_range < e_end

        at_offsets = m_range[None, :] * stride_am + k_range[:, None] * stride_ak
        at_mask = k_mask[:, None] & m_mask_inner[None, :]
        at = tl.load(A_ptr + at_offsets, mask=at_mask, other=0.0)

        dc_offsets = m_range[:, None] * stride_dcm + n_range[None, :] * stride_dcn
        dc_mask = m_mask_inner[:, None] & n_mask[None, :]
        dc = tl.load(dC_ptr + dc_offsets, mask=dc_mask, other=0.0)

        acc += tl.dot(at, dc)

    # ── Store dB tile ──
    db_offsets = (expert_id * stride_dbe +
                  k_range[:, None] * stride_dbk +
                  n_range[None, :] * stride_dbn)
    db_mask = k_mask[:, None] & n_mask[None, :]
    tl.store(dB_ptr + db_offsets, acc.to(dB_ptr.dtype.element_ty), mask=db_mask)


# ─── FP8 Grouped GEMM ───

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def grouped_gemm_fp8_kernel(
    # ── Data pointers ──
    A_ptr,                  # [total_tokens, K] — FP8 activations
    B_ptr,                  # [num_experts, K, N] — FP8 weights
    C_ptr,                  # [total_tokens, N] — FP16/BF16 output
    # ── Scale factors ──
    A_scale_ptr,            # scalar — per-tensor activation scale
    B_scale_ptr,            # [num_experts] — per-expert weight scales
    # ── Tile mapping ──
    tile_expert_ids_ptr,
    tile_m_offsets_ptr,
    tile_m_ends_ptr,
    # ── Dimensions ──
    N,
    K,
    total_m_tiles,
    # ── Layout ──
    stride_ak, stride_am,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 8,
):
    """FP8 Grouped GEMM: C[e] = (A_e @ B[e]) * a_scale * b_scale[e].

    Same structure as grouped_gemm_kernel but loads FP8 tiles, upcasts to FP16
    for tl.dot, and applies scale factors after accumulation.
    """
    pid = tl.program_id(0)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(total_m_tiles - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load FP8 A tile and upcast to FP16
        a_offsets = m_range[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_fp8 = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        a = a_fp8.to(tl.float16)

        # Load FP8 B tile and upcast to FP16
        b_offsets = (expert_id * stride_be +
                     k_range[:, None] * stride_bk +
                     n_range[None, :] * stride_bn)
        b_mask = k_mask[:, None] & n_mask[None, :]
        b_fp8 = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        b = b_fp8.to(tl.float16)

        acc += tl.dot(a, b)

    # Apply scale factors: result = acc * a_scale * b_scale[expert_id]
    a_scale = tl.load(A_scale_ptr)
    b_scale = tl.load(B_scale_ptr + expert_id)
    acc = acc * (a_scale * b_scale)

    # Store output
    c_offsets = m_range[:, None] * stride_cm + n_range[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptr + c_offsets, acc.to(C_ptr.dtype.element_ty), mask=c_mask)
