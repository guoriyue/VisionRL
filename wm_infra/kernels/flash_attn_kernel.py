"""Flash Attention v2 Triton kernels (forward + backward).

Based on the Triton official tutorial (06-fused-attention.py) with extensions:
  - Grouped-Query Attention (GQA): multiple Q heads share K/V heads
  - Causal masking: skip fully-masked blocks, element-wise mask on boundary
  - Stores LSE (log-sum-exp) for memory-efficient backward

Layout convention: tensors are passed as [B*H, S, D] with batch*heads merged.
The wrapper handles GQA head mapping via a precomputed KV_Head_Map tensor.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=3, num_warps=4),
    ],
    key=["HEAD_DIM", "IS_CAUSAL"],
)
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    KV_Head_Map_ptr,    # [B * Hq] -> index into B*Hkv space
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    SEQ_Q, SEQ_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash Attention forward.

    Grid: (cdiv(SEQ_Q, BLOCK_M), B * Hq)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)  # index into B*Hq

    # GQA: map Q head -> KV head
    kv_bh = tl.load(KV_Head_Map_ptr + pid_bh)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block: [BLOCK_M, HEAD_DIM]
    q_base = pid_bh * SEQ_Q
    q_ptrs = Q_ptr + q_base * stride_qs + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_Q, other=0.0)

    # KV base offset
    kv_base = kv_bh * SEQ_K

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Determine iteration range for K/V
    end_n = SEQ_K
    if IS_CAUSAL:
        end_n = tl.minimum(start_m + BLOCK_M, SEQ_K)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block: [BLOCK_N, HEAD_DIM]
        k_ptrs = K_ptr + kv_base * stride_ks + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_K, other=0.0)

        # S = Q @ K^T * scale: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Masking
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < SEQ_K, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2((m_i - m_new) * 1.44269504)
        p = tl.exp2((qk - m_new[:, None]) * 1.44269504)

        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        # Load V block: [BLOCK_N, HEAD_DIM]
        v_ptrs = V_ptr + kv_base * stride_vs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_K, other=0.0)

        # Accumulate: acc += P @ V
        acc += tl.dot(p.to(v.dtype), v)
        l_i += tl.sum(p, axis=1)
        m_i = m_new

    # Finalize
    acc = acc / l_i[:, None]

    # Store O
    o_ptrs = O_ptr + q_base * stride_os + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.load(Q_ptr).dtype), mask=offs_m[:, None] < SEQ_Q)

    # Store LSE = m + ln(l)
    lse = m_i + tl.log(l_i) / 1.44269504
    tl.store(LSE_ptr + q_base + offs_m, lse, mask=offs_m < SEQ_Q)


@triton.jit
def flash_attn_bwd_preprocess(
    O_ptr, DO_ptr, Delta_ptr,
    stride_os, stride_od,
    SEQ_Q,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Precompute Delta = rowsum(O * dO).

    Grid: (cdiv(SEQ_Q, BLOCK_M), B * Hq)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    base = pid_bh * SEQ_Q
    o_ptrs = O_ptr + base * stride_os + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = DO_ptr + base * stride_os + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od

    mask = offs_m[:, None] < SEQ_Q
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta_ptr + base + offs_m, delta, mask=offs_m < SEQ_Q)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_stages=2, num_warps=4),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def flash_attn_decode_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    KV_Head_Map_ptr,
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_oh, stride_od,
    SEQ_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-query decode attention.

    Expects Q laid out as [B*Hq, 1, D], K/V as [B*Hkv, Sk, D], and O as [B*Hq, D].
    Grid: (B * Hq,)
    """
    pid_bh = tl.program_id(0)
    kv_bh = tl.load(KV_Head_Map_ptr + pid_bh)

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + pid_bh * stride_qs + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0)

    kv_base = kv_bh * SEQ_K
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, SEQ_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SEQ_K
        mask_2d = n_mask[:, None] & (offs_d[None, :] < HEAD_DIM)

        k_ptrs = K_ptr + kv_base * stride_ks + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + kv_base * stride_vs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_2d, other=0.0)
        v = tl.load(v_ptrs, mask=mask_2d, other=0.0)

        qk = tl.sum(k * q[None, :], axis=1) * sm_scale
        qk = tl.where(n_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2((m_i - m_new) * 1.44269504)
        p = tl.exp2((qk - m_new) * 1.44269504)

        acc = acc * alpha
        acc += tl.sum(v * p[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    acc = acc / l_i
    o_ptrs = O_ptr + pid_bh * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, acc.to(tl.load(Q_ptr).dtype), mask=offs_d < HEAD_DIM)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    ],
    key=["HEAD_DIM", "IS_CAUSAL"],
)
@triton.jit
def flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    LSE_ptr, Delta_ptr,
    KV_Head_Map_ptr,
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    SEQ_Q, SEQ_K,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash Attention backward (per Q-head).

    Computes dQ, and accumulates dK/dV via atomic_add (since multiple Q heads
    may share the same KV head in GQA).

    Grid: (cdiv(SEQ_Q, BLOCK_M), B * Hq)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)  # index into B*Hq

    kv_bh = tl.load(KV_Head_Map_ptr + pid_bh)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = pid_bh * SEQ_Q
    kv_base = kv_bh * SEQ_K

    # Load Q, dO, LSE, Delta for this block
    q_ptrs = Q_ptr + q_base * stride_qs + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = DO_ptr + q_base * stride_os + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    m_mask = offs_m[:, None] < SEQ_Q

    q = tl.load(q_ptrs, mask=m_mask, other=0.0)
    do = tl.load(do_ptrs, mask=m_mask, other=0.0)
    lse = tl.load(LSE_ptr + q_base + offs_m, mask=offs_m < SEQ_Q, other=0.0)
    delta = tl.load(Delta_ptr + q_base + offs_m, mask=offs_m < SEQ_Q, other=0.0)

    # Accumulator for dQ
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Determine K/V range
    end_n = SEQ_K
    if IS_CAUSAL:
        end_n = tl.minimum(start_m + BLOCK_M, SEQ_K)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < SEQ_K

        # Load K, V
        k_ptrs = K_ptr + kv_base * stride_ks + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + kv_base * stride_vs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)
        v = tl.load(v_ptrs, mask=n_mask, other=0.0)

        # Recompute S = Q @ K^T * scale
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < SEQ_K, qk, float("-inf"))

        # Recompute P from LSE
        p = tl.exp2((qk - lse[:, None]) * 1.44269504)

        # dV += P^T @ dO (atomic since multiple Q heads share KV)
        dv_block = tl.dot(tl.trans(p.to(do.dtype)), do)
        dv_ptrs = DV_ptr + kv_base * stride_vs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv_block, mask=n_mask)

        # dP = dO @ V^T
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # dS = P * (dP - Delta)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dQ += dS @ K
        dq += tl.dot(ds.to(k.dtype), k)

        # dK += dS^T @ Q (atomic since multiple Q heads share KV)
        dk_block = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dk_ptrs = DK_ptr + kv_base * stride_ks + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        tl.atomic_add(dk_ptrs, dk_block, mask=n_mask)

    # Store dQ (not atomic - each Q head has its own dQ)
    dq_ptrs = DQ_ptr + q_base * stride_qs + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq.to(tl.load(Q_ptr).dtype), mask=m_mask)
