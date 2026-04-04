"""Flash Attention op with autograd support.

Wraps the forward and backward Triton kernels with GQA and causal masking.
Supports multiple backends: flash_attn, flashinfer, triton (custom), sdpa.
"""

import math
import torch
import triton
from typing import Optional

from wm_infra.kernels.flash_attn_kernel import (
    flash_attn_fwd_kernel,
    flash_attn_decode_kernel,
    flash_attn_bwd_preprocess,
    flash_attn_bwd_kernel,
)

# ─── Backend detection ───
_HAS_FA2 = False
_HAS_FLASHINFER = False

try:
    from flash_attn import flash_attn_func as _fa2_func
    _HAS_FA2 = True
except ImportError:
    pass

try:
    from flashinfer import single_prefill_with_kv_cache as _flashinfer_prefill
    from flashinfer import single_decode_with_kv_cache as _flashinfer_decode
    # FlashInfer is importable but uses JIT compilation that may fail on
    # unsupported GPU architectures (e.g., Blackwell/sm_120a with nvcc < 12.8).
    # Verify the GPU's compute capability is supported before declaring available.
    _fi_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    # FlashInfer 0.6.x supports up to sm_90a (Hopper). Blackwell (sm_120a) is not yet supported.
    _HAS_FLASHINFER = _fi_cap[0] <= 9
except ImportError:
    pass


def resolve_attention_backend(backend: str = "auto") -> str:
    """Resolve the requested backend to the concrete runtime implementation."""
    if backend == "auto":
        if _HAS_FA2:
            return "flash_attn"
        if _HAS_FLASHINFER:
            return "flashinfer"
        return "triton"

    if backend == "flash_attn":
        if not _HAS_FA2:
            raise ImportError("flash_attn not installed. pip install flash-attn")
        return backend

    if backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError("flashinfer not installed. pip install flashinfer")
        return backend

    if backend not in {"triton", "sdpa"}:
        raise ValueError(f"Unknown attention backend: {backend}")

    return backend


def _build_kv_head_map(B: int, Hq: int, Hkv: int, device: torch.device) -> torch.Tensor:
    """Build GQA head mapping: [B * Hq] -> index into [B * Hkv].

    For each (b, hq), maps to b * Hkv + hq // kv_groups.
    """
    groups = Hq // Hkv
    # For batch b, q-head hq: kv index = b * Hkv + hq // groups
    b_idx = torch.arange(B, device=device).repeat_interleave(Hq)
    hq_idx = torch.arange(Hq, device=device).repeat(B)
    kv_map = b_idx * Hkv + hq_idx // groups
    return kv_map.to(torch.int64)


def _get_kv_head_map(
    B: int,
    Hq: int,
    Hkv: int,
    device: torch.device,
    kv_head_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reuse a caller-provided GQA head map when available."""
    if kv_head_map is not None:
        return kv_head_map
    return _build_kv_head_map(B, Hq, Hkv, device)


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal):
        """
        Args:
            Q: [B, Hq, Sq, D]
            K: [B, Hkv, Sk, D]
            V: [B, Hkv, Sk, D]
            causal: bool

        Returns:
            O: [B, Hq, Sq, D]
        """
        B, Hq, Sq, D = Q.shape
        _, Hkv, Sk, _ = K.shape
        assert D in (16, 32, 64, 128, 256), f"HEAD_DIM must be power of 2, got {D}"
        assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"

        # Make contiguous with merged batch*head dims
        Q_3d = Q.reshape(B * Hq, Sq, D).contiguous()
        K_3d = K.reshape(B * Hkv, Sk, D).contiguous()
        V_3d = V.reshape(B * Hkv, Sk, D).contiguous()

        O_3d = torch.empty_like(Q_3d)
        LSE = torch.empty(B * Hq, Sq, device=Q.device, dtype=torch.float32)

        kv_head_map = _get_kv_head_map(B, Hq, Hkv, Q.device)

        sm_scale = 1.0 / math.sqrt(D)
        HEAD_DIM = triton.next_power_of_2(D)

        grid = lambda META: (triton.cdiv(Sq, META["BLOCK_M"]), B * Hq)

        flash_attn_fwd_kernel[grid](
            Q_3d, K_3d, V_3d, O_3d, LSE,
            kv_head_map,
            Q_3d.stride(1), Q_3d.stride(2),
            K_3d.stride(1), K_3d.stride(2),
            V_3d.stride(1), V_3d.stride(2),
            O_3d.stride(1), O_3d.stride(2),
            Sq, Sk,
            sm_scale,
            HEAD_DIM=HEAD_DIM, IS_CAUSAL=causal,
        )

        O = O_3d.reshape(B, Hq, Sq, D)

        ctx.save_for_backward(Q_3d, K_3d, V_3d, O_3d, LSE, kv_head_map)
        ctx.causal = causal
        ctx.shape = (B, Hq, Hkv, Sq, Sk, D)
        ctx.sm_scale = sm_scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q_3d, K_3d, V_3d, O_3d, LSE, kv_head_map = ctx.saved_tensors
        B, Hq, Hkv, Sq, Sk, D = ctx.shape

        dO_3d = dO.reshape(B * Hq, Sq, D).contiguous()

        HEAD_DIM = triton.next_power_of_2(D)
        BLOCK_M_pre = min(128, triton.next_power_of_2(Sq))

        # Precompute Delta = rowsum(O * dO)
        Delta = torch.empty(B * Hq, Sq, device=dO.device, dtype=torch.float32)
        grid_pre = (triton.cdiv(Sq, BLOCK_M_pre), B * Hq)
        flash_attn_bwd_preprocess[grid_pre](
            O_3d, dO_3d, Delta,
            O_3d.stride(1), O_3d.stride(2),
            Sq,
            HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M_pre,
        )

        # Allocate grads
        dQ_3d = torch.empty_like(Q_3d)
        dK_3d = torch.zeros_like(K_3d)  # zero: accumulated via atomic_add from multiple Q heads
        dV_3d = torch.zeros_like(V_3d)  # zero: same reason

        grid = lambda META: (triton.cdiv(Sq, META["BLOCK_M"]), B * Hq)

        flash_attn_bwd_kernel[grid](
            Q_3d, K_3d, V_3d, dO_3d,
            dQ_3d, dK_3d, dV_3d,
            LSE, Delta,
            kv_head_map,
            Q_3d.stride(1), Q_3d.stride(2),
            K_3d.stride(1), K_3d.stride(2),
            V_3d.stride(1), V_3d.stride(2),
            O_3d.stride(1), O_3d.stride(2),
            Sq, Sk,
            ctx.sm_scale,
            HEAD_DIM=HEAD_DIM, IS_CAUSAL=ctx.causal,
        )

        dQ = dQ_3d.reshape(B, Hq, Sq, D)
        dK = dK_3d.reshape(B, Hkv, Sk, D)
        dV = dV_3d.reshape(B, Hkv, Sk, D)

        return dQ, dK, dV, None  # None for causal


def _flash_attention_fa2(Q, K, V, causal):
    """FlashAttention-2 backend. Expects [B, H, S, D], converts to [B, S, H, D]."""
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape

    # FA2 expects [B, S, H, D]
    q = Q.transpose(1, 2).contiguous()
    k = K.transpose(1, 2).contiguous()
    v = V.transpose(1, 2).contiguous()

    # FA2 handles GQA natively when num_heads_q != num_heads_k
    out = _fa2_func(q, k, v, causal=causal)  # [B, S, H, D]
    return out.transpose(1, 2)  # [B, H, S, D]


def _flash_attention_sdpa(Q, K, V, causal):
    """PyTorch SDPA backend with native GQA support (PyTorch 2.6+).

    Uses enable_gqa=True to let SDPA handle Hq != Hkv internally,
    avoiding the expensive K/V unsqueeze+expand+reshape+contiguous copies.
    """
    import torch.nn.functional as F
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape
    return F.scaled_dot_product_attention(
        Q, K, V, is_causal=causal, enable_gqa=(Hq != Hkv),
    )


def _flash_attention_triton_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_head_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton decode-only fast path for Sq=1 autoregressive inference."""
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape
    assert Sq == 1, f"decode fast path expects Sq=1, got {Sq}"
    assert D in (16, 32, 64, 128, 256), f"HEAD_DIM must be power of 2, got {D}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"

    Q_3d = Q.reshape(B * Hq, Sq, D).contiguous()
    K_3d = K.reshape(B * Hkv, Sk, D).contiguous()
    V_3d = V.reshape(B * Hkv, Sk, D).contiguous()
    O_2d = torch.empty(B * Hq, D, device=Q.device, dtype=Q.dtype)

    kv_head_map = _get_kv_head_map(B, Hq, Hkv, Q.device, kv_head_map)
    sm_scale = 1.0 / math.sqrt(D)
    HEAD_DIM = triton.next_power_of_2(D)

    flash_attn_decode_kernel[(B * Hq,)](
        Q_3d, K_3d, V_3d, O_2d,
        kv_head_map,
        Q_3d.stride(1), Q_3d.stride(2),
        K_3d.stride(1), K_3d.stride(2),
        V_3d.stride(1), V_3d.stride(2),
        O_2d.stride(0), O_2d.stride(1),
        Sk,
        sm_scale,
        HEAD_DIM=HEAD_DIM,
    )
    return O_2d.view(B, Hq, 1, D)


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    """Flash Attention with GQA support and multiple backend dispatch.

    Args:
        Q: [B, Hq, Sq, D] queries
        K: [B, Hkv, Sk, D] keys
        V: [B, Hkv, Sk, D] values
        causal: whether to apply causal masking
        backend: "auto" | "flash_attn" | "flashinfer" | "triton" | "sdpa"

    Returns:
        Output: [B, Hq, Sq, D]
    """
    backend = resolve_attention_backend(backend)

    if backend == "flash_attn":
        return _flash_attention_fa2(Q, K, V, causal)
    elif backend == "flashinfer":
        # FlashInfer prefill: use SDPA as fallback for prefill path
        return _flash_attention_sdpa(Q, K, V, causal)
    elif backend == "sdpa":
        return _flash_attention_sdpa(Q, K, V, causal)
    else:
        # "triton" or fallback
        return FlashAttentionFunction.apply(Q, K, V, causal)


def _flash_attention_flashinfer_decode(Q, K, V):
    """FlashInfer decode backend for Sq=1 with GQA support.

    FlashInfer's single_decode_with_kv_cache uses specialized decode kernels
    that are significantly faster than SDPA for single-query attention.

    Args:
        Q: [B, Hq, 1, D]
        K: [B, Hkv, Sk, D]
        V: [B, Hkv, Sk, D]

    Returns:
        Output: [B, Hq, 1, D]
    """
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, Dv = V.shape
    assert Sq == 1, f"FlashInfer decode expects Sq=1, got {Sq}"

    # Process each batch element separately since flashinfer.single_decode
    # expects unbatched inputs: q=[Hq, D], k=[Sk, Hkv, D], v=[Sk, Hkv, Dv]
    outputs = []
    for b in range(B):
        q_b = Q[b, :, 0, :]        # [Hq, D]
        k_b = K[b].transpose(0, 1)  # [Sk, Hkv, D]  (HND -> NHD)
        v_b = V[b].transpose(0, 1)  # [Sk, Hkv, Dv] (HND -> NHD)
        # single_decode_with_kv_cache returns [Hq, Dv] with kv_layout="NHD"
        o_b = _flashinfer_decode(
            q_b.contiguous(),
            k_b.contiguous(),
            v_b.contiguous(),
            kv_layout="NHD",
        )  # [Hq, Dv]
        outputs.append(o_b)
    # Stack: [B, Hq, Dv] -> [B, Hq, 1, Dv]
    return torch.stack(outputs, dim=0).unsqueeze(2)


def flash_attention_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    backend: str = "auto",
    kv_head_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode-only attention path for Sq=1 autoregressive inference."""
    backend = resolve_attention_backend(backend)

    if backend == "flashinfer":
        return _flash_attention_flashinfer_decode(Q, K, V)
    if backend == "flash_attn":
        return _flash_attention_fa2(Q, K, V, causal=False)
    if backend == "sdpa":
        return _flash_attention_sdpa(Q, K, V, causal=False)
    return _flash_attention_triton_decode(Q, K, V, kv_head_map=kv_head_map)


def flash_attention_naive(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Reference attention using PyTorch SDPA, for testing."""
    import torch.nn.functional as F
    # SDPA expects [B, H, S, D] and handles GQA via broadcasting
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape
    groups = Hq // Hkv
    # Expand K/V to match Q heads
    K_exp = K.unsqueeze(2).expand(B, Hkv, groups, Sk, D).reshape(B, Hq, Sk, D)
    V_exp = V.unsqueeze(2).expand(B, Hkv, groups, Sk, D).reshape(B, Hq, Sk, D)
    return F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=causal)
