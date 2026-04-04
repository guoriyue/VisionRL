"""End-to-end fused MoE operation.

Two modes:
  1. Composable: individual ops chained together (for debugging / flexibility)
  2. Fused: 2-stage kernel (gate+up+SwiGLU, then down+scatter)

Training always uses composable mode (need intermediate activations for backward).
Inference can use either; fused is faster at small batch (decode), composable
is faster at large batch (prefill) due to atomic_add contention in fused mode.
When mode="fused", auto-fallback to composable at large batch sizes.
"""

import torch
from typing import Optional, Tuple

from wm_infra.ops.routing import topk_route, compute_expert_offsets
from wm_infra.ops.permute import permute_tokens, unpermute_tokens
from wm_infra.ops.group_gemm import grouped_gemm
from wm_infra.ops.activation import fused_swiglu


def fused_moe(
    hidden_states: torch.Tensor,      # [num_tokens, hidden_dim]
    gate_weight: torch.Tensor,        # [num_experts, hidden_dim]
    w_gate: torch.Tensor,             # [num_experts, hidden_dim, intermediate_dim]
    w_up: torch.Tensor,               # [num_experts, hidden_dim, intermediate_dim]
    w_down: torch.Tensor,             # [num_experts, intermediate_dim, hidden_dim]
    top_k: int = 2,
    renormalize: bool = True,
    aux_loss_weight: float = 0.0,
    mode: str = "composable",         # "composable" | "fused" | "naive"
    expert_bias: Optional[torch.Tensor] = None,  # [num_experts] bias for selection
    decode_mode: bool = False,        # skip GPU→CPU syncs in GEMM ops
    decode_buffers: Optional[dict] = None,  # pre-allocated buffers for decode
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Full MoE forward pass.

    Args:
        hidden_states: input from transformer layer
        gate_weight: router projection weights
        w_gate, w_up, w_down: expert FFN weights (SwiGLU: gate + up -> activate -> down)
        top_k: number of experts per token
        renormalize: renormalize top-k weights to sum to 1
        aux_loss_weight: weight for load balancing loss (0 = no loss)
        mode: execution mode
            "naive"      - PyTorch loops, for correctness testing
            "composable" - Triton kernels, separate launches, autograd works
            "fused"      - 2-stage fused kernel, inference only (no backward)
        expert_bias: optional [num_experts] bias added to logits before top-k selection
                     (DeepSeek-V3 style aux-loss-free load balancing)
        decode_mode: skip GPU→CPU syncs in GEMM tile mapping (safe at small batch)

    Returns:
        output: [num_tokens, hidden_dim]
        aux_loss: scalar or None
    """
    num_experts = gate_weight.shape[0]
    num_tokens = hidden_states.shape[0]

    # Fused mode auto-fallback: at large batch sizes, tl.atomic_add scatter
    # in fused_down_kernel causes contention that makes fused 2x slower than
    # composable. Threshold 256 is the empirical crossover point (RTX 5090).
    _FUSED_FALLBACK_THRESHOLD = 256

    if mode == "naive":
        return _moe_naive(hidden_states, gate_weight, w_gate, w_up, w_down,
                          top_k, num_experts, renormalize, aux_loss_weight,
                          expert_bias)
    elif mode == "composable":
        return _moe_composable(hidden_states, gate_weight, w_gate, w_up, w_down,
                               top_k, num_experts, renormalize, aux_loss_weight,
                               expert_bias, decode_mode=decode_mode)
    elif mode == "fused":
        if num_tokens > _FUSED_FALLBACK_THRESHOLD:
            return _moe_composable(hidden_states, gate_weight, w_gate, w_up, w_down,
                                   top_k, num_experts, renormalize, aux_loss_weight,
                                   expert_bias, decode_mode=decode_mode)
        return _moe_fused(hidden_states, gate_weight, w_gate, w_up, w_down,
                          top_k, num_experts, renormalize, aux_loss_weight,
                          expert_bias, decode_mode=decode_mode,
                          decode_buffers=decode_buffers)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _moe_naive(hidden_states, gate_weight, w_gate, w_up, w_down,
               top_k, num_experts, renormalize, aux_loss_weight,
               expert_bias=None):
    """Pure PyTorch reference. Slow but definitely correct."""
    num_tokens, hidden_dim = hidden_states.shape
    intermediate_dim = w_gate.shape[2]

    # Route
    topk_weights, topk_ids, aux_loss = topk_route(
        hidden_states, gate_weight, top_k, renormalize,
        aux_loss_weight=aux_loss_weight, expert_bias=expert_bias)

    # Process each token
    output = torch.zeros_like(hidden_states)

    for t in range(num_tokens):
        for k in range(top_k):
            expert_id = topk_ids[t, k].item()
            weight = topk_weights[t, k]

            # Expert FFN: SwiGLU
            gate_out = hidden_states[t] @ w_gate[expert_id]     # [intermediate_dim]
            up_out = hidden_states[t] @ w_up[expert_id]         # [intermediate_dim]
            activated = torch.nn.functional.silu(gate_out) * up_out
            expert_out = activated @ w_down[expert_id]           # [hidden_dim]

            output[t] += weight * expert_out

    return output, aux_loss


def _moe_composable(hidden_states, gate_weight, w_gate, w_up, w_down,
                    top_k, num_experts, renormalize, aux_loss_weight,
                    expert_bias=None, decode_mode=False,
):
    """Triton kernels, separate launches. Supports backward (training)."""
    num_tokens, hidden_dim = hidden_states.shape

    # Step 1: Route
    topk_weights, topk_ids, aux_loss = topk_route(
        hidden_states, gate_weight, top_k, renormalize,
        aux_loss_weight=aux_loss_weight, expert_bias=expert_bias)

    # Step 2: Compute expert offsets (sort tokens by expert)
    sorted_token_ids, expert_offsets, expert_counts = compute_expert_offsets(
        topk_ids, num_experts)

    # Step 3: Permute hidden states into expert order
    permuted = permute_tokens(hidden_states, sorted_token_ids, top_k)

    # Step 4: Grouped GEMM — gate projection
    gate_out = grouped_gemm(permuted, w_gate, expert_offsets, num_experts,
                            decode_mode=decode_mode)

    # Step 5: Grouped GEMM — up projection
    up_out = grouped_gemm(permuted, w_up, expert_offsets, num_experts,
                          decode_mode=decode_mode)

    # Step 6: Fused SwiGLU activation
    activated = fused_swiglu(gate_out, up_out)

    # Step 7: Grouped GEMM — down projection
    expert_out = grouped_gemm(activated, w_down, expert_offsets, num_experts,
                              decode_mode=decode_mode)

    # Step 8: Unpermute + weighted combine
    output = unpermute_tokens(expert_out, sorted_token_ids, topk_weights, num_tokens, top_k)

    return output, aux_loss


def _moe_fused(hidden_states, gate_weight, w_gate, w_up, w_down,
               top_k, num_experts, renormalize, aux_loss_weight,
               expert_bias=None, decode_mode=False, decode_buffers=None):
    """2-stage fused kernel. Inference only (no backward support).

    Stage 1: fused_gate_up_kernel — gather + gate/up GEMM + SwiGLU
    Stage 2: fused_down_kernel    — down GEMM + weighted scatter

    Args:
        decode_buffers: Optional pre-allocated buffer dict to eliminate
            tensor allocation overhead during decode. Keys:
            'topk_weights', 'topk_ids', 'expert_counts', 'expert_offsets',
            'expert_counters', 'sorted_token_ids', 'tile_expert_ids',
            'output', 'intermediate'.
    """
    from wm_infra.kernels.fused_moe_kernel import fused_gate_up_kernel, fused_down_kernel
    from wm_infra.ops.group_gemm import _build_tile_mapping, _select_block_m
    import triton

    num_tokens, hidden_dim = hidden_states.shape
    intermediate_dim = w_gate.shape[2]
    total_sorted = num_tokens * top_k

    if decode_buffers is not None:
        # Fast path: use pre-allocated buffers, skip all tensor allocations
        topk_weights, topk_ids, aux_loss = topk_route(
            hidden_states, gate_weight, top_k, renormalize,
            aux_loss_weight=aux_loss_weight, expert_bias=expert_bias)

        sorted_token_ids, expert_offsets, expert_counts = compute_expert_offsets(
            topk_ids, num_experts)

        # Reuse pre-allocated tile mapping (static for decode)
        tile_expert_ids = decode_buffers['tile_expert_ids']

        # Build tile offsets from expert_offsets (no sync, no allocation)
        starts = expert_offsets[:num_experts]
        ends = expert_offsets[1:num_experts + 1]
        tile_m_offsets = decode_buffers['tile_m_offsets']
        tile_m_ends = decode_buffers['tile_m_ends']
        tile_m_offsets.copy_(starts)  # in-place, no new tensor
        tile_m_ends.copy_(ends)

        BLOCK_M = 16
        total_m_tiles = num_experts

        # Reuse output buffer (must zero for atomic_add)
        output = decode_buffers['output']
        output.zero_()

        intermediate = decode_buffers['intermediate']
    else:
        # Standard path: allocate tensors
        topk_weights, topk_ids, aux_loss = topk_route(
            hidden_states, gate_weight, top_k, renormalize,
            aux_loss_weight=aux_loss_weight, expert_bias=expert_bias)

        sorted_token_ids, expert_offsets, expert_counts = compute_expert_offsets(
            topk_ids, num_experts)

        BLOCK_M = _select_block_m(expert_offsets, num_experts,
                                  decode_mode=decode_mode)

        tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles = \
            _build_tile_mapping(expert_offsets, num_experts, BLOCK_M,
                                hidden_states.device, decode_mode=decode_mode)

        output = torch.zeros_like(hidden_states)

        if total_m_tiles == 0:
            return output, aux_loss

        intermediate = torch.empty(
            total_sorted, intermediate_dim,
            dtype=hidden_states.dtype, device=hidden_states.device)

    # Stage 1 — gate+up GEMM + SwiGLU
    def grid_stage1(META):
        return (total_m_tiles * triton.cdiv(intermediate_dim, META["BLOCK_INTER"]),)

    fused_gate_up_kernel[grid_stage1](
        hidden_states,
        w_gate, w_up,
        sorted_token_ids,
        tile_expert_ids, tile_m_offsets, tile_m_ends,
        intermediate,
        hidden_dim, intermediate_dim, top_k, total_m_tiles,
        hidden_states.stride(0), hidden_states.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        intermediate.stride(0), intermediate.stride(1),
        BLOCK_M=BLOCK_M,
    )

    # Stage 2 — down GEMM + weighted scatter
    def grid_stage2(META):
        return (total_m_tiles * triton.cdiv(hidden_dim, META["BLOCK_HIDDEN"]),)

    fused_down_kernel[grid_stage2](
        intermediate,
        w_down,
        sorted_token_ids, topk_weights,
        tile_expert_ids, tile_m_offsets, tile_m_ends,
        output,
        hidden_dim, intermediate_dim, top_k, total_m_tiles,
        intermediate.stride(0), intermediate.stride(1),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
    )

    return output, aux_loss
