"""Routing ops: top-k expert selection with auxiliary losses."""

import torch
import torch.nn.functional as F
import triton
from typing import Tuple, Optional

from wm_infra.kernels.routing_kernel import topk_softmax_kernel


def _triton_topk_softmax(
    logits: torch.Tensor,  # [num_tokens, num_experts]
    top_k: int,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch topk_softmax_kernel and return (topk_weights, topk_ids)."""
    num_tokens, num_experts = logits.shape
    topk_weights = torch.empty(num_tokens, top_k, device=logits.device, dtype=logits.dtype)
    topk_ids = torch.empty(num_tokens, top_k, device=logits.device, dtype=torch.int64)
    BLOCK_E = triton.next_power_of_2(num_experts)
    BLOCK_K = triton.next_power_of_2(top_k)
    topk_softmax_kernel[(num_tokens,)](
        logits, topk_weights, topk_ids,
        num_tokens, num_experts=num_experts, top_k=top_k,
        renormalize=renormalize,
        BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
    )
    return topk_weights, topk_ids


def topk_route(
    hidden_states: torch.Tensor,     # [num_tokens, hidden_dim]
    gate_weight: torch.Tensor,       # [num_experts, hidden_dim]
    top_k: int,
    renormalize: bool = True,
    jitter: float = 0.0,
    aux_loss_weight: float = 0.0,
    expert_bias: Optional[torch.Tensor] = None,  # [num_experts] bias for selection
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute top-k routing with optional auxiliary loss.

    When expert_bias is provided (DeepSeek-V3 style), the bias is added to
    logits BEFORE top-k selection to influence which experts are chosen, but
    the routing WEIGHTS are computed from the original un-biased logits via
    softmax. This shifts expert selection without distorting weight distribution.

    Returns:
        topk_weights: [num_tokens, top_k] — normalized routing weights
        topk_ids:     [num_tokens, top_k] — expert indices (int64)
        aux_loss:     scalar or None — load balancing loss for training
    """
    num_tokens = hidden_states.shape[0]
    num_experts = gate_weight.shape[0]

    # Router logits (cast gate to match input dtype for FP8 path where gate stays fp32)
    if gate_weight.dtype != hidden_states.dtype:
        gate_weight = gate_weight.to(hidden_states.dtype)
    logits = hidden_states @ gate_weight.T  # [num_tokens, num_experts]

    # Optional jitter for training stability
    if jitter > 0 and (hidden_states.requires_grad or gate_weight.requires_grad):
        noise = torch.randn_like(logits) * jitter
        logits = logits + noise

    # Bias-based selection (DeepSeek-V3 style):
    # Add bias to logits for top-k selection, but use original logits for weights.
    if expert_bias is not None:
        biased_logits = logits + expert_bias.unsqueeze(0)  # [num_tokens, num_experts]
    else:
        biased_logits = logits

    # Auxiliary load balancing loss (needs full softmax for gradient flow to gate)
    aux_loss = None
    if aux_loss_weight > 0 and (hidden_states.requires_grad or gate_weight.requires_grad):
        scores = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]
        # Use biased_logits for selection (determines which experts are picked)
        topk_weights, topk_ids = _triton_topk_softmax(biased_logits, top_k, renormalize)
        if expert_bias is not None:
            # Recompute weights from un-biased softmax for the selected experts
            topk_weights = _recompute_weights_from_ids(logits, topk_ids, renormalize)
        aux_loss = _load_balance_loss(scores, topk_ids, num_experts) * aux_loss_weight
    else:
        # Inference path: fused kernel only — 1 launch instead of 3
        topk_weights, topk_ids = _triton_topk_softmax(biased_logits, top_k, renormalize)
        if expert_bias is not None:
            # Recompute weights from un-biased softmax for the selected experts
            topk_weights = _recompute_weights_from_ids(logits, topk_ids, renormalize)

    return topk_weights, topk_ids, aux_loss


def _load_balance_loss(
    scores: torch.Tensor,    # [num_tokens, num_experts] — softmax scores
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    num_experts: int,
) -> torch.Tensor:
    """Switch Transformer load balancing loss.
    
    L = num_experts * sum_e(f_e * P_e)
    where:
      f_e = fraction of tokens assigned to expert e
      P_e = mean routing probability for expert e
    
    Minimizing this encourages uniform expert utilization.
    """
    num_tokens = scores.shape[0]
    
    # f_e: fraction of tokens routed to each expert
    # Count how many times each expert appears in topk_ids
    expert_mask = torch.zeros(num_tokens, num_experts, device=scores.device)
    expert_mask.scatter_(1, topk_ids, 1.0)
    f = expert_mask.mean(dim=0)  # [num_experts]
    
    # P_e: mean routing probability for each expert
    P = scores.mean(dim=0)  # [num_experts]
    
    return num_experts * (f * P).sum()


def _recompute_weights_from_ids(
    logits: torch.Tensor,     # [num_tokens, num_experts] — original un-biased logits
    topk_ids: torch.Tensor,   # [num_tokens, top_k]
    renormalize: bool = True,
) -> torch.Tensor:
    """Recompute routing weights from un-biased logits for given expert ids.

    When expert_bias shifts selection, the top-k ids come from biased logits
    but the weights should reflect the original (un-biased) softmax probabilities.
    """
    # Full softmax over un-biased logits
    scores = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

    # Gather weights for the selected experts
    topk_weights = scores.gather(1, topk_ids)  # [num_tokens, top_k]

    # Renormalize so selected weights sum to 1
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights


def update_expert_bias(
    expert_bias: torch.Tensor,    # [num_experts] — updated in-place
    expert_counts: torch.Tensor,  # [num_experts] — int tensor from compute_expert_offsets
    num_experts: int,
    target_load: Optional[float] = None,
    lr: float = 0.001,
) -> None:
    """Update expert bias to encourage balanced load (DeepSeek-V3 style).

    If expert e is overloaded (count > target), decrease bias[e].
    If expert e is underloaded (count < target), increase bias[e].

    Args:
        expert_bias: [num_experts] tensor to update in-place
        expert_counts: [num_experts] int tensor from compute_expert_offsets
        num_experts: int
        target_load: target per-expert count (default: mean of expert_counts)
        lr: update learning rate for bias adjustment
    """
    counts = expert_counts.float()
    if target_load is None:
        target_load = counts.mean().item()

    # Positive error = underloaded, negative error = overloaded
    error = target_load - counts
    expert_bias.add_(lr * error)


def compute_expert_offsets(
    topk_ids: torch.Tensor,   # [num_tokens, top_k]
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sorted token indices and expert offsets for GroupedGEMM.

    This is the "Align & Sort" algorithm, fully on GPU:
      Phase 1: compute_expert_counts_kernel → expert_counts [num_experts]
      Phase 2: expert_offsets = cumsum (tiny, ≤256 elements)
      Phase 3: moe_align_sort_kernel → sorted_token_ids [num_tokens * top_k]

    Returns:
        sorted_token_ids: [num_tokens * top_k] — flat index into (token, k) space
        expert_offsets:    [num_experts + 1] — cumulative counts
        expert_counts:     [num_experts] — per-expert token counts
    """
    import triton
    from wm_infra.kernels.routing_kernel import compute_expert_counts_kernel
    from wm_infra.kernels.permute_kernel import moe_align_sort_kernel

    num_tokens, top_k = topk_ids.shape
    total = num_tokens * top_k
    device = topk_ids.device

    # Flatten topk_ids to [num_tokens * top_k]
    flat_ids = topk_ids.view(-1)

    # Phase 1: GPU histogram — count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    BLOCK_T = 256
    grid_counts = (triton.cdiv(num_tokens, BLOCK_T),)
    compute_expert_counts_kernel[grid_counts](
        flat_ids, expert_counts,
        num_tokens,
        top_k=top_k,
        BLOCK_T=BLOCK_T,
    )

    # Phase 2: cumsum for offsets (tiny — at most 256 elements)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    # Phase 3: GPU scatter — place token indices into expert-sorted order
    sorted_token_ids = torch.empty(total, dtype=torch.int64, device=device)
    expert_counters = torch.zeros(num_experts, dtype=torch.int32, device=device)
    BLOCK_T_SORT = 512
    grid_sort = (triton.cdiv(total, BLOCK_T_SORT),)
    moe_align_sort_kernel[grid_sort](
        flat_ids,
        expert_offsets,
        expert_counters,
        sorted_token_ids,
        num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        BLOCK_T=BLOCK_T_SORT,
    )

    return sorted_token_ids, expert_offsets, expert_counts
