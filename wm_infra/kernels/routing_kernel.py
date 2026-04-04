"""Routing kernels: top-k gating with softmax normalization.

Computes: topk_weights, topk_ids = topk(softmax(logits), k)

Why a custom kernel? PyTorch's topk + softmax is 2 kernel launches + 2
global memory round-trips. We fuse them into 1 launch.
"""

import triton
import triton.language as tl


@triton.jit
def topk_softmax_kernel(
    # Pointers
    logits_ptr,         # [num_tokens, num_experts]
    topk_weights_ptr,   # [num_tokens, top_k] — output
    topk_ids_ptr,       # [num_tokens, top_k] — output (int32)
    # Shapes
    num_tokens,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    # Optional
    router_jitter: tl.constexpr = 0.0,
    renormalize: tl.constexpr = True,
    # Block sizes
    BLOCK_E: tl.constexpr = 256,  # must be >= num_experts
    BLOCK_K: tl.constexpr = 8,    # must be >= top_k, power of 2
):
    """Fused top-k selection with softmax normalization.
    
    Each program handles one token (one row of logits).
    
    Algorithm:
      1. Load all expert logits for this token
      2. Compute softmax over all experts
      3. Find top-k by iterative argmax (k is small, typically 2-8)
      4. Optionally renormalize top-k weights to sum to 1
      5. Store topk_weights and topk_ids
    """
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return
    
    # ── Load logits for this token ──
    expert_range = tl.arange(0, BLOCK_E)
    mask = expert_range < num_experts
    logits_offset = token_id * num_experts + expert_range
    logits = tl.load(logits_ptr + logits_offset, mask=mask, other=float("-inf"))
    
    # ── Softmax ──
    # Numerically stable: subtract max first
    logits_max = tl.max(logits, axis=0)
    logits_exp = tl.exp(logits - logits_max)
    logits_sum = tl.sum(logits_exp, axis=0)
    softmax_out = logits_exp / logits_sum
    
    # ── Iterative top-k selection ──
    # For k <= 8, iterative argmax is fast enough
    # We mask out selected experts after each pick
    remaining = softmax_out
    
    for k_idx in tl.static_range(top_k):
        # Find argmax
        best_val = tl.max(remaining, axis=0)
        # Create mask for the max value (handles ties by picking first)
        is_best = (remaining == best_val) & mask
        # Get the index — use argmin on negated to get first occurrence
        best_idx = tl.argmax(remaining, axis=0)
        
        # Store result
        out_offset = token_id * top_k + k_idx
        tl.store(topk_weights_ptr + out_offset, best_val)
        tl.store(topk_ids_ptr + out_offset, best_idx.to(tl.int64))
        
        # Mask out selected expert
        remaining = tl.where(expert_range == best_idx, 0.0, remaining)
    
    # ── Renormalize top-k weights ──
    if renormalize:
        # Reload what we just wrote (top_k is tiny)
        # BLOCK_K is next-power-of-2 >= top_k, passed from launcher
        k_range = tl.arange(0, BLOCK_K)
        k_mask = k_range < top_k
        k_offset = token_id * top_k + k_range
        weights = tl.load(topk_weights_ptr + k_offset, mask=k_mask, other=0.0)
        weight_sum = tl.sum(weights, axis=0)
        weights = weights / weight_sum
        tl.store(topk_weights_ptr + k_offset, weights, mask=k_mask)


@triton.jit
def compute_expert_counts_kernel(
    topk_ids_ptr,           # [num_tokens, top_k]
    expert_counts_ptr,      # [num_experts] — output (int32), atomic adds
    num_tokens,
    top_k: tl.constexpr,
    BLOCK_T: tl.constexpr = 256,
):
    """Count how many tokens are assigned to each expert.
    
    Needed for computing expert_offsets for GroupedGEMM.
    Uses atomic adds — simple but correct.
    """
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    t_range = t_start + tl.arange(0, BLOCK_T)
    
    for k_idx in tl.static_range(top_k):
        t_mask = t_range < num_tokens
        ids = tl.load(topk_ids_ptr + t_range * top_k + k_idx, mask=t_mask, other=0)
        tl.atomic_add(expert_counts_ptr + ids, 1, mask=t_mask)
