"""Permute kernels: reorder tokens by expert assignment.

The MoE forward pass needs tokens grouped by expert so that GroupedGEMM
can process each expert's tokens contiguously. This module provides:

1. moe_align_sort_kernel: Compute sorted indices + expert offsets
2. permute_tokens_kernel: Gather hidden states into sorted order
3. unpermute_tokens_kernel: Scatter results back to original order + weighted combine

This is the "MoE Align & Sort" algorithm from SGLang.
"""

import triton
import triton.language as tl


@triton.jit
def moe_align_sort_kernel(
    # Inputs
    topk_ids_ptr,           # [num_tokens * top_k] — flat expert assignments
    expert_offsets_ptr,     # [num_experts + 1] — cumulative counts (read-only)
    expert_counters_ptr,    # [num_experts] — atomic write counters, zeroed before launch
    # Outputs
    sorted_token_ids_ptr,   # [num_tokens * top_k] — reordered token indices
    # Shapes
    num_tokens,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    # Tuning
    BLOCK_T: tl.constexpr = 512,
):
    """Scatter token indices into expert-sorted order.

    Each thread:
      1. Loads expert_id = topk_ids[position]
      2. Claims a slot via atomic_add on expert_counters[expert_id]
      3. Writes position to sorted_token_ids[expert_offsets[expert_id] + slot]

    Position i in the flat topk_ids array encodes token_id = i // top_k,
    k_idx = i % top_k, so storing i preserves the encoding for downstream kernels.

    Within-expert ordering is non-deterministic (atomic contention) — this is fine
    since GroupedGEMM treats all tokens within an expert identically.
    """
    pid = tl.program_id(0)
    total = num_tokens * top_k

    t_start = pid * BLOCK_T
    t_range = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_range < total

    # Load expert assignments for this block
    expert_ids = tl.load(topk_ids_ptr + t_range, mask=t_mask, other=0)

    # Load base offsets for each expert
    base = tl.load(expert_offsets_ptr + expert_ids, mask=t_mask, other=0)

    # Atomically claim a slot within each expert's segment
    slot = tl.atomic_add(expert_counters_ptr + expert_ids, 1, mask=t_mask)

    # Write position (flat index) to the sorted output
    tl.store(sorted_token_ids_ptr + base + slot, t_range, mask=t_mask)


@triton.jit
def permute_tokens_kernel(
    # Inputs
    hidden_ptr,             # [num_tokens, hidden_dim] — original hidden states
    sorted_ids_ptr,         # [num_tokens * top_k] — which token to read
    # Output
    permuted_ptr,           # [num_tokens * top_k, hidden_dim] — sorted hidden states
    # Shapes
    num_tokens,
    hidden_dim: tl.constexpr,
    top_k: tl.constexpr,
    # Tuning
    BLOCK_D: tl.constexpr = 128,
):
    """Gather hidden states into expert-sorted order.
    
    permuted[i] = hidden[sorted_ids[i] // top_k]
    
    This is essentially an indexed gather along dim=0.
    We tile along hidden_dim for coalesced memory access.
    """
    pid_t = tl.program_id(0)  # which sorted token
    pid_d = tl.program_id(1)  # which hidden_dim tile
    
    total = num_tokens * top_k
    if pid_t >= total:
        return
    
    # Which original token does this sorted position read from?
    sorted_id = tl.load(sorted_ids_ptr + pid_t)
    token_id = sorted_id // top_k  # map back to original token
    
    # Load hidden_dim tile from original position
    d_range = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_range < hidden_dim
    
    src_offset = token_id * hidden_dim + d_range
    vals = tl.load(hidden_ptr + src_offset, mask=d_mask)
    
    # Store to sorted position
    dst_offset = pid_t * hidden_dim + d_range
    tl.store(permuted_ptr + dst_offset, vals, mask=d_mask)


@triton.jit
def unpermute_tokens_kernel(
    # Inputs
    expert_out_ptr,         # [num_tokens * top_k, hidden_dim] — expert outputs (sorted)
    sorted_ids_ptr,         # [num_tokens * top_k] — reverse mapping
    topk_weights_ptr,       # [num_tokens, top_k] — routing weights
    # Output
    output_ptr,             # [num_tokens, hidden_dim] — combined output (atomic add)
    # Shapes
    num_tokens,
    hidden_dim: tl.constexpr,
    top_k: tl.constexpr,
    # Tuning
    BLOCK_D: tl.constexpr = 128,
):
    """Scatter expert outputs back to original order with weighted combination.
    
    For each sorted position i:
      token_id = sorted_ids[i] // top_k
      k_idx = sorted_ids[i] % top_k
      weight = topk_weights[token_id, k_idx]
      output[token_id] += weight * expert_out[i]
    
    Uses atomic adds because multiple experts write to the same token.
    
    NOTE: For backward pass, this kernel's "reverse" is permute_tokens
    with gradient scaling — no separate kernel needed.
    """
    pid_t = tl.program_id(0)  # which sorted position
    pid_d = tl.program_id(1)  # which hidden_dim tile
    
    total = num_tokens * top_k
    if pid_t >= total:
        return
    
    # Reverse mapping
    sorted_id = tl.load(sorted_ids_ptr + pid_t)
    token_id = sorted_id // top_k
    k_idx = sorted_id % top_k
    
    # Load routing weight for this (token, expert) pair
    weight = tl.load(topk_weights_ptr + token_id * top_k + k_idx)
    
    # Load expert output tile
    d_range = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_range < hidden_dim
    
    expert_vals = tl.load(expert_out_ptr + pid_t * hidden_dim + d_range, mask=d_mask)
    weighted = expert_vals * weight
    
    # Atomic add to output (multiple experts contribute to same token)
    out_offset = token_id * hidden_dim + d_range
    tl.atomic_add(output_ptr + out_offset, weighted, mask=d_mask)
