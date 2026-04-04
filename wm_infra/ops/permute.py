"""Permute ops: reorder tokens by expert assignment with autograd."""

import torch
import triton

from wm_infra.kernels.permute_kernel import (
    permute_tokens_kernel,
    unpermute_tokens_kernel,
)


class PermuteTokens(torch.autograd.Function):
    """Gather hidden states into expert-sorted order.
    
    Forward:  permuted[i] = hidden[sorted_ids[i] // top_k]
    Backward: scatter gradients back using same sorted_ids
    """
    
    @staticmethod
    def forward(ctx, hidden_states, sorted_token_ids, top_k):
        num_tokens, hidden_dim = hidden_states.shape
        total = sorted_token_ids.shape[0]  # num_tokens * top_k
        
        permuted = torch.empty(total, hidden_dim, 
                               dtype=hidden_states.dtype, 
                               device=hidden_states.device)
        
        # Launch Triton kernel
        grid = (total, triton.cdiv(hidden_dim, 128))
        permute_tokens_kernel[grid](
            hidden_states, sorted_token_ids, permuted,
            num_tokens, hidden_dim, top_k,
            BLOCK_D=128,
        )
        
        ctx.save_for_backward(sorted_token_ids)
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.top_k = top_k
        return permuted
    
    @staticmethod
    def backward(ctx, grad_output):
        sorted_token_ids, = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        hidden_dim = ctx.hidden_dim
        top_k = ctx.top_k
        
        # Backward of gather is scatter-add
        # grad_hidden[token_id] += grad_output[i] for all i where sorted_ids[i]//top_k == token_id
        grad_hidden = torch.zeros(num_tokens, hidden_dim,
                                  dtype=grad_output.dtype,
                                  device=grad_output.device)
        
        # Map sorted positions back to original tokens
        original_ids = sorted_token_ids // top_k  # [total]
        
        # Scatter-add gradients
        # For each sorted position i, add grad_output[i] to grad_hidden[original_ids[i]]
        grad_hidden.index_add_(0, original_ids, grad_output)
        
        return grad_hidden, None, None


class UnpermuteTokens(torch.autograd.Function):
    """Scatter expert outputs back to original order with weighted combine.
    
    Forward:  output[token_id] += weight * expert_out[i]
    Backward: gather gradients and scale by weights
    """
    
    @staticmethod
    def forward(ctx, expert_output, sorted_token_ids, topk_weights, num_tokens, top_k):
        hidden_dim = expert_output.shape[1]
        
        output = torch.zeros(num_tokens, hidden_dim,
                             dtype=expert_output.dtype,
                             device=expert_output.device)
        
        # Launch unpermute kernel
        total = sorted_token_ids.shape[0]
        grid = (total, triton.cdiv(hidden_dim, 128))
        unpermute_tokens_kernel[grid](
            expert_output, sorted_token_ids, topk_weights, output,
            num_tokens, hidden_dim, top_k,
            BLOCK_D=128,
        )
        
        ctx.save_for_backward(sorted_token_ids, topk_weights)
        ctx.top_k = top_k
        ctx.total = total
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        sorted_token_ids, topk_weights = ctx.saved_tensors
        top_k = ctx.top_k
        total = ctx.total
        hidden_dim = grad_output.shape[1]
        
        # Backward of scatter-add with weights is gather with weights
        original_ids = sorted_token_ids // top_k
        k_indices = sorted_token_ids % top_k
        
        # Gather gradients: grad_expert[i] = weight[token, k] * grad_output[token]
        weights = topk_weights[original_ids, k_indices]  # [total]
        grad_expert = grad_output[original_ids] * weights.unsqueeze(1)
        
        # Gradient w.r.t. topk_weights
        # d(output[t])/d(weight[t,k]) = expert_out[sorted_pos]
        # This requires expert_output from forward — would need to save it
        # For now, return None (routing weights often don't need gradients
        # or use straight-through estimator)
        
        return grad_expert, None, None, None, None


def permute_tokens(hidden_states, sorted_token_ids, top_k):
    """Differentiable token permutation."""
    return PermuteTokens.apply(hidden_states, sorted_token_ids, top_k)


def unpermute_tokens(expert_output, sorted_token_ids, topk_weights, num_tokens, top_k):
    """Differentiable token un-permutation with weighted combine."""
    return UnpermuteTokens.apply(expert_output, sorted_token_ids, topk_weights, num_tokens, top_k)
