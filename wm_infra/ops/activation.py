"""Fused activation ops with autograd."""

import torch
import triton

from wm_infra.kernels.activation_kernel import (
    swiglu_fwd_kernel,
    swiglu_bwd_kernel,
    swiglu_fused_gate_up_kernel,
)


class FusedSwiGLU(torch.autograd.Function):
    """Fused SwiGLU: output = SiLU(gate) * up"""
    
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        assert gate.is_contiguous() and up.is_contiguous()
        
        output = torch.empty_like(gate)
        num_elements = gate.numel()
        
        BLOCK = 1024
        grid = (triton.cdiv(num_elements, BLOCK),)
        
        swiglu_fwd_kernel[grid](
            gate, up, output,
            num_elements,
            BLOCK=BLOCK,
        )
        
        ctx.save_for_backward(gate, up)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        num_elements = gate.numel()

        BLOCK = 1024
        grid = (triton.cdiv(num_elements, BLOCK),)

        swiglu_bwd_kernel[grid](
            gate, up, grad_output,
            dgate, dup,
            num_elements,
            BLOCK=BLOCK,
        )

        return dgate, dup


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation.
    
    Computes: SiLU(gate) * up = (gate * sigmoid(gate)) * up
    
    2x less memory traffic than separate silu + mul.
    """
    return FusedSwiGLU.apply(gate, up)


def swiglu_fused_gate_up(gate_up: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU from a contiguous gate_up buffer (inference only, no autograd).

    Args:
        gate_up: [top_k, 2*I] contiguous tensor where [:, :I] is gate, [:, I:] is up.
        output: [top_k, I] pre-allocated output buffer to write activated result into.

    Returns:
        output tensor (same as the ``output`` argument).
    """
    assert gate_up.is_contiguous()
    total_rows = gate_up.shape[0]
    two_I = gate_up.shape[1]
    half_N = two_I // 2
    num_elements = total_rows * half_N

    BLOCK = 1024
    grid = (triton.cdiv(num_elements, BLOCK),)

    swiglu_fused_gate_up_kernel[grid](
        gate_up, output,
        half_N, total_rows,
        BLOCK=BLOCK,
    )
    return output


def swiglu_naive(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Naive reference: for correctness testing."""
    return torch.nn.functional.silu(gate) * up
