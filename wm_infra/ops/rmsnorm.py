"""RMSNorm op with autograd support."""

import torch
import triton

from wm_infra.kernels.rmsnorm_kernel import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        assert x.is_contiguous()
        M, N = x.shape
        BLOCK_N = triton.next_power_of_2(N)

        y = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        rmsnorm_fwd_kernel[(M,)](
            x, weight, y, rstd,
            N, eps=eps, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.N = N
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        M, N = x.shape
        BLOCK_N = triton.next_power_of_2(N)

        dx = torch.empty_like(x)
        # Per-row partial dweight: [M, N]
        dw_partial = torch.empty_like(x)

        rmsnorm_bwd_kernel[(M,)](
            dy, x, weight, rstd,
            dx, dw_partial,
            N, BLOCK_N=BLOCK_N,
        )

        # Reduce dweight across rows
        dweight = dw_partial.sum(dim=0)

        return dx, dweight, None  # None for eps


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton-accelerated RMSNorm.

    Args:
        x: [*, N] input (will be reshaped to [M, N] internally)
        weight: [N] learnable scale
        eps: epsilon for numerical stability

    Returns:
        Normalized tensor, same shape as x
    """
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = RMSNormFunction.apply(x_2d, weight, eps)
    return out.view(orig_shape)


def rms_norm_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd_buf: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Triton-accelerated RMSNorm that writes into pre-allocated buffers.

    Used in the decode fast path to eliminate per-call tensor allocations.
    Does NOT save tensors for backward (inference only).

    Args:
        x: [*, N] input (will be reshaped to [M, N] internally)
        weight: [N] learnable scale
        out: [*, N] pre-allocated output buffer (same shape as x)
        rstd_buf: [M] pre-allocated buffer for rstd (M = product of leading dims)
        eps: epsilon for numerical stability

    Returns:
        out (same tensor passed in, now filled with normalized values)
    """
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out_2d = out.view(-1, x.shape[-1])
    M, N = x_2d.shape
    BLOCK_N = triton.next_power_of_2(N)

    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out_2d, rstd_buf,
        N, eps=eps, BLOCK_N=BLOCK_N,
    )

    return out.view(orig_shape)


def rms_norm_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm for testing."""
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)
