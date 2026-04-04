"""Rotary Position Embedding op with autograd support."""

import torch
import triton

from wm_infra.kernels.rope_kernel import rope_fwd_kernel, rope_bwd_kernel


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns:
        cos: [max_seq_len, head_dim // 2]
        sin: [max_seq_len, head_dim // 2]
    """
    half_d = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_d, device=device).float() / half_d))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [max_seq_len, half_d]
    return angles.cos(), angles.sin()


class ApplyRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, positions):
        """
        Args:
            x: [B, S, H, D] — queries or keys
            cos: [max_seq_len, D//2]
            sin: [max_seq_len, D//2]
            positions: [S] — position indices (int64)
        """
        assert x.is_contiguous()
        B, S, H, D = x.shape
        HALF_D = D // 2
        BLOCK_D = triton.next_power_of_2(HALF_D)

        y = torch.empty_like(x)
        grid = (B * S * H,)

        rope_fwd_kernel[grid](
            x, cos, sin, positions, y,
            S, H, D,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            cos.stride(0), cos.stride(1),
            HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(cos, sin, positions)
        ctx.shape = (B, S, H, D)
        return y

    @staticmethod
    def backward(ctx, dy):
        cos, sin, positions = ctx.saved_tensors
        B, S, H, D = ctx.shape
        HALF_D = D // 2
        BLOCK_D = triton.next_power_of_2(HALF_D)

        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        grid = (B * S * H,)

        rope_bwd_kernel[grid](
            dy, cos, sin, positions, dx,
            S, H, D,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            cos.stride(0), cos.stride(1),
            HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )

        return dx, None, None, None


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: [B, S, H, D] input tensor
        cos: [max_seq_len, D//2] precomputed cos table
        sin: [max_seq_len, D//2] precomputed sin table
        positions: [S] position indices

    Returns:
        Rotated tensor, same shape as x
    """
    return ApplyRoPEFunction.apply(x, cos, sin, positions)


def apply_rope_naive(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference RoPE for testing."""
    B, S, H, D = x.shape
    half_d = D // 2

    # Gather cos/sin for these positions: [S, D//2] -> [1, S, 1, D//2]
    cos_pos = cos[positions].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]
    sin_pos = sin[positions].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]

    x0 = x[..., :half_d]
    x1 = x[..., half_d:]

    y0 = x0 * cos_pos - x1 * sin_pos
    y1 = x0 * sin_pos + x1 * cos_pos

    return torch.cat([y0, y1], dim=-1)
