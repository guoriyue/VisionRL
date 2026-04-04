"""Quantization utilities: FP8 per-tensor/per-expert and INT4 per-group."""

import torch
from typing import Tuple


def quantize_per_tensor(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with per-tensor scale.

    Args:
        x: input tensor (any shape)
        dtype: target FP8 dtype

    Returns:
        x_fp8: quantized tensor
        scale: scalar scale factor (fp32)
    """
    fp8_max = torch.finfo(dtype).max
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / fp8_max).float()
    x_scaled = x.float() / scale
    x_fp8 = x_scaled.to(dtype)
    return x_fp8, scale


def dequantize_per_tensor(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to higher precision.

    Args:
        x_fp8: FP8 tensor
        scale: per-tensor scale factor
        output_dtype: desired output dtype

    Returns:
        x: dequantized tensor
    """
    return x_fp8.to(output_dtype) * scale.to(output_dtype)


def quantize_per_expert(
    weights: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert weight matrices with per-expert scales.

    Args:
        weights: [num_experts, K, N] weight tensor

    Returns:
        weights_fp8: [num_experts, K, N] quantized weights
        scales: [num_experts] per-expert scale factors (fp32)
    """
    num_experts = weights.shape[0]
    fp8_max = torch.finfo(dtype).max

    # Compute per-expert absmax
    flat = weights.view(num_experts, -1)
    amax = flat.abs().max(dim=1).values.clamp(min=1e-12)  # [num_experts]
    scales = (amax / fp8_max).float()  # [num_experts]

    # Scale and quantize
    weights_scaled = weights.float() / scales[:, None, None]
    weights_fp8 = weights_scaled.to(dtype)

    return weights_fp8, scales


# ─── INT4 Per-Group Quantization ───

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a tensor of INT4 values (range 0-15) into uint8 (2 values per byte).

    Packing: packed[..., n] = tensor[..., 2*n] | (tensor[..., 2*n+1] << 4)

    Args:
        tensor: integer tensor with values in [0, 15], last dim must be even.

    Returns:
        packed: uint8 tensor with last dim halved.
    """
    assert tensor.shape[-1] % 2 == 0, "Last dim must be even for INT4 packing"
    low = tensor[..., 0::2].to(torch.uint8)
    high = tensor[..., 1::2].to(torch.uint8)
    return low | (high << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor to INT4 values (0-15), doubling the last dim.

    Args:
        packed: uint8 tensor.

    Returns:
        tensor: integer tensor with values in [0, 15], last dim doubled.
    """
    low = (packed & 0xF).to(torch.int32)
    high = ((packed >> 4) & 0xF).to(torch.int32)
    # Interleave: [low0, high0, low1, high1, ...]
    shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    result = torch.empty(shape, dtype=torch.int32, device=packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result


def quantize_per_group_int4(
    weights: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights to INT4 with per-group scales and zeros (asymmetric).

    For each group of `group_size` rows along K, computes:
        scale = (max - min) / 15
        zero = round(-min / scale)  (stored as fp16, applied during dequant)
        q = round(w / scale + zero), clamped to [0, 15]

    Args:
        weights: [K, N] or [E, K, N] weight tensor (fp16/fp32).
        group_size: number of K elements per quantization group.

    Returns:
        packed: uint8 tensor with N//2 in last dim.
        scales: fp16 tensor [K//group_size, N] or [E, K//group_size, N].
        zeros: fp16 tensor, same shape as scales.
    """
    original_shape = weights.shape
    is_3d = weights.dim() == 3

    if is_3d:
        E, K, N = weights.shape
        weights_flat = weights.reshape(E * K, N)
    else:
        K, N = weights.shape
        E = 1
        weights_flat = weights

    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    num_groups = K // group_size

    w = weights_flat.float()

    if is_3d:
        # Reshape to [E, num_groups, group_size, N]
        w = w.reshape(E, num_groups, group_size, N)
    else:
        w = w.reshape(num_groups, group_size, N)

    # Per-group min/max along the group_size dimension
    w_min = w.min(dim=-2).values  # [E?, num_groups, N]
    w_max = w.max(dim=-2).values

    # Compute scale and zero
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp(min=1e-10)  # avoid division by zero
    zero = torch.round(-w_min / scale).clamp(0, 15)

    # Quantize
    if is_3d:
        q = torch.round(w / scale[:, :, None, :] + zero[:, :, None, :])
    else:
        q = torch.round(w / scale[:, None, :] + zero[:, None, :])
    q = q.clamp(0, 15).to(torch.int32)

    # Reshape back
    if is_3d:
        q = q.reshape(E, K, N)
    else:
        q = q.reshape(K, N)

    # Pack into uint8
    packed = pack_int4(q)

    scales = scale.to(torch.float16)
    zeros = zero.to(torch.float16)

    return packed, scales, zeros


def dequantize_per_group_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize INT4 packed weights back to fp16.

    Args:
        packed: uint8 [K, N//2] or [E, K, N//2].
        scales: fp16 [K//group_size, N] or [E, K//group_size, N].
        zeros: fp16, same shape as scales.
        group_size: quantization group size.

    Returns:
        weights: fp16 [K, N] or [E, K, N].
    """
    q = unpack_int4(packed)  # [K, N] or [E, K, N] int32
    is_3d = q.dim() == 3

    if is_3d:
        E, K, N = q.shape
        q_grouped = q.float().reshape(E, K // group_size, group_size, N)
        w = (q_grouped - zeros[:, :, None, :].float()) * scales[:, :, None, :].float()
        w = w.reshape(E, K, N)
    else:
        K, N = q.shape
        q_grouped = q.float().reshape(K // group_size, group_size, N)
        w = (q_grouped - zeros[:, None, :].float()) * scales[:, None, :].float()
        w = w.reshape(K, N)

    return w.to(torch.float16)


def quantize_per_expert_int4(
    weights: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize expert weight matrices to INT4 with per-group scales/zeros.

    Convenience wrapper around quantize_per_group_int4 for [E, K, N] tensors.

    Args:
        weights: [num_experts, K, N] weight tensor.
        group_size: quantization group size.

    Returns:
        packed: [num_experts, K, N//2] uint8.
        scales: [num_experts, K//group_size, N] fp16.
        zeros: [num_experts, K//group_size, N] fp16.
    """
    return quantize_per_group_int4(weights, group_size)
