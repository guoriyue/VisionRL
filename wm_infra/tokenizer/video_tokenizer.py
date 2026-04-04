"""COSMOS-style video tokenizer: frames -> discrete latent tokens.

Architecture:
  1. Spatial encoder: 2D conv downsampling (8x spatial)
  2. Temporal encoder: causal 1D conv (4x temporal) — ensures frame-by-frame decoding
  3. FSQ quantizer: Finite Scalar Quantization — no codebook collapse, no EMA

The tokenizer converts video frames [B, T, C, H, W] into discrete latent tokens
[B, T', N_spatial, D_latent] where T' = T // temporal_downsample and
N_spatial = (H * W) // spatial_downsample^2.

Reference: COSMOS tokenizer (NVIDIA, 2025) — 8x total compression with causal temporal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_infra.config import TokenizerConfig


class CausalConv1d(nn.Module):
    """Causal 1D convolution for temporal processing.

    Left-pads the input so that output[t] depends only on input[<=t].
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SpatialEncoder(nn.Module):
    """Spatial downsampler: 2D strided convolutions on each frame."""

    def __init__(self, in_channels: int, latent_channels: int, downsample: int):
        super().__init__()
        num_stages = int(math.log2(downsample))
        layers = []
        ch = in_channels
        for i in range(num_stages):
            out_ch = latent_channels if i == num_stages - 1 else min(ch * 2, latent_channels)
            layers.extend([
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ])
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W] -> [B*T, latent_ch, H', W']
        return self.net(x)


class SpatialDecoder(nn.Module):
    """Spatial upsampler: transpose convolutions to reconstruct frames."""

    def __init__(self, latent_channels: int, out_channels: int, upsample: int):
        super().__init__()
        num_stages = int(math.log2(upsample))
        layers = []
        ch = latent_channels
        for i in range(num_stages):
            out_ch = out_channels if i == num_stages - 1 else max(ch // 2, out_channels)
            layers.extend([
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch) if i < num_stages - 1 else nn.Identity(),
                nn.SiLU() if i < num_stages - 1 else nn.Sigmoid(),
            ])
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalEncoder(nn.Module):
    """Causal temporal downsampler operating on spatial latent tokens."""

    def __init__(self, channels: int, downsample: int):
        super().__init__()
        num_stages = int(math.log2(downsample))
        layers = []
        for _ in range(num_stages):
            layers.append(CausalConv1d(channels, channels, kernel_size=3, stride=2))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N_spatial, C, T] -> [B*N_spatial, C, T']
        return self.net(x)


class TemporalDecoder(nn.Module):
    """Temporal upsampler with causal structure."""

    def __init__(self, channels: int, upsample: int):
        super().__init__()
        num_stages = int(math.log2(upsample))
        layers = []
        for _ in range(num_stages):
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(CausalConv1d(channels, channels, kernel_size=3, stride=1))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization — no codebook, no collapse.

    Each dimension is independently quantized to a finite set of levels.
    Total codebook size = product(levels).

    Reference: Mentzer et al., "Finite Scalar Quantization" (2023)
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32))
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and return (quantized, indices).

        Args:
            z: [..., D] continuous latent (D must equal len(levels))

        Returns:
            z_q: [..., D] quantized (straight-through gradient)
            indices: [...] integer codes
        """
        # Bound to (-1, 1) via tanh, then map to level grid
        z_bounded = torch.tanh(z)

        # Map each dim to [0, L-1] then snap to nearest integer
        half_levels = (self.levels.float() - 1) / 2.0
        z_scaled = z_bounded * half_levels
        z_quantized = torch.round(z_scaled)

        # Straight-through estimator
        z_q = z + (z_quantized / half_levels - z_bounded).detach()

        # Compute flat indices
        indices = torch.zeros(z.shape[:-1], dtype=torch.int64, device=z.device)
        multiplier = 1
        for i in reversed(range(self.dim)):
            level_idx = (z_quantized[..., i] + half_levels[i]).long()
            indices = indices + level_idx * multiplier
            multiplier *= self.levels[i].item()

        return z_q, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert integer codes back to quantized vectors."""
        z_q = torch.zeros(*indices.shape, self.dim, device=indices.device, dtype=torch.float32)
        remainder = indices
        half_levels = (self.levels.float() - 1) / 2.0
        for i in reversed(range(self.dim)):
            L = self.levels[i].item()
            level_idx = remainder % L
            remainder = remainder // L
            z_q[..., i] = (level_idx.float() - half_levels[i]) / half_levels[i]
        return z_q


class VideoTokenizer(nn.Module):
    """Full video tokenizer: encode frames to discrete tokens, decode back.

    Encode: [B, T, C, H, W] -> [B, T', N, D] latent tokens + [B, T', N] indices
    Decode: [B, T', N, D] -> [B, T, C, H, W] reconstructed frames
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        self.spatial_encoder = SpatialEncoder(
            config.input_channels, config.latent_channels, config.spatial_downsample
        )
        self.spatial_decoder = SpatialDecoder(
            config.latent_channels, config.input_channels, config.spatial_downsample
        )

        if config.temporal_downsample > 1:
            self.temporal_encoder = TemporalEncoder(config.latent_channels, config.temporal_downsample)
            self.temporal_decoder = TemporalDecoder(config.latent_channels, config.temporal_downsample)
        else:
            self.temporal_encoder = nn.Identity()
            self.temporal_decoder = nn.Identity()

        # Project to FSQ dimension if needed
        fsq_dim = len(config.fsq_levels)
        self.to_fsq = nn.Linear(config.latent_channels, fsq_dim) if config.latent_channels != fsq_dim else nn.Identity()
        self.from_fsq = nn.Linear(fsq_dim, config.latent_channels) if config.latent_channels != fsq_dim else nn.Identity()

        self.quantizer = FSQQuantizer(config.fsq_levels)

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video frames to discrete latent tokens.

        Args:
            video: [B, T, C, H, W] input video (pixel values in [0, 1])

        Returns:
            z_q: [B, T', N, D] quantized latent tokens
            indices: [B, T', N] discrete token indices
        """
        B, T, C, H, W = video.shape

        # Spatial encoding: process each frame independently
        frames = video.reshape(B * T, C, H, W)
        spatial = self.spatial_encoder(frames)  # [B*T, latent_ch, H', W']
        _, ch, h, w = spatial.shape
        N = h * w  # number of spatial tokens per frame

        # Reshape for temporal: [B*N, ch, T]
        spatial = spatial.reshape(B, T, ch, N).permute(0, 3, 2, 1).reshape(B * N, ch, T)

        # Temporal encoding
        temporal = self.temporal_encoder(spatial)  # [B*N, ch, T']
        T_prime = temporal.shape[2]

        # Reshape to [B, T', N, ch]
        latent = temporal.reshape(B, N, ch, T_prime).permute(0, 3, 1, 2)

        # Quantize
        z_fsq = self.to_fsq(latent)
        z_q, indices = self.quantizer(z_fsq)

        return z_q, indices

    def decode(self, z_q: torch.Tensor, original_shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """Decode quantized latent tokens back to video frames.

        Args:
            z_q: [B, T', N, D] quantized latent tokens
            original_shape: Optional (B, T, C, H, W) for exact output sizing

        Returns:
            video: [B, T, C, H, W] reconstructed frames
        """
        B, T_prime, N, D = z_q.shape

        # Unquantize
        latent = self.from_fsq(z_q)  # [B, T', N, latent_ch]
        ch = latent.shape[-1]

        # Temporal decoding: reshape to [B*N, ch, T']
        temporal_in = latent.permute(0, 2, 3, 1).reshape(B * N, ch, T_prime)
        temporal_out = self.temporal_decoder(temporal_in)  # [B*N, ch, T]
        T = temporal_out.shape[2]

        # Reshape back to spatial: [B*T, ch, h, w]
        h = w = int(math.sqrt(N))
        spatial_in = temporal_out.reshape(B, N, ch, T).permute(0, 3, 2, 1).reshape(B * T, ch, h, w)

        # Spatial decoding
        frames = self.spatial_decoder(spatial_in)  # [B*T, C, H_out, W_out]
        _, C, H_out, W_out = frames.shape

        return frames.reshape(B, T, C, H_out, W_out)

    def encode_frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single frame (no temporal processing).

        Args:
            frame: [B, C, H, W] single frame

        Returns:
            z_q: [B, N, D] quantized spatial tokens
            indices: [B, N] discrete indices
        """
        return self.encode(frame.unsqueeze(1))  # Add T=1 dim

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from discrete token indices."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decode(z_q)
