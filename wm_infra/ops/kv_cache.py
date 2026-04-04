"""KV Cache: contiguous pre-allocated buffer for autoregressive decoding.

Usage:
    cache = KVCache(batch_size=2, num_kv_heads=8, max_seq_len=4096,
                    head_dim=128, dtype=torch.float16, device="cuda")

    # Prefill: write all positions at once
    cache.update(k, v, positions)  # k,v: [B, Hkv, S, D], positions: [S]

    # Decode: write one token at a time
    cache.update(k_new, v_new, torch.tensor([seq_len]))  # k_new: [B, Hkv, 1, D]

    # Get cached K/V up to current length
    k_cached, v_cached = cache.get()  # [B, Hkv, cur_len, D]
"""

import torch
import triton

from wm_infra.kernels.kv_cache_kernel import kv_cache_append_kernel


class KVCache:
    """Contiguous KV cache with Triton-accelerated append."""

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = "cuda",
    ):
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # Pre-allocate: [B, Hkv, max_seq_len, D]
        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )

        self.seq_len = 0  # current sequence length

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V into cache at given positions, return full cached K/V.

        Args:
            k: [B, Hkv, S_new, D] new keys
            v: [B, Hkv, S_new, D] new values
            positions: [S_new] position indices (int64)

        Returns:
            (k_cached, v_cached): [B, Hkv, seq_len, D] sliced views
        """
        B, H, S_new, D = k.shape
        assert k.is_contiguous() and v.is_contiguous()

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * H * S_new,)

        # Append K
        kv_cache_append_kernel[grid](
            k, self.k_cache, positions,
            H, S_new, self.max_seq_len, D,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            self.k_cache.stride(0), self.k_cache.stride(1),
            self.k_cache.stride(2), self.k_cache.stride(3),
            BLOCK_D=BLOCK_D,
        )

        # Append V
        kv_cache_append_kernel[grid](
            v, self.v_cache, positions,
            H, S_new, self.max_seq_len, D,
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            self.v_cache.stride(0), self.v_cache.stride(1),
            self.v_cache.stride(2), self.v_cache.stride(3),
            BLOCK_D=BLOCK_D,
        )

        # Update seq_len from positions without GPU→CPU sync
        # positions is monotonically increasing, so last element is max
        S_new = positions.shape[0]
        # For prefill: positions = [0,1,...,S-1], new_max = S
        # For decode:  positions = [seq_len], new_max = seq_len+1
        # Use the counter-based approach: caller provides positions that are
        # consecutive, so we can track seq_len via S_new increment.
        self.seq_len = self.seq_len + S_new if self.seq_len > 0 else S_new

        return self.get()

    def update_decode(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast path for S=1 autoregressive decode: skip positions entirely.

        Writes k/v at self.seq_len, increments seq_len by 1.
        No GPU→CPU sync, no positions tensor needed.

        Args:
            k: [B, Hkv, 1, D] new key
            v: [B, Hkv, 1, D] new value

        Returns:
            (k_cached, v_cached): [B, Hkv, seq_len+1, D]
        """
        pos = self.seq_len
        # Direct indexing: write at position pos
        self.k_cache[:, :, pos, :] = k[:, :, 0, :]
        self.v_cache[:, :, pos, :] = v[:, :, 0, :]
        self.seq_len = pos + 1
        return self.get()

    def update_decode_cudagraph(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CUDA graph-compatible decode update: writes at GPU tensor position.

        Unlike update_decode(), this returns the FULL pre-allocated cache buffers
        (no dynamic slicing) so tensor shapes/addresses are static across graph
        replays. The caller uses an attention mask to ignore padding positions.

        The seq_len counter is NOT incremented here -- the caller (TransformerModel)
        manages it externally and updates pos_index before graph replay.

        Uses index_copy_ on the sequence dimension for GPU-side positional write
        that is compatible with CUDA graph capture (no CPU-GPU sync).

        Args:
            k: [B, Hkv, 1, D] new key
            v: [B, Hkv, 1, D] new value
            pos_index: [1] int64 GPU tensor with the write position index

        Returns:
            (k_full, v_full): [B, Hkv, max_seq_len, D] full cache buffers
        """
        # index_copy_ on dim=2 (sequence dimension) is a single GPU kernel,
        # fully graph-compatible — no CPU-GPU sync needed
        self.k_cache.index_copy_(2, pos_index, k)
        self.v_cache.index_copy_(2, pos_index, v)
        return self.k_cache, self.v_cache

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached K/V up to current sequence length.

        Returns:
            (k, v): [B, Hkv, seq_len, D] sliced views (no copy)
        """
        return (
            self.k_cache[:, :, :self.seq_len, :],
            self.v_cache[:, :, :self.seq_len, :],
        )

    def reset(self):
        """Reset cache for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


class MLAKVCache:
    """Compressed KV cache for Multi-head Latent Attention (DeepSeek-V2/V3).

    Instead of storing full K/V per head, stores the low-rank latent
    representation (kv_latent) and the RoPE'd key component (k_rope).
    Full K/V are reconstructed during attention via kv_b_proj.

    Memory: (kv_lora_rank + qk_rope_head_dim) per position
    vs (num_heads * head_dim * 2) for standard KV cache.

    Usage:
        cache = MLAKVCache(batch_size=2, max_seq_len=4096,
                           kv_lora_rank=512, qk_rope_head_dim=64,
                           dtype=torch.float16, device="cuda")
        cache.update(kv_latent, k_rope, positions)  # returns cached slices
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = "cuda",
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # [B, 1, max_seq_len, kv_lora_rank] — H=1 for kernel compatibility
        self.latent_cache = torch.zeros(
            batch_size, 1, max_seq_len, kv_lora_rank,
            dtype=dtype, device=device,
        )
        # [B, 1, max_seq_len, qk_rope_head_dim]
        self.k_rope_cache = torch.zeros(
            batch_size, 1, max_seq_len, qk_rope_head_dim,
            dtype=dtype, device=device,
        )

        self.seq_len = 0

    def update(
        self,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new latent/k_rope into cache, return cached slices.

        Args:
            kv_latent: [B, S_new, kv_lora_rank]
            k_rope: [B, S_new, qk_rope_head_dim]
            positions: [S_new] position indices

        Returns:
            (latent_cached, k_rope_cached): [B, seq_len, D] sliced views
        """
        B, S_new, _ = kv_latent.shape

        # Reshape to [B, 1, S_new, D] for the H=1 kernel
        latent_4d = kv_latent.unsqueeze(1).contiguous()
        k_rope_4d = k_rope.unsqueeze(1).contiguous()

        BLOCK_D_latent = triton.next_power_of_2(self.kv_lora_rank)
        BLOCK_D_rope = triton.next_power_of_2(self.qk_rope_head_dim)

        grid_latent = (B * 1 * S_new,)
        grid_rope = (B * 1 * S_new,)

        # Append latent
        kv_cache_append_kernel[grid_latent](
            latent_4d, self.latent_cache, positions,
            1, S_new, self.max_seq_len, self.kv_lora_rank,
            latent_4d.stride(0), latent_4d.stride(1), latent_4d.stride(2), latent_4d.stride(3),
            self.latent_cache.stride(0), self.latent_cache.stride(1),
            self.latent_cache.stride(2), self.latent_cache.stride(3),
            BLOCK_D=BLOCK_D_latent,
        )

        # Append k_rope
        kv_cache_append_kernel[grid_rope](
            k_rope_4d, self.k_rope_cache, positions,
            1, S_new, self.max_seq_len, self.qk_rope_head_dim,
            k_rope_4d.stride(0), k_rope_4d.stride(1), k_rope_4d.stride(2), k_rope_4d.stride(3),
            self.k_rope_cache.stride(0), self.k_rope_cache.stride(1),
            self.k_rope_cache.stride(2), self.k_rope_cache.stride(3),
            BLOCK_D=BLOCK_D_rope,
        )

        # Update seq_len without GPU→CPU sync
        S_new = positions.shape[0]
        self.seq_len = self.seq_len + S_new if self.seq_len > 0 else S_new

        return self.get()

    def update_decode(
        self,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast path for S=1 decode: no positions, no GPU→CPU sync.

        Args:
            kv_latent: [B, 1, kv_lora_rank]
            k_rope: [B, 1, qk_rope_head_dim]

        Returns:
            (latent_cached, k_rope_cached): [B, seq_len+1, D]
        """
        pos = self.seq_len
        self.latent_cache[:, 0, pos, :] = kv_latent[:, 0, :]
        self.k_rope_cache[:, 0, pos, :] = k_rope[:, 0, :]
        self.seq_len = pos + 1
        return self.get()

    def update_decode_cudagraph(
        self,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
        pos_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CUDA graph-compatible decode update for MLA cache.

        Writes at GPU tensor position, returns full cache buffers (no slicing).
        The caller manages seq_len and attention masking externally.

        Args:
            kv_latent: [B, 1, kv_lora_rank]
            k_rope: [B, 1, qk_rope_head_dim]
            pos_index: [1] int64 GPU tensor with the write position

        Returns:
            (latent_full, k_rope_full): [B, max_seq_len, D] full cache buffers
        """
        # Reshape to [B, 1, 1, D] for index_copy_ on dim=2
        latent_4d = kv_latent.unsqueeze(1)  # [B, 1, 1, kv_lora_rank]
        k_rope_4d = k_rope.unsqueeze(1)     # [B, 1, 1, qk_rope_head_dim]
        self.latent_cache.index_copy_(2, pos_index, latent_4d)
        self.k_rope_cache.index_copy_(2, pos_index, k_rope_4d)
        # Return full buffers squeezed to [B, max_seq, D]
        return self.latent_cache[:, 0, :, :], self.k_rope_cache[:, 0, :, :]

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached latent/k_rope up to current sequence length.

        Returns:
            (latent, k_rope): [B, seq_len, D] sliced (squeeze H=1 dim)
        """
        return (
            self.latent_cache[:, 0, :self.seq_len, :],
            self.k_rope_cache[:, 0, :self.seq_len, :],
        )

    def reset(self):
        """Reset cache for new sequence."""
        self.latent_cache.zero_()
        self.k_rope_cache.zero_()
        self.seq_len = 0
