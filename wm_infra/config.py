"""Configuration for world model inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class SchedulerPolicy(str, Enum):
    FCFS = "fcfs"  # first-come first-serve
    SJF = "sjf"  # shortest-job-first (fewest remaining steps)
    DEADLINE = "deadline"  # earliest-deadline-first


@dataclass
class TokenizerConfig:
    """Video tokenizer configuration (COSMOS-style)."""

    spatial_downsample: int = 8
    temporal_downsample: int = 4
    latent_channels: int = 16
    codebook_size: int = 64  # FSQ levels per dimension
    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    causal_temporal: bool = True
    input_channels: int = 3  # RGB


@dataclass
class DynamicsConfig:
    """Latent dynamics model configuration."""

    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    action_dim: int = 64
    latent_token_dim: int = 16
    max_rollout_steps: int = 128
    dropout: float = 0.0
    use_triton_attention: bool = True


@dataclass
class StateCacheConfig:
    """Latent state cache configuration."""

    max_batch_size: int = 64
    max_rollout_steps: int = 128
    latent_dim: int = 16
    num_latent_tokens: int = 256  # tokens per frame after spatial tokenization
    pool_size_gb: float = 4.0  # GPU memory pool for state cache
    eviction_policy: str = "lru"


@dataclass
class SchedulerConfig:
    """Rollout scheduler configuration."""

    max_batch_size: int = 32
    max_waiting_time_ms: float = 50.0
    policy: SchedulerPolicy = SchedulerPolicy.SJF
    max_concurrent_rollouts: int = 64


@dataclass
class ServerConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8400
    max_concurrent_requests: int = 128
    stream_chunk_interval_ms: float = 33.0  # ~30fps streaming


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    device: DeviceType = DeviceType.CUDA
    dtype: str = "float16"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    state_cache: StateCacheConfig = field(default_factory=StateCacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    model_path: Optional[str] = None
    seed: int = 42
