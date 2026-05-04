"""Collector configuration schemas keyed by explicit family names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SD3_5CollectorConfig:
    """Configuration for SD3.5 rollout collection."""

    num_steps: int = 10
    guidance_scale: float = 4.5
    height: int = 512
    width: int = 512
    max_sequence_length: int = 256
    noise_level: float = 0.7
    cfg: bool = True
    sample_batch_size: int = 8
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    max_batch_requests: int = 1

    @property
    def return_kl(self) -> bool:
        return self.kl_reward > 0


@dataclass(slots=True)
class Wan_2_1CollectorConfig:
    """Configuration for Wan 2.1 rollout collection."""

    num_steps: int = 20
    guidance_scale: float = 4.5
    height: int = 240
    width: int = 416
    num_frames: int = 33
    max_sequence_length: int = 512
    noise_level: float = 1.0
    cfg: bool = True
    sample_batch_size: int = 1
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    max_batch_requests: int = 1

    @property
    def return_kl(self) -> bool:
        return self.kl_reward > 0


@dataclass(slots=True)
class CosmosPredict2CollectorConfig:
    """Configuration for Cosmos Predict2 rollout collection."""

    num_steps: int = 35
    guidance_scale: float = 7.0
    height: int = 704
    width: int = 1280
    num_frames: int = 93
    max_sequence_length: int = 512
    fps: int = 16
    noise_level: float = 1.0
    cfg: bool = True
    sample_batch_size: int = 8
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    max_batch_requests: int = 1

    @property
    def return_kl(self) -> bool:
        return self.kl_reward > 0


@dataclass(slots=True)
class JanusProCollectorConfig:
    """Configuration for Janus-Pro rollout collection."""

    n_samples_per_prompt: int = 8
    cfg_weight: float = 5.0
    temperature: float = 1.0
    image_token_num: int = 576
    image_size: int = 384
    rescale_to_unit: bool = True
    max_text_length: int = 256
    max_batch_requests: int = 1


@dataclass(slots=True)
class NextStep1CollectorConfig:
    """Configuration for NextStep-1 rollout collection."""

    n_samples_per_prompt: int = 4
    cfg_scale: float = 4.5
    num_flow_steps: int = 20
    noise_level: float = 1.0
    image_token_num: int = 1024
    image_size: int = 256
    rescale_to_unit: bool = True
    max_text_length: int = 256
    max_batch_requests: int = 1


__all__ = [
    "CosmosPredict2CollectorConfig",
    "JanusProCollectorConfig",
    "NextStep1CollectorConfig",
    "SD3_5CollectorConfig",
    "Wan_2_1CollectorConfig",
]
