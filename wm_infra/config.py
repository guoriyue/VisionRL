"""Configuration for world model inference engine.

Supports layered config loading: defaults → YAML file → env vars → CLI args.
Use ``load_config()`` to merge all layers into a single ``EngineConfig``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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


# ─── Config loading ───


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_yaml(path: str | Path) -> dict:
    """Load YAML config file. Returns empty dict if PyYAML not installed."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _env_overrides() -> dict:
    """Read WM_* environment variables into a nested dict.

    Mapping (all prefixed with WM_):
        WM_DEVICE=cpu            → {"device": "cpu"}
        WM_DTYPE=bfloat16        → {"dtype": "bfloat16"}
        WM_MODEL_PATH=/path      ��� {"model_path": "/path"}
        WM_PORT=9000             → {"server": {"port": 9000}}
        WM_HOST=127.0.0.1        → {"server": {"host": "127.0.0.1"}}
        WM_MAX_BATCH_SIZE=16     → {"scheduler": {"max_batch_size": 16}}
        WM_SEED=123              → {"seed": 123}
    """
    overrides: dict[str, Any] = {}
    env_map: dict[str, tuple[list[str], type]] = {
        "WM_DEVICE": (["device"], str),
        "WM_DTYPE": (["dtype"], str),
        "WM_MODEL_PATH": (["model_path"], str),
        "WM_SEED": (["seed"], int),
        "WM_PORT": (["server", "port"], int),
        "WM_HOST": (["server", "host"], str),
        "WM_MAX_BATCH_SIZE": (["scheduler", "max_batch_size"], int),
        "WM_MAX_CONCURRENT_ROLLOUTS": (["scheduler", "max_concurrent_rollouts"], int),
    }

    for env_key, (path, typ) in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        converted = typ(val)
        d = overrides
        for part in path[:-1]:
            d = d.setdefault(part, {})
        d[path[-1]] = converted

    return overrides


def _dict_to_config(d: dict) -> EngineConfig:
    """Convert a flat/nested dict into an EngineConfig dataclass."""
    # Handle device enum
    if "device" in d and isinstance(d["device"], str):
        d["device"] = DeviceType(d["device"])

    # Build sub-configs
    tok_d = d.pop("tokenizer", {})
    dyn_d = d.pop("dynamics", {})
    sc_d = d.pop("state_cache", {})
    sched_d = d.pop("scheduler", {})
    serv_d = d.pop("server", {})

    # Handle scheduler policy enum
    if "policy" in sched_d and isinstance(sched_d["policy"], str):
        sched_d["policy"] = SchedulerPolicy(sched_d["policy"])

    return EngineConfig(
        tokenizer=TokenizerConfig(**tok_d) if tok_d else TokenizerConfig(),
        dynamics=DynamicsConfig(**dyn_d) if dyn_d else DynamicsConfig(),
        state_cache=StateCacheConfig(**sc_d) if sc_d else StateCacheConfig(),
        scheduler=SchedulerConfig(**sched_d) if sched_d else SchedulerConfig(),
        server=ServerConfig(**serv_d) if serv_d else ServerConfig(),
        **d,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for wm-serve."""
    parser = argparse.ArgumentParser(
        prog="wm-serve",
        description="World Model Inference Server",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model weights")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32", "bfloat16"], default=None)
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8400)")
    parser.add_argument("--host", type=str, default=None, help="Server host (default: 0.0.0.0)")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Max batch size for scheduling")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def load_config(
    cli_args: Optional[list[str]] = None,
    config_path: Optional[str] = None,
) -> EngineConfig:
    """Load config by merging: defaults → YAML → env vars → CLI args.

    Args:
        cli_args: CLI argument list (None = use sys.argv)
        config_path: Explicit YAML path (overrides --config CLI arg)
    """
    # 1. Start with defaults
    merged: dict[str, Any] = asdict(EngineConfig())
    # Convert enum values back to strings for merging
    merged["device"] = merged["device"].value if hasattr(merged["device"], "value") else merged["device"]
    merged["scheduler"]["policy"] = (
        merged["scheduler"]["policy"].value
        if hasattr(merged["scheduler"]["policy"], "value")
        else merged["scheduler"]["policy"]
    )

    # 2. Parse CLI (need config path first)
    parser = build_parser()
    args = parser.parse_args(cli_args if cli_args is not None else [])

    # 3. YAML overlay
    yaml_path = config_path or args.config
    if yaml_path and Path(yaml_path).exists():
        yaml_d = _load_yaml(yaml_path)
        merged = _deep_merge(merged, yaml_d)

    # 4. Env var overlay
    env_d = _env_overrides()
    if env_d:
        merged = _deep_merge(merged, env_d)

    # 5. CLI arg overlay (only non-None values)
    cli_overrides: dict[str, Any] = {}
    if args.device is not None:
        cli_overrides["device"] = args.device
    if args.dtype is not None:
        cli_overrides["dtype"] = args.dtype
    if args.model_path is not None:
        cli_overrides["model_path"] = args.model_path
    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.port is not None:
        cli_overrides.setdefault("server", {})["port"] = args.port
    if args.host is not None:
        cli_overrides.setdefault("server", {})["host"] = args.host
    if args.max_batch_size is not None:
        cli_overrides.setdefault("scheduler", {})["max_batch_size"] = args.max_batch_size

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return _dict_to_config(merged)
