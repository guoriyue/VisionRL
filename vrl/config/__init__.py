"""Config loading utilities (verl-style YAML overlay via OmegaConf)."""

from vrl.config.cli import parse_and_load
from vrl.config.loader import build_configs, load_config

__all__ = ["build_configs", "load_config", "parse_and_load"]
