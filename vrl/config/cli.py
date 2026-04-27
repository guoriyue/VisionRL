"""Shared CLI helper for YAML-driven training scripts."""

from __future__ import annotations

import argparse
import logging

from omegaconf import DictConfig


def parse_and_load(default_config: str, description: str) -> DictConfig:
    """Build the standard ``--config`` + dotlist-overrides argparse parser
    and return the merged ``DictConfig``.
    """
    from vrl.config.loader import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default=default_config,
        help="YAML config name (under configs/) or absolute path.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dotlist overrides, e.g. trainer.seed=42 actor.optim.lr=2e-4",
    )
    args = parser.parse_args()
    return load_config(args.config, overrides=args.overrides)
