"""Unified YAML-driven training entry point.

The experiment name and implementation entrypoint belong in
``configs/experiment/*.yaml``. This module is only the CLI/import layer: it
loads one YAML config, imports ``trainer.entrypoint``, then runs it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class TrainTarget:
    """Resolved implementation for one merged training config."""

    import_path: str


def resolve_train_target(cfg: DictConfig) -> TrainTarget:
    """Resolve a merged YAML config to its declared training callable."""

    try:
        import_path = cfg.trainer.entrypoint
    except Exception as exc:
        raise ValueError("config missing required field: trainer.entrypoint") from exc
    if not isinstance(import_path, str) or not import_path.strip():
        raise ValueError("trainer.entrypoint must be a non-empty import path")
    return TrainTarget(import_path.strip())


def _import_callable(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(
            "trainer.entrypoint must use 'module:function' import path syntax",
        )
    module_name, attr_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def run_config(cfg: DictConfig) -> Any:
    """Run the family trainer selected by ``cfg``."""

    target = resolve_train_target(cfg)
    trainer = _import_callable(target.import_path)
    return trainer(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YAML-driven VRL training job.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config name under configs/ or an absolute path.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. trainer.seed=42 actor.optim.lr=2e-4",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from vrl.config.loader import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    result = run_config(cfg)
    if inspect.isawaitable(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
