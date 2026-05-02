"""Unified YAML-driven training entry point.

The experiment name belongs in ``configs/experiment/*.yaml``. This module is
the only CLI layer: it loads one YAML config, resolves the model family and
algorithm kind, then dispatches to the family implementation module.
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


def _positive_reward(cfg: DictConfig, name: str) -> bool:
    reward = cfg.get("reward")
    if reward is None or "components" not in reward:
        return False
    return float(reward.components.get(name, 0.0)) > 0.0


def resolve_train_target(cfg: DictConfig) -> TrainTarget:
    """Resolve a merged YAML config to the concrete family trainer."""

    family = str(cfg.model.family)
    kind = str(cfg.algorithm.kind)

    if family == "cosmos" and kind == "grpo":
        return TrainTarget("vrl.scripts.cosmos.train:train_cosmos_predict2_grpo")
    if family == "sd3_5" and kind == "grpo":
        return TrainTarget("vrl.scripts.sd3_5.train:train_sd3_5_grpo")
    if family == "wan" and kind == "grpo":
        return TrainTarget("vrl.scripts.wan_2_1.train:train_wan_2_1_grpo")
    if family == "wan" and kind == "diffusion_dpo":
        return TrainTarget("vrl.scripts.wan_2_1.train_dpo:train_wan_2_1_dpo")
    if family == "nextstep_1" and kind == "token_grpo":
        return TrainTarget("vrl.scripts.nextstep_1.train:train_nextstep_1_ocr_grpo")
    if family == "janus_pro" and kind == "token_grpo":
        if _positive_reward(cfg, "ocr"):
            return TrainTarget("vrl.scripts.janus_pro.train:train_janus_pro_ocr_grpo")
        return TrainTarget("vrl.scripts.janus_pro.train:train_janus_pro_grpo")

    raise ValueError(
        f"unsupported training config: model.family={family!r}, "
        f"algorithm.kind={kind!r}",
    )


def _import_callable(import_path: str) -> Any:
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
