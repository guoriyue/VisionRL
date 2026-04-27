"""verl/Hydra-style YAML loader with ``defaults:`` overlay.

Each YAML file may include a top-level ``defaults:`` list. Each entry is
a path (without ``.yaml``) relative to the ``configs/`` root. Defaults
are merged in list order with the file's own content layered on top
(unless ``_self_`` reorders).

Examples:
    # configs/experiment/wan_2_1_1_3b_ocr_grpo.yaml
    defaults:
      - /base/algorithm/grpo
      - /base/actor
      - /base/rollout/diffusion
      - /base/trainer
      - /model/wan/wan_2_1_1_3b
      - /model/wan/generation
    reward:
      ocr: 1.0

CLI override syntax matches OmegaConf dotlist:
    python -m vrl.scripts.x --config experiment/wan_ocr trainer.seed=42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"

_SELF_ = "_self_"


def _resolve_default(entry: str | dict, root: Path) -> Path:
    """Resolve a ``defaults:`` entry to a yaml path under ``root``.

    Accepts either a leading-slash absolute-style path (``/base/actor``)
    or a relative one (``base/actor``). Adds ``.yaml`` if missing.
    """
    if isinstance(entry, dict):
        # Hydra group syntax: {group: option}. We don't support groups yet —
        # so flatten {key: val} into "key/val".
        if len(entry) != 1:
            raise ValueError(f"defaults dict must have exactly one key: {entry}")
        k, v = next(iter(entry.items()))
        entry = f"{k}/{v}"
    s = entry.lstrip("/")
    if not s.endswith((".yaml", ".yml")):
        s = f"{s}.yaml"
    return root / s


def _load_one(path: Path, root: Path, _seen: set[Path] | None = None) -> DictConfig:
    """Load a single YAML and recursively merge its ``defaults:`` list."""
    path = path.resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise RuntimeError(f"Cyclic defaults: {path}")
    _seen = _seen | {path}

    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"{path}: top-level must be a mapping")

    defaults = raw.pop("defaults", None) if "defaults" in raw else None
    merged: DictConfig = OmegaConf.create({})

    if defaults is not None:
        if not isinstance(defaults, (list, ListConfig)):
            raise TypeError(f"{path}: 'defaults' must be a list")
        self_seen = False
        for entry in defaults:
            entry_val = entry if not hasattr(entry, "_content") else OmegaConf.to_container(entry)
            if entry_val == _SELF_:
                merged = OmegaConf.merge(merged, raw)
                self_seen = True
                continue
            sub_path = _resolve_default(entry_val, root)
            sub = _load_one(sub_path, root, _seen)
            merged = OmegaConf.merge(merged, sub)
        if not self_seen:
            merged = OmegaConf.merge(merged, raw)
    else:
        merged = raw

    assert isinstance(merged, DictConfig)
    return merged


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    """Load a YAML config with verl-style defaults overlay.

    Args:
        path: Either an absolute filesystem path, or a name relative to
            ``configs/`` (e.g. ``experiment/wan_2_1_1_3b_ocr_grpo``).
        overrides: OmegaConf dotlist overrides (e.g. ``["trainer.seed=42"]``).
        root: Override the configs root (defaults to repo ``configs/``).

    Returns:
        Merged ``DictConfig``. Resolves interpolations on the way out.
    """
    root = (root or CONFIGS_ROOT).resolve()

    p = Path(path)
    if not p.is_absolute() and not p.exists():
        rel = path if isinstance(path, str) else str(path)
        rel = rel.lstrip("/")
        if not rel.endswith((".yaml", ".yml")):
            rel = f"{rel}.yaml"
        p = root / rel

    cfg = _load_one(p, root)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        assert isinstance(cfg, DictConfig)

    OmegaConf.resolve(cfg)
    return cfg


def build_configs(cfg: DictConfig) -> dict[str, Any]:
    """Map the merged YAML into the typed dataclasses the runtime expects.

    Returns a dict with keys:
      ``trainer``      -> ``TrainerConfig``
      ``algorithm``    -> ``GRPOConfig``
      ``raw``          -> the original ``DictConfig`` for script-side reads
                          (model/generation/reward/data sections)
    """
    from vrl.algorithms.grpo import GRPOConfig
    from vrl.trainers.types import (
        DebugConfig,
        EMAConfig,
        OptimConfig,
        TrainerConfig,
    )

    actor = cfg.get("actor", {})
    optim_cfg = actor.get("optim", {})
    ema_cfg = actor.get("ema", {})
    rollout = cfg.get("rollout", {})
    trainer_section = cfg.get("trainer", {})
    debug_cfg = trainer_section.get("debug", {})
    algo = cfg.get("algorithm", {})

    trainer_cfg = TrainerConfig(
        optim=OptimConfig(**OmegaConf.to_container(optim_cfg, resolve=True)),
        ema=EMAConfig(**OmegaConf.to_container(ema_cfg, resolve=True)),
        debug=DebugConfig(**OmegaConf.to_container(debug_cfg, resolve=True)),
        max_norm=actor.get("max_norm", 1.0),
        ppo_epochs=actor.get("ppo_epochs", 1),
        bf16=actor.get("bf16", True),
        gradient_checkpointing=actor.get("gradient_checkpointing", True),
        n=rollout.get("n", 4),
        rollout_batch_size=rollout.get("rollout_batch_size", 4),
        timestep_fraction=rollout.get("timestep_fraction", 1.0),
        total_epochs=trainer_section.get("total_epochs", 10000),
        save_freq=trainer_section.get("save_freq", 50),
        log_freq=trainer_section.get("log_freq", 1),
        output_dir=trainer_section.get("output_dir", "outputs/"),
        seed=trainer_section.get("seed", 0),
        profile=trainer_section.get("profile", False),
    )

    algo_dict = OmegaConf.to_container(algo, resolve=True) if algo else {}
    # Drop fields that are not GRPOConfig-recognized (e.g. ``adv_estimator``,
    # ``per_prompt_stat_tracking``) so we don't error on **kwargs unpack.
    algo_kwargs = {
        k: v for k, v in algo_dict.items()
        if k in {"eps_clip", "init_kl_coef", "eps", "adv_clip_max", "global_std"}
    }
    algorithm_cfg = GRPOConfig(**algo_kwargs)

    return {
        "trainer": trainer_cfg,
        "algorithm": algorithm_cfg,
        "raw": cfg,
    }
