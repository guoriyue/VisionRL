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


_GRPO_FIELDS = {"eps_clip", "init_kl_coef", "eps", "adv_clip_max", "global_std"}
_TOKEN_GRPO_EXTRA_FIELDS = {"mask_key", "kl_estimator"}
_DPO_FIELDS = {"beta", "sft_weight"}

_KIND_LEGACY_ALIASES = {
    "grpo": "grpo",
    "token_grpo": "token_grpo",
    "diffusion_dpo": "diffusion_dpo",
    "dpo": "diffusion_dpo",  # legacy adv_estimator
}


def _resolve_algorithm_kind(algo: DictConfig) -> str:
    """Resolve ``algorithm.kind`` with legacy ``adv_estimator`` alias.

    Conflict detection: if both fields are set and disagree (after legacy
    alias normalisation) — fail fast. Per SPRINT_config_yaml_unification_patch
    Phase 1.
    """
    kind = algo.get("kind", None)
    legacy = algo.get("adv_estimator", None)
    if kind is None and legacy is None:
        raise ValueError(
            "algorithm config missing both `kind` and legacy `adv_estimator`",
        )
    if kind is None:
        normalised = _KIND_LEGACY_ALIASES.get(str(legacy))
        if normalised is None:
            raise ValueError(
                f"unknown legacy algorithm.adv_estimator={legacy!r}; "
                f"expected one of {sorted(_KIND_LEGACY_ALIASES)}",
            )
        return normalised
    kind = str(kind)
    if kind not in {"grpo", "token_grpo", "diffusion_dpo"}:
        raise ValueError(
            f"unknown algorithm.kind={kind!r}; "
            f"expected grpo / token_grpo / diffusion_dpo",
        )
    if legacy is not None:
        normalised = _KIND_LEGACY_ALIASES.get(str(legacy))
        if normalised != kind:
            raise ValueError(
                f"algorithm.kind={kind!r} conflicts with "
                f"legacy adv_estimator={legacy!r}",
            )
    return kind


def build_trainer_config(cfg: DictConfig):
    """Slice merged YAML into ``TrainerConfig``."""
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

    return TrainerConfig(
        optim=OptimConfig(**OmegaConf.to_container(optim_cfg, resolve=True)),
        ema=EMAConfig(**OmegaConf.to_container(ema_cfg, resolve=True)),
        debug=DebugConfig(**OmegaConf.to_container(debug_cfg, resolve=True)),
        max_norm=actor.get("max_norm", 1.0),
        ppo_epochs=actor.get("ppo_epochs", 1),
        bf16=actor.get("bf16", True),
        gradient_checkpointing=actor.get("gradient_checkpointing", True),
        n=rollout.get("n", 4),
        rollout_batch_size=rollout.get("rollout_batch_size", 4),
        timestep_fraction=actor.get("timestep_fraction", 1.0),
        total_epochs=trainer_section.get("total_epochs", 10000),
        save_freq=trainer_section.get("save_freq", 50),
        log_freq=trainer_section.get("log_freq", 1),
        output_dir=trainer_section.get("output_dir", "outputs/"),
        seed=trainer_section.get("seed", 0),
        profile=trainer_section.get("profile", False),
    )


def build_algorithm_config(cfg: DictConfig):
    """Dispatch on ``algorithm.kind`` and return the typed algorithm config.

    Returns ``GRPOConfig`` / ``TokenGRPOConfig`` / ``DiffusionDPOConfig``.
    Unknown kind, missing section, or kind/adv_estimator conflict → fail fast.
    """
    if "algorithm" not in cfg:
        raise ValueError("config missing `algorithm` section")
    algo = cfg.algorithm
    kind = _resolve_algorithm_kind(algo)
    raw = OmegaConf.to_container(algo, resolve=True) or {}

    if kind == "grpo":
        from vrl.algorithms.grpo import GRPOConfig

        return GRPOConfig(**{k: v for k, v in raw.items() if k in _GRPO_FIELDS})

    if kind == "token_grpo":
        from vrl.algorithms.grpo_token import TokenGRPOConfig

        allowed = _GRPO_FIELDS | _TOKEN_GRPO_EXTRA_FIELDS
        return TokenGRPOConfig(**{k: v for k, v in raw.items() if k in allowed})

    if kind == "diffusion_dpo":
        from vrl.algorithms.dpo import DiffusionDPOConfig

        return DiffusionDPOConfig(**{k: v for k, v in raw.items() if k in _DPO_FIELDS})

    raise AssertionError(f"unreachable: kind={kind}")  # pragma: no cover


def build_reward_config(cfg: DictConfig) -> tuple[dict[str, float], dict[str, dict]]:
    """Slice ``cfg.reward`` into ``(weights, kwargs)``.

    - ``weights``: ``{name: float}`` for components with weight > 0.
    - ``kwargs``: ``{name: {kwarg: value}}`` forwarded to reward constructors.

    Back-compat: ``reward.ocr_debug_dir`` is auto-injected into
    ``kwargs["ocr"]["debug_dir"]`` IFF ``kwargs.ocr.debug_dir`` is not already
    set. If both are set with different values → fail fast.
    """
    if "reward" not in cfg:
        raise ValueError("config missing `reward` section")
    reward = cfg.reward
    if "components" not in reward:
        raise ValueError("config missing `reward.components`")

    components = OmegaConf.to_container(reward.components, resolve=True) or {}
    weights = {name: float(w) for name, w in components.items() if float(w) > 0}

    raw_kwargs = reward.get("kwargs", None)
    kwargs: dict[str, dict] = (
        OmegaConf.to_container(raw_kwargs, resolve=True) or {} if raw_kwargs else {}
    )

    # Legacy: reward.ocr_debug_dir
    legacy_ocr_dir = reward.get("ocr_debug_dir", "") if "ocr_debug_dir" in reward else ""
    if legacy_ocr_dir:
        ocr_kwargs = kwargs.setdefault("ocr", {})
        existing = ocr_kwargs.get("debug_dir")
        if existing and existing != legacy_ocr_dir:
            raise ValueError(
                f"reward.ocr_debug_dir={legacy_ocr_dir!r} conflicts with "
                f"reward.kwargs.ocr.debug_dir={existing!r}",
            )
        if not existing:
            ocr_kwargs["debug_dir"] = legacy_ocr_dir

    return weights, kwargs


def build_configs(cfg: DictConfig) -> dict[str, Any]:
    """Thin wrapper bundling typed configs for downstream training scripts.

    Keys:
      ``trainer``    -> ``TrainerConfig``
      ``algorithm``  -> ``GRPOConfig | TokenGRPOConfig | DiffusionDPOConfig``
      ``reward``     -> ``(weights: dict[str, float], kwargs: dict[str, dict])``
      ``raw``        -> the original ``DictConfig``
    """
    out: dict[str, Any] = {
        "trainer": build_trainer_config(cfg),
        "algorithm": build_algorithm_config(cfg),
        "raw": cfg,
    }
    # DPO has no online reward; skip reward slicing for cfgs without `reward`.
    if "reward" in cfg:
        out["reward"] = build_reward_config(cfg)
    return out
