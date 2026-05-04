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


# ---------------------------------------------------------------------------
# Phase 3: Required-access helpers
#
# These replace `.get(path, default)` for active experiment values. The intent
# is that YAML is the single source of truth: if a path is missing, fail fast
# with the exact dotted path in the error message so the operator immediately
# knows which YAML key is expected.
# ---------------------------------------------------------------------------


def require(cfg: DictConfig, path: str) -> Any:
    """Fetch a dotted ``path`` from ``cfg`` or raise with the exact path.

    Raises:
        ValueError: if the path is missing or the resolved value is ``None``.
            The error message always contains the dotted path so callers can
            grep their YAML directly.
    """
    keys = path.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, DictConfig) or k not in node:
            raise ValueError(f"config missing required field: {path}")
        node = node[k]
    if node is None:
        raise ValueError(f"config missing required field: {path} (got None)")
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


def optional_none(cfg: DictConfig, path: str) -> Any | None:
    """Fetch a dotted ``path`` that YAML may legitimately set to ``null``.

    The path being completely absent is still an error — the contract is that
    the YAML *explicitly* opts in to ``null``. Returns ``None`` only if the
    YAML value is ``null``.
    """
    keys = path.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, DictConfig) or k not in node:
            raise ValueError(f"config missing required field: {path}")
        node = node[k]
    if node is None:
        return None
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


# ---------------------------------------------------------------------------
# Phase 2: Reward validation
# ---------------------------------------------------------------------------

# Reward components that are model-backed and therefore must declare their
# backbone explicitly under `reward.kwargs.<name>`. Each entry maps a reward
# name to the list of `kwargs.<name>.<subkey>` paths that are required when
# the component's weight is > 0.
_REWARD_REQUIRED_KWARGS: dict[str, tuple[str, ...]] = {
    "aesthetic": ("model_name",),
    "clipscore": ("model_name",),
    "pickscore": ("processor_name", "model_name"),
    # OCR's `debug_dir` is allowed to be the empty string but the key must be
    # explicitly present in YAML — we validate presence, not non-emptiness.
    "ocr": ("debug_dir",),
}


def validate_reward_config(cfg: DictConfig) -> None:
    """Validate ``cfg.reward`` shape per SPRINT patch 3 Phase 2.

    Rules:
      - ``reward.components`` must exist.
      - For every component with ``weight > 0`` that is model-backed, the
        corresponding ``reward.kwargs.<name>.<field>`` must be present.
      - ``reward.kwargs.<name>`` must be a mapping if present.
    """
    if "reward" not in cfg:
        raise ValueError("config missing required field: reward")
    reward = cfg.reward
    if "components" not in reward:
        raise ValueError("config missing required field: reward.components")

    components_raw = OmegaConf.to_container(reward.components, resolve=True) or {}
    if not isinstance(components_raw, dict):
        raise ValueError("reward.components must be a mapping of name -> weight")

    raw_kwargs_node = reward.get("kwargs", None) if "kwargs" in reward else None
    kwargs_dict: dict[str, Any] = (
        OmegaConf.to_container(raw_kwargs_node, resolve=True) or {}
        if raw_kwargs_node is not None
        else {}
    )

    # `reward.kwargs.<name>` must be a dict if present.
    for name, sub in kwargs_dict.items():
        if sub is None:
            # Treat explicit null as "no kwargs"; downstream check handles the
            # required-keys case below.
            continue
        if not isinstance(sub, dict):
            raise ValueError(
                f"reward.kwargs.{name} must be a mapping, got {type(sub).__name__}",
            )

    # For each model-backed component with weight > 0, its required kwargs
    # subkeys must be present.
    for name, required_subkeys in _REWARD_REQUIRED_KWARGS.items():
        if name not in components_raw:
            continue
        try:
            weight = float(components_raw[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reward.components.{name} must be numeric, got "
                f"{components_raw[name]!r}",
            ) from exc
        if weight <= 0:
            continue
        sub = kwargs_dict.get(name)
        if not isinstance(sub, dict):
            raise ValueError(
                f"config missing required field: reward.kwargs.{name} "
                f"(component {name!r} has weight {weight} > 0)",
            )
        for subkey in required_subkeys:
            if subkey not in sub:
                raise ValueError(
                    f"config missing required field: reward.kwargs.{name}.{subkey}",
                )


# ---------------------------------------------------------------------------
# Phase 6: Experiment-level validation
# ---------------------------------------------------------------------------

# Common required fields for every active training experiment (Phase 4
# lines 209-232). Filesystem-existence checks are explicitly out of scope.
_COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
    "actor.optim.lr",
    "actor.optim.adam_beta1",
    "actor.optim.adam_beta2",
    "actor.optim.weight_decay",
    "actor.optim.eps",
    "actor.optim.use_8bit_adam",
    "actor.optim.allow_tf32",
    "actor.max_norm",
    "actor.ppo_epochs",
    "actor.bf16",
    "actor.gradient_checkpointing",
    "actor.timestep_fraction",
    "actor.ema.enable",
    "actor.ema.decay",
    "actor.ema.update_interval",
    "trainer.entrypoint",
    "trainer.total_epochs",
    "trainer.save_freq",
    "trainer.log_freq",
    "trainer.output_dir",
    "trainer.seed",
    "trainer.profile",
    "trainer.debug.first_step",
    "trainer.debug.grad_split",
)

# GRPO (diffusion) extra rollout fields (Phase 4 lines 234-242).
_GRPO_DIFFUSION_REQUIRED: tuple[str, ...] = (
    "rollout.n",
    "rollout.rollout_batch_size",
    "rollout.noise_level",
    "rollout.sample_batch_size",
    "rollout.sde.window_size",
    "rollout.sde.window_range",
    "rollout.same_latent",
)

# Token-GRPO (autoregressive) extra rollout fields (Phase 4 lines 244-251).
_TOKEN_GRPO_REQUIRED: tuple[str, ...] = (
    "rollout.n_samples_per_prompt",
    "rollout.rollout_batch_size",
    "rollout.rescale_to_unit",
    "rollout.max_text_length",
)

# Diffusion-DPO extra fields per the patch §"Wan-DPO" (lines 322-344).
# `data.max_train_samples` is intentionally excluded here — DPO YAML opts in
# to ``null``, so callers must use ``optional_none`` instead of ``require``.
_DPO_REQUIRED: tuple[str, ...] = (
    "actor.mixed_precision",
    "actor.gradient_checkpointing",
    "actor.train_batch_size",
    "actor.gradient_accumulation_steps",
    "actor.scale_lr",
    "actor.use_adafactor",
    "actor.prediction_type",
    "data.random_crop",
    "data.no_hflip",
    "data.dataloader_num_workers",
    "data.resolution",
    "trainer.max_train_steps",
    "trainer.checkpointing_steps",
    "trainer.log_interval",
    "rollout.n",
    "rollout.rollout_batch_size",
)


def _require_path_present(cfg: DictConfig, path: str) -> None:
    """Like ``require`` but only checks presence — returns nothing.

    This is the validator-side counterpart that gives the same error message
    shape ``require`` does; we don't care about the value, only that YAML
    declares it.
    """
    require(cfg, path)


def _path_exists(cfg: DictConfig, path: str) -> bool:
    """Return True iff a dotted path resolves to a present (non-missing) node.

    Unlike :func:`require`, presence with value ``None`` counts as present;
    this is used for *gating* family-specific dispatch decisions.
    """
    keys = path.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, DictConfig) or k not in node:
            return False
        node = node[k]
    return True


def validate_training_config(cfg: DictConfig) -> None:
    """Top-level fail-fast validator for active training experiments.

    Per SPRINT patch 3 Phase 6:
      1. Common actor/trainer/debug fields must be present and non-None.
      2. ``reward`` is validated only if it's declared (DPO has no online reward).
      3. Algorithm-specific dispatch on ``algorithm.kind``:
            - ``grpo``         -> diffusion rollout fields
            - ``token_grpo``   -> AR rollout fields (+ NextStep noise_level)
            - ``diffusion_dpo``-> DPO actor/data/trainer fields
      4. No filesystem-existence checks happen here (per patch line 390).
    """
    # 1. Common required fields.
    for path in _COMMON_REQUIRED_FIELDS:
        _require_path_present(cfg, path)
    entrypoint = require(cfg, "trainer.entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("trainer.entrypoint must be a non-empty import path")

    # 2. Optional reward block.
    if "reward" in cfg:
        validate_reward_config(cfg)

    # 3. Algorithm-kind dispatch.
    if "algorithm" not in cfg:
        raise ValueError("config missing required field: algorithm")
    kind = _resolve_algorithm_kind(cfg.algorithm)

    if kind == "grpo":
        for path in _GRPO_DIFFUSION_REQUIRED:
            _require_path_present(cfg, path)
    elif kind == "token_grpo":
        for path in _TOKEN_GRPO_REQUIRED:
            _require_path_present(cfg, path)
        # NextStep-1 (continuous-token AR) additionally needs `rollout.noise_level`.
        # Janus (discrete-token AR) does not — gate on `model.family`.
        family = None
        if _path_exists(cfg, "model.family"):
            family = cfg.model.family
        if family == "nextstep_1":
            _require_path_present(cfg, "rollout.noise_level")
    elif kind == "diffusion_dpo":
        for path in _DPO_REQUIRED:
            _require_path_present(cfg, path)
        # `data.max_train_samples` may legitimately be null but the key must
        # be declared — touch it via `optional_none` so missing-key fails.
        optional_none(cfg, "data.max_train_samples")
    else:  # pragma: no cover — _resolve_algorithm_kind already raises.
        raise AssertionError(f"unreachable: kind={kind}")


def _resolve_algorithm_kind(algo: DictConfig) -> str:
    """Resolve ``algorithm.kind`` as the only algorithm dispatch field."""
    if "adv_estimator" in algo:
        raise ValueError("algorithm.adv_estimator is no longer supported; use algorithm.kind")
    kind = algo.get("kind", None)
    if kind is None:
        raise ValueError("algorithm.kind required")
    kind = str(kind)
    if kind not in {"grpo", "token_grpo", "diffusion_dpo"}:
        raise ValueError(
            f"unknown algorithm.kind={kind!r}; "
            f"expected grpo / token_grpo / diffusion_dpo",
        )
    return kind


def build_trainer_config(cfg: DictConfig):
    """Slice merged YAML into ``TrainerConfig`` via fail-fast required access.

    Per SPRINT patch 3 Phase 4: every actor/trainer/debug field must come
    from YAML — no Python-side experiment-default fallbacks allowed. The
    rollout group schema is family-specific: AR rollouts declare
    ``n_samples_per_prompt`` and diffusion rollouts declare ``n``.
    """
    from vrl.trainers.types import (
        DebugConfig,
        EMAConfig,
        OptimConfig,
        TrainerConfig,
    )

    # Build nested dataclasses from YAML — `require` already errors with
    # exact paths, so the unpack below is guaranteed to receive complete
    # dicts when validation runs first.
    optim_dict = require(cfg, "actor.optim")
    ema_dict = require(cfg, "actor.ema")
    debug_dict = require(cfg, "trainer.debug")

    # Rollout n / batch. AR uses `n_samples_per_prompt`, diffusion uses `n`.
    # Offline trainers still declare explicit values so config slicing stays
    # single-source and fail-fast.
    if _path_exists(cfg, "rollout.n_samples_per_prompt"):
        n_value = require(cfg, "rollout.n_samples_per_prompt")
        rollout_batch_size = require(cfg, "rollout.rollout_batch_size")
    elif _path_exists(cfg, "rollout.n"):
        n_value = require(cfg, "rollout.n")
        rollout_batch_size = require(cfg, "rollout.rollout_batch_size")
    else:
        raise ValueError(
            "config missing rollout group size: expected rollout.n or "
            "rollout.n_samples_per_prompt",
        )

    return TrainerConfig(
        optim=OptimConfig(**optim_dict),
        ema=EMAConfig(**ema_dict),
        debug=DebugConfig(**debug_dict),
        max_norm=require(cfg, "actor.max_norm"),
        ppo_epochs=require(cfg, "actor.ppo_epochs"),
        bf16=require(cfg, "actor.bf16"),
        gradient_checkpointing=require(cfg, "actor.gradient_checkpointing"),
        n=n_value,
        rollout_batch_size=rollout_batch_size,
        timestep_fraction=require(cfg, "actor.timestep_fraction"),
        total_epochs=require(cfg, "trainer.total_epochs"),
        save_freq=require(cfg, "trainer.save_freq"),
        log_freq=require(cfg, "trainer.log_freq"),
        output_dir=require(cfg, "trainer.output_dir"),
        seed=require(cfg, "trainer.seed"),
        profile=require(cfg, "trainer.profile"),
    )


def build_algorithm_config(cfg: DictConfig):
    """Dispatch on ``algorithm.kind`` and return the typed algorithm config.

    Returns ``GRPOConfig`` / ``TokenGRPOConfig`` / ``DiffusionDPOConfig``.
    Unknown kind or missing section → fail fast.
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

    Reward-specific options must live under ``reward.kwargs.<component>``.
    """
    # Fail-fast on shape before slicing; this keeps the contract that the
    # active experiment YAML is the single source of truth for reward
    # backbones (no silent constructor fallback).
    validate_reward_config(cfg)
    reward = cfg.reward

    components = OmegaConf.to_container(reward.components, resolve=True) or {}
    weights = {name: float(w) for name, w in components.items() if float(w) > 0}

    raw_kwargs = reward.get("kwargs", None)
    kwargs: dict[str, dict] = (
        OmegaConf.to_container(raw_kwargs, resolve=True) or {} if raw_kwargs else {}
    )

    return weights, kwargs


def build_configs(cfg: DictConfig) -> dict[str, Any]:
    """Thin wrapper bundling typed configs for downstream training scripts.

    Keys:
      ``trainer``    -> ``TrainerConfig``
      ``algorithm``  -> ``GRPOConfig | TokenGRPOConfig | DiffusionDPOConfig``
      ``reward``     -> ``(weights: dict[str, float], kwargs: dict[str, dict])``
      ``raw``        -> the original ``DictConfig``
    """
    validate_training_config(cfg)
    out: dict[str, Any] = {
        "trainer": build_trainer_config(cfg),
        "algorithm": build_algorithm_config(cfg),
        "raw": cfg,
    }
    # DPO has no online reward; skip reward slicing for cfgs without `reward`.
    if "reward" in cfg:
        out["reward"] = build_reward_config(cfg)
    return out
