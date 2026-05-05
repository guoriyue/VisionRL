"""Checkpoint helpers for resumable training."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
TRAINING_CHECKPOINT_NAME = "checkpoint.pt"
LORA_WEIGHTS_NAME = "lora_weights"
CHECKPOINT_META_NAME = "checkpoint_meta.json"


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Torch-style training checkpoint payload.

    ``checkpoint.pt`` is the source of truth for resume. Export directories
    such as ``lora_weights/`` are optional artifacts for warm-start/publishing.
    """

    checkpoint_dir: Path
    checkpoint_path: Path
    payload: dict[str, Any]
    meta: dict[str, Any]

    @property
    def trainer_state(self) -> dict[str, Any]:
        trainer = self.payload.get("trainer")
        if not isinstance(trainer, dict):
            raise TypeError("checkpoint payload missing dict field: trainer")
        return trainer

    @property
    def trainable_state(self) -> dict[str, Any]:
        model = self.payload.get("model")
        if not isinstance(model, dict):
            raise TypeError("checkpoint payload missing dict field: model")
        state = model.get("trainable_modules")
        if not isinstance(state, dict):
            raise TypeError("checkpoint payload missing dict field: model.trainable_modules")
        return state

    @property
    def progress(self) -> dict[str, Any]:
        progress = self.payload.get("progress", {})
        if not isinstance(progress, dict):
            raise TypeError("checkpoint payload field progress must be a dict")
        return progress

    @property
    def rng_state(self) -> dict[str, Any]:
        rng = self.payload.get("rng", {})
        if not isinstance(rng, dict):
            raise TypeError("checkpoint payload field rng must be a dict")
        return rng

    @property
    def next_epoch(self) -> int:
        if "next_epoch" in self.progress:
            return _non_negative_int(self.progress["next_epoch"], "progress.next_epoch")
        return infer_next_epoch(self.checkpoint_dir, self.trainer_state, self.meta)

    @property
    def next_step(self) -> int:
        if "next_step" in self.progress:
            return _non_negative_int(self.progress["next_step"], "progress.next_step")
        if "global_step" in self.trainer_state:
            return _non_negative_int(self.trainer_state["global_step"], "trainer.global_step")
        if "step" in self.trainer_state:
            return _non_negative_int(self.trainer_state["step"], "trainer.step")
        return self.next_epoch


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    bundle: Any,
    family: str,
    progress: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
    export_modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a generic Torch training checkpoint.

    Every model family participates through ``RuntimeBundle.trainable_modules``.
    No family-specific serialization code is needed for resume.
    """

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    trainer_state = trainer.state_dict()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "trainer": trainer_state,
        "model": {
            "trainable_modules": export_trainable_state(bundle),
        },
        "progress": dict(progress),
        "rng": rng_state or capture_rng_state(),
    }
    torch.save(payload, path / TRAINING_CHECKPOINT_NAME)

    for name, module in (export_modules or {}).items():
        save_pretrained = getattr(module, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError(f"export module {name!r} does not expose save_pretrained()")
        save_pretrained(path / name)

    meta = write_checkpoint_meta(
        path,
        family=family,
        trainer_state=trainer_state,
        completed_epoch=int(progress.get("completed_epoch", progress.get("next_epoch", 0))),
        next_epoch=int(progress.get("next_epoch", progress.get("next_step", 0))),
        uses_lora=bool((export_modules or {}).get(LORA_WEIGHTS_NAME)),
    )
    meta["checkpoint_file"] = TRAINING_CHECKPOINT_NAME
    (path / CHECKPOINT_META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Load ``checkpoint.pt`` from a checkpoint directory or direct file path."""

    raw_path = Path(path).expanduser().resolve()
    checkpoint_path = raw_path if raw_path.is_file() else raw_path / TRAINING_CHECKPOINT_NAME
    checkpoint_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"training checkpoint file not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{checkpoint_path} must contain a dict payload")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version={schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}",
        )
    return TrainingCheckpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        payload=payload,
        meta=read_checkpoint_meta(checkpoint_dir),
    )


def load_training_checkpoint_from_config(cfg: Any) -> TrainingCheckpoint | None:
    """Return configured resume checkpoint, or ``None`` for a fresh run."""

    resume_from = str(_cfg_path(cfg, "trainer.resume_from", "") or "").strip()
    if not resume_from:
        return None
    return load_training_checkpoint(resume_from)


def prepare_model_config_for_training_resume(
    cfg: Any,
    checkpoint: TrainingCheckpoint | None,
    *,
    strict: bool = True,
) -> None:
    """Remove warm-start adapter paths when doing full training resume.

    Full resume restores ``RuntimeBundle.trainable_modules`` from
    ``checkpoint.pt``. Loading an unrelated ``model.lora.path`` before that can
    silently alter adapter structure, so strict mode rejects the combination.
    """

    if checkpoint is None:
        return
    lora_path = _cfg_path(cfg, "model.lora.path", None)
    if lora_path is None:
        return
    text = str(lora_path or "").strip()
    if text and strict:
        raise ValueError(
            "trainer.resume_from cannot be combined with model.lora.path; "
            "checkpoint.pt is the resume source of truth",
        )
    _set_cfg_path(cfg, "model.lora.path", "")


def restore_training_checkpoint(
    checkpoint: TrainingCheckpoint | None,
    *,
    trainer: Any,
    bundle: Any,
    strict: bool = True,
) -> None:
    """Restore model trainable modules and trainer state from checkpoint."""

    if checkpoint is None:
        return
    checkpoint_family = checkpoint.payload.get("family")
    bundle_family = getattr(bundle, "metadata", {}).get("family")
    if strict and checkpoint_family and bundle_family and str(checkpoint_family) != str(bundle_family):
        raise ValueError(
            f"checkpoint family mismatch: checkpoint={checkpoint_family!r}, "
            f"bundle={bundle_family!r}",
        )
    load_trainable_state(bundle, checkpoint.trainable_state, strict=strict)
    trainer.load_state_dict(checkpoint.trainer_state, strict=strict)


def export_trainable_state(bundle: Any) -> dict[str, dict[str, Any]]:
    """Export all trainable module state_dicts to CPU tensors."""

    modules = _require_trainable_modules(bundle)
    out: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"trainable module {name!r} does not expose state_dict()")
        out[name] = _to_cpu(state_dict())
    return out


def load_trainable_state(
    bundle: Any,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load trainable module state_dicts into a runtime bundle."""

    modules = _require_trainable_modules(bundle)
    missing = sorted(set(modules) - set(state))
    extra = sorted(set(state) - set(modules))
    if strict and (missing or extra):
        raise ValueError(
            "checkpoint trainable module keys mismatch: "
            f"missing={missing}, extra={extra}",
        )
    for name, module in modules.items():
        if name not in state:
            continue
        load_state_dict = getattr(module, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError(f"trainable module {name!r} does not expose load_state_dict()")
        load_state_dict(state[name], strict=strict)


def capture_rng_state(**generators: torch.Generator) -> dict[str, Any]:
    """Capture process RNG state plus named torch.Generator states."""

    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "generators": {name: gen.get_state() for name, gen in generators.items()},
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import random

        state["python_random"] = random.getstate()
    except Exception:
        pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any] | None, **generators: torch.Generator) -> None:
    """Restore process RNG state and named torch.Generator states when present."""

    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    named = state.get("generators", {})
    if isinstance(named, dict):
        for name, gen in generators.items():
            if name in named:
                gen.set_state(named[name])
    if "python_random" in state:
        try:
            import random

            random.setstate(state["python_random"])
        except Exception:
            pass
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass


def save_resolved_config(cfg: Any, output_dir: str | Path, *, resumed: bool) -> None:
    """Save resolved config without overwriting the original on resume."""

    from omegaconf import OmegaConf

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    resolved_path = path / "resolved_config.yaml"
    if not resumed or not resolved_path.exists():
        OmegaConf.save(cfg, resolved_path)
        return
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    OmegaConf.save(cfg, path / f"resume_config_{stamp}.yaml")


def prepare_metrics_csv(csv_path: str | Path, header: str, *, resume: bool) -> None:
    """Create metrics CSV unless resume should append to an existing file."""

    path = Path(csv_path)
    if resume and path.exists():
        return
    if resume:
        logger.warning("Resume requested but metrics file does not exist; creating %s", path)
    path.write_text(header)


def sample_prompt_indices(
    rng: torch.Generator,
    *,
    num_examples: int,
    rollout_batch_size: int,
) -> list[int]:
    """Sample prompt indices with the training prompt generator."""

    if num_examples < 1:
        raise ValueError("prompt manifest must contain at least one example")
    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be >= 1")
    return torch.randperm(num_examples, generator=rng)[:rollout_batch_size].tolist()


def read_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata if present."""

    meta_path = Path(checkpoint_dir) / CHECKPOINT_META_NAME
    if not meta_path.exists():
        return {}
    raw = json.loads(meta_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{meta_path} must contain a JSON object")
    return raw


def infer_next_epoch(
    checkpoint_dir: str | Path,
    trainer_state: dict[str, Any],
    meta: dict[str, Any] | None,
) -> int:
    """Infer the epoch index to start from when resuming."""

    meta = meta or {}
    if "next_epoch" in meta:
        return _non_negative_int(meta["next_epoch"], "checkpoint_meta.next_epoch")

    if "step" in trainer_state:
        return _non_negative_int(trainer_state["step"], "trainer_state.step")

    checkpoint_name = Path(checkpoint_dir).name
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_name)
    if match:
        return _non_negative_int(match.group(1), "checkpoint directory suffix")

    raise ValueError(
        "cannot infer next_epoch: checkpoint_meta.next_epoch and "
        "trainer_state.step are missing",
    )


def write_checkpoint_meta(
    checkpoint_dir: str | Path,
    *,
    family: str,
    trainer_state: dict[str, Any],
    completed_epoch: int,
    next_epoch: int,
    uses_lora: bool,
) -> dict[str, Any]:
    """Write human-readable checkpoint metadata next to ``checkpoint.pt``."""

    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "trainer_step": int(trainer_state.get("step", 0)),
        "global_step": int(trainer_state.get("global_step", 0)),
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(next_epoch),
        "uses_lora": bool(uses_lora),
    }
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / CHECKPOINT_META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def _non_negative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be >= 0, got {parsed}")
    return parsed


def _require_trainable_modules(bundle: Any) -> dict[str, Any]:
    modules = getattr(bundle, "trainable_modules", None)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("RuntimeBundle.trainable_modules must be a non-empty dict")
    return modules


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_cpu(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(inner) for inner in value)
    return value


_MISSING = object()


def _cfg_path(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _set_cfg_path(cfg: Any, path: str, value: Any) -> None:
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return
    try:
        node[keys[-1]] = value
    except TypeError:
        setattr(node, keys[-1], value)


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


__all__ = [
    "CHECKPOINT_META_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "LORA_WEIGHTS_NAME",
    "TRAINING_CHECKPOINT_NAME",
    "TrainingCheckpoint",
    "capture_rng_state",
    "export_trainable_state",
    "infer_next_epoch",
    "load_trainable_state",
    "load_training_checkpoint",
    "load_training_checkpoint_from_config",
    "prepare_metrics_csv",
    "prepare_model_config_for_training_resume",
    "read_checkpoint_meta",
    "restore_rng_state",
    "restore_training_checkpoint",
    "sample_prompt_indices",
    "save_resolved_config",
    "save_training_checkpoint",
    "write_checkpoint_meta",
]
