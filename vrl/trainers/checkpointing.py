"""Checkpoint helpers for resumable online training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 1
TRAINER_STATE_NAME = "trainer_state.pt"
LORA_WEIGHTS_NAME = "lora_weights"
CHECKPOINT_META_NAME = "checkpoint_meta.json"


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    """Resolved checkpoint directory and trainer state for full resume."""

    checkpoint_dir: Path
    trainer_state_path: Path
    lora_weights_path: Path | None
    trainer_state: dict[str, Any]
    meta: dict[str, Any]
    next_epoch: int


def resolve_resume_checkpoint(
    path: str | Path,
    *,
    strict: bool = True,
    uses_lora: bool = True,
) -> ResumeCheckpoint:
    """Resolve a full training checkpoint and load ``trainer_state.pt``.

    ``model.lora.path`` warm-starts only adapter weights. This helper is for
    full training resume, so ``trainer_state.pt`` is always required.
    """

    checkpoint_dir = Path(path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"resume checkpoint must be a directory: {checkpoint_dir}")

    trainer_state_path = checkpoint_dir / TRAINER_STATE_NAME
    if not trainer_state_path.exists():
        raise FileNotFoundError(
            f"resume checkpoint missing {TRAINER_STATE_NAME}: {trainer_state_path}",
        )
    trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    if not isinstance(trainer_state, dict):
        raise TypeError(f"{trainer_state_path} must contain a dict trainer state")

    meta = read_checkpoint_meta(checkpoint_dir)
    next_epoch = infer_next_epoch(checkpoint_dir, trainer_state, meta)

    lora_weights_path = checkpoint_dir / LORA_WEIGHTS_NAME
    if lora_weights_path.exists():
        resolved_lora_path: Path | None = lora_weights_path
    else:
        resolved_lora_path = None
        if strict and uses_lora:
            raise FileNotFoundError(
                f"resume checkpoint missing {LORA_WEIGHTS_NAME}: {lora_weights_path}",
            )

    return ResumeCheckpoint(
        checkpoint_dir=checkpoint_dir,
        trainer_state_path=trainer_state_path,
        lora_weights_path=resolved_lora_path,
        trainer_state=trainer_state,
        meta=meta,
        next_epoch=next_epoch,
    )


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


def save_online_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    family: str,
    completed_epoch: int,
    next_epoch: int,
    lora_module: Any | None = None,
    uses_lora: bool = True,
) -> dict[str, Any]:
    """Save trainer state, optional LoRA adapter weights, and metadata."""

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    trainer_state = trainer.state_dict()
    torch.save(trainer_state, path / TRAINER_STATE_NAME)

    if uses_lora:
        if lora_module is None or not callable(getattr(lora_module, "save_pretrained", None)):
            raise TypeError("LoRA checkpoint save requires a module with save_pretrained()")
        lora_module.save_pretrained(path / LORA_WEIGHTS_NAME)

    return write_checkpoint_meta(
        path,
        family=family,
        trainer_state=trainer_state,
        completed_epoch=completed_epoch,
        next_epoch=next_epoch,
        uses_lora=uses_lora,
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
    """Write checkpoint metadata next to ``trainer_state.pt``."""

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


__all__ = [
    "CHECKPOINT_META_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "LORA_WEIGHTS_NAME",
    "TRAINER_STATE_NAME",
    "ResumeCheckpoint",
    "infer_next_epoch",
    "read_checkpoint_meta",
    "resolve_resume_checkpoint",
    "save_online_checkpoint",
    "write_checkpoint_meta",
]
