from __future__ import annotations

import json

import pytest
import torch

from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    LORA_WEIGHTS_NAME,
    TRAINING_CHECKPOINT_NAME,
    infer_next_epoch,
    load_trainable_state,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


def test_training_checkpoint_round_trips_trainer_and_trainable_modules(tmp_path) -> None:

    trainer = _Trainer()
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(3.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-2",
        trainer=trainer,
        bundle=source,
        family="unit",
        progress={"next_epoch": 2, "global_step": 5},
        rng_state={},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-2")
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.fill_(0.0)
    restore_training_checkpoint(
        checkpoint,
        trainer=trainer,
        bundle=restored,
        strict=True,
    )

    assert (tmp_path / "checkpoint-2" / TRAINING_CHECKPOINT_NAME).exists()
    assert checkpoint.next_epoch == 2
    assert trainer.loaded == {"step": 2, "global_step": 5}
    assert restored.module.weight.item() == pytest.approx(3.0)


def test_training_checkpoint_writes_optional_lora_export(tmp_path) -> None:
    class _ExportModule:
        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    save_training_checkpoint(
        tmp_path / "checkpoint-1",
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        export_modules={LORA_WEIGHTS_NAME: _ExportModule()},
    )

    assert (tmp_path / "checkpoint-1" / TRAINING_CHECKPOINT_NAME).exists()
    assert (tmp_path / "checkpoint-1" / LORA_WEIGHTS_NAME / "adapter_model.safetensors").exists()
    assert (tmp_path / "checkpoint-1" / CHECKPOINT_META_NAME).exists()


def test_load_training_checkpoint_requires_checkpoint_pt(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()

    with pytest.raises(FileNotFoundError, match=TRAINING_CHECKPOINT_NAME):
        load_training_checkpoint(ckpt)


def test_load_training_checkpoint_rejects_bad_schema(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save({"schema_version": 999}, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises(ValueError, match="schema_version"):
        load_training_checkpoint(ckpt)


def test_load_trainable_state_strict_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="missing"):
        load_trainable_state(_Bundle(), {}, strict=True)


def test_infer_next_epoch_falls_back_to_trainer_step_for_checkpoint_final(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-final"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {"step": 12}, None) == 12


def test_infer_next_epoch_falls_back_to_numeric_checkpoint_suffix(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {}, {}) == 42


def test_load_training_checkpoint_rejects_non_object_meta(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "trainer": {},
            "model": {"trainable_modules": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )
    (ckpt / CHECKPOINT_META_NAME).write_text(json.dumps([{"next_epoch": 1}]))

    with pytest.raises(TypeError, match="JSON object"):
        load_training_checkpoint(ckpt)


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"step": 2, "global_step": 5}

    def load_state_dict(self, state, *, strict=True):
        del strict
        self.loaded = dict(state)


class _Bundle:
    def __init__(self) -> None:
        import torch.nn as nn

        self.module = nn.Linear(1, 1, bias=False)
        self.trainable_modules = {"module": self.module}
        self.metadata = {"family": "unit"}
