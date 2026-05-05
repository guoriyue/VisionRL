from __future__ import annotations

import json

import pytest
import torch

from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    LORA_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    infer_next_epoch,
    resolve_resume_checkpoint,
    save_online_checkpoint,
    write_checkpoint_meta,
)


def test_resolve_resume_checkpoint_reads_meta_next_epoch(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    torch.save({"step": 7, "global_step": 11}, ckpt / TRAINER_STATE_NAME)
    (ckpt / LORA_WEIGHTS_NAME).mkdir()
    write_checkpoint_meta(
        ckpt,
        family="sd3_5",
        trainer_state={"step": 10, "global_step": 12},
        completed_epoch=10,
        next_epoch=10,
        uses_lora=True,
    )

    resolved = resolve_resume_checkpoint(ckpt, strict=True, uses_lora=True)

    assert resolved.next_epoch == 10
    assert resolved.lora_weights_path == ckpt / LORA_WEIGHTS_NAME
    assert resolved.trainer_state["global_step"] == 11
    assert resolved.meta["family"] == "sd3_5"


def test_infer_next_epoch_falls_back_to_trainer_step_for_checkpoint_final(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-final"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {"step": 12}, None) == 12


def test_infer_next_epoch_falls_back_to_numeric_checkpoint_suffix(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {}, {}) == 42


def test_resolve_resume_checkpoint_strict_requires_trainer_state(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / LORA_WEIGHTS_NAME).mkdir()

    with pytest.raises(FileNotFoundError, match=TRAINER_STATE_NAME):
        resolve_resume_checkpoint(ckpt, strict=True, uses_lora=True)


def test_resolve_resume_checkpoint_strict_requires_lora_for_lora_training(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    torch.save({"step": 10}, ckpt / TRAINER_STATE_NAME)

    with pytest.raises(FileNotFoundError, match=LORA_WEIGHTS_NAME):
        resolve_resume_checkpoint(ckpt, strict=True, uses_lora=True)


def test_resolve_resume_checkpoint_allows_missing_lora_for_full_finetune(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    torch.save({"step": 10}, ckpt / TRAINER_STATE_NAME)

    resolved = resolve_resume_checkpoint(ckpt, strict=True, uses_lora=False)

    assert resolved.lora_weights_path is None
    assert resolved.next_epoch == 10


def test_save_online_checkpoint_writes_trainer_state_lora_and_meta(tmp_path) -> None:
    class _Trainer:
        def state_dict(self):
            return {"step": 3, "global_step": 4}

    class _LoraModule:
        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    meta = save_online_checkpoint(
        tmp_path / "checkpoint-3",
        trainer=_Trainer(),
        family="sd3_5",
        completed_epoch=3,
        next_epoch=3,
        lora_module=_LoraModule(),
        uses_lora=True,
    )

    ckpt = tmp_path / "checkpoint-3"
    assert (ckpt / TRAINER_STATE_NAME).exists()
    assert (ckpt / LORA_WEIGHTS_NAME / "adapter_model.safetensors").exists()
    assert (ckpt / CHECKPOINT_META_NAME).exists()
    assert meta["next_epoch"] == 3
    assert meta["global_step"] == 4


def test_read_checkpoint_meta_rejects_non_object_json(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save({"step": 1}, ckpt / TRAINER_STATE_NAME)
    (ckpt / CHECKPOINT_META_NAME).write_text(json.dumps([{"next_epoch": 1}]))

    with pytest.raises(TypeError, match="JSON object"):
        resolve_resume_checkpoint(ckpt, strict=False, uses_lora=False)
