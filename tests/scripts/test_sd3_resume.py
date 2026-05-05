from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.scripts.sd3_5.train import (
    _apply_resume_lora_path,
    _prepare_metrics_csv,
    advance_prompt_rng,
    sample_prompt_indices,
)
from vrl.trainers.checkpointing import ResumeCheckpoint


def test_sd3_resume_prompt_rng_skip_matches_continuous_sampling() -> None:
    continuous_rng = torch.Generator().manual_seed(123)
    epoch0 = sample_prompt_indices(
        continuous_rng,
        num_examples=20,
        rollout_batch_size=5,
    )
    epoch1 = sample_prompt_indices(
        continuous_rng,
        num_examples=20,
        rollout_batch_size=5,
    )
    epoch2 = sample_prompt_indices(
        continuous_rng,
        num_examples=20,
        rollout_batch_size=5,
    )

    resumed_rng = torch.Generator().manual_seed(123)
    advance_prompt_rng(
        resumed_rng,
        num_examples=20,
        rollout_batch_size=5,
        completed_epochs=2,
    )
    resumed_epoch2 = sample_prompt_indices(
        resumed_rng,
        num_examples=20,
        rollout_batch_size=5,
    )

    assert epoch0 != epoch1
    assert resumed_epoch2 == epoch2


def test_sd3_metrics_csv_append_does_not_rewrite_history(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    header = "epoch,loss\n"
    csv_path.write_text(header + "0,1.0\n")

    _prepare_metrics_csv(csv_path, header, resume=True)

    assert csv_path.read_text() == header + "0,1.0\n"


def test_sd3_metrics_csv_new_run_writes_header(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"

    _prepare_metrics_csv(csv_path, "epoch,loss\n", resume=False)

    assert csv_path.read_text() == "epoch,loss\n"


def test_sd3_resume_overrides_empty_lora_path(tmp_path) -> None:
    resume_lora = tmp_path / "checkpoint-2" / "lora_weights"
    resume_lora.mkdir(parents=True)
    cfg = OmegaConf.create({"model": {"use_lora": True, "lora": {"path": ""}}})
    checkpoint = _resume_checkpoint(tmp_path / "checkpoint-2", resume_lora)

    _apply_resume_lora_path(cfg, checkpoint, strict=True)

    assert cfg.model.lora.path == str(resume_lora.resolve())


def test_sd3_resume_rejects_mismatched_lora_path_in_strict_mode(tmp_path) -> None:
    resume_lora = tmp_path / "checkpoint-2" / "lora_weights"
    other_lora = tmp_path / "other_lora"
    resume_lora.mkdir(parents=True)
    other_lora.mkdir()
    cfg = OmegaConf.create(
        {"model": {"use_lora": True, "lora": {"path": str(other_lora)}}},
    )
    checkpoint = _resume_checkpoint(tmp_path / "checkpoint-2", resume_lora)

    with pytest.raises(ValueError, match="different adapters"):
        _apply_resume_lora_path(cfg, checkpoint, strict=True)


def _resume_checkpoint(checkpoint_dir: Path, lora_path: Path) -> ResumeCheckpoint:
    return ResumeCheckpoint(
        checkpoint_dir=checkpoint_dir.resolve(),
        trainer_state_path=checkpoint_dir / "trainer_state.pt",
        lora_weights_path=lora_path.resolve(),
        trainer_state={"step": 2, "global_step": 2},
        meta={"family": "sd3_5", "next_epoch": 2},
        next_epoch=2,
    )
