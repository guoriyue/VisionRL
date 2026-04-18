"""Red-line tests for vrl.trainers.offline_dpo._sample_timesteps.

Catches the silent fallback where an empty ``scheduler.timesteps`` would
quietly substitute ``num_train_timesteps`` and shift the RL sampling
distribution without the trainer noticing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.trainers.offline_dpo import OfflineDPOTrainer, OfflineDPOTrainerConfig


def _noop_forward(model, noisy, ts, encoder, extra=None):  # pragma: no cover
    return noisy


def _noop_encode_pix(pix):  # pragma: no cover
    return pix


def _noop_encode_text(captions):  # pragma: no cover
    return None


def _make_trainer(scheduler_timesteps, *, timestep_subset=None) -> OfflineDPOTrainer:
    scheduler = SimpleNamespace(
        timesteps=scheduler_timesteps,
        config=SimpleNamespace(num_train_timesteps=1000),
    )
    cfg = OfflineDPOTrainerConfig(
        prediction_type="epsilon",
        timestep_subset=timestep_subset,
        num_frames=1,
    )
    return OfflineDPOTrainer(
        model=torch.nn.Linear(4, 4),
        ref_model=None,
        forward_fn=_noop_forward,
        noise_scheduler=scheduler,
        encode_pixels=_noop_encode_pix,
        encode_text=_noop_encode_text,
        config=cfg,
        device="cpu",
    )


class TestSampleTimesteps:
    def test_uses_scheduler_timesteps_when_set(self) -> None:
        trainer = _make_trainer(torch.arange(20))
        ts = trainer._sample_timesteps(8)
        assert ts.shape == (8,)
        assert (ts >= 0).all() and (ts < 20).all()

    def test_explicit_subset_takes_precedence(self) -> None:
        # Subset wins even if scheduler has its own timesteps.
        trainer = _make_trainer(torch.arange(50), timestep_subset=(5, 10))
        ts = trainer._sample_timesteps(20)
        assert (ts >= 5).all() and (ts < 10).all()

    def test_empty_timesteps_raises(self) -> None:
        """Red-line: do not silently fall back to num_train_timesteps."""
        trainer = _make_trainer(torch.empty(0, dtype=torch.long))
        with pytest.raises(RuntimeError, match="set_timesteps"):
            trainer._sample_timesteps(4)

    def test_missing_timesteps_attr_raises(self) -> None:
        """Same red line when the scheduler doesn't expose timesteps at all."""
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        # Avoid setting ``timesteps`` — getattr should return None.
        cfg = OfflineDPOTrainerConfig(prediction_type="epsilon", num_frames=1)
        trainer = OfflineDPOTrainer(
            model=torch.nn.Linear(4, 4),
            ref_model=None,
            forward_fn=_noop_forward,
            noise_scheduler=scheduler,
            encode_pixels=_noop_encode_pix,
            encode_text=_noop_encode_text,
            config=cfg,
            device="cpu",
        )
        with pytest.raises(RuntimeError, match="set_timesteps"):
            trainer._sample_timesteps(4)
