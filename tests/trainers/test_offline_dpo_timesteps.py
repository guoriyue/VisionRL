"""Red-line tests for vrl.trainers.offline_dpo._sample_timesteps.

Catches the silent fallback where an empty ``scheduler.timesteps`` would
quietly substitute ``num_train_timesteps`` and shift the RL sampling
distribution without the trainer noticing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.trainers.offline_dpo import (
    OfflineDPOTrainer,
    OfflineDPOTrainerConfig,
    wan_forward,
)


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
        # Avoid setting ``timesteps`` - getattr should return None.
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


class _WanTransformerStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
            },
        )
        assert return_dict is False
        return (hidden_states * self.weight + encoder_hidden_states,)


class _DiffusionPolicyWrapperStub(torch.nn.Module):
    def __init__(self, transformer: _WanTransformerStub) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, *args, **kwargs):  # pragma: no cover - red-line guard
        raise AssertionError("wan_forward must call the registered transformer")


def test_wan_forward_unwraps_policy_transformer() -> None:
    transformer = _WanTransformerStub()
    policy = _DiffusionPolicyWrapperStub(transformer)
    noisy = torch.ones(2, 3)
    timesteps = torch.tensor([1, 2])
    encoder_hidden_states = torch.full((2, 3), 0.5)

    out = wan_forward(policy, noisy, timesteps, encoder_hidden_states)

    assert torch.equal(out, noisy * transformer.weight + encoder_hidden_states)
    assert transformer.calls == [
        {
            "hidden_states": noisy,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
        },
    ]


def test_wan_forward_still_accepts_raw_transformer() -> None:
    transformer = _WanTransformerStub()
    noisy = torch.ones(1, 2)
    timesteps = torch.tensor([4])
    encoder_hidden_states = torch.full((1, 2), 0.25)

    out = wan_forward(transformer, noisy, timesteps, encoder_hidden_states)

    assert torch.equal(out, noisy * transformer.weight + encoder_hidden_states)
    assert len(transformer.calls) == 1
