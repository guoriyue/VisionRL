"""Flow-matching signal extraction for diffusion model training."""

from __future__ import annotations

import contextlib
from typing import Any

import vrl.algorithms.flow_matching as flow_matching_math
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest

# ------------------------------------------------------------------
# FlowMatchingEvaluator
# ------------------------------------------------------------------

class FlowMatchingEvaluator(Evaluator):
    """Signal extraction for flow-matching diffusion models.

    Uses ``sde_step_with_logprob`` to compute log-probabilities and
    optionally reference model signals for latent-space KL.
    """

    def __init__(
        self,
        scheduler: Any,
        noise_level: float = 1.0,
        sde_type: str = "sde",
    ) -> None:
        self.scheduler = scheduler
        self.noise_level = noise_level
        self.sde_type = sde_type

    def evaluate(
        self,
        model: Any,
        batch: RolloutBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Replay one diffusion step -> sde_step_with_logprob -> SignalBatch.

        Replay forward ownership lives on the family policy adapter. ``model``
        must be that policy and must expose ``replay_forward``.

        When ref_model is the same object as model (LoRA scenario),
        uses ``disable_adapter()`` to get base-model predictions —
        matching flow_grpo train_wan2_1.py:940.
        """
        import torch

        if signal_request is None:
            signal_request = SignalRequest()

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        observations = batch.observations[:, timestep_idx]  # x_t
        actions = batch.actions[:, timestep_idx]             # x_{t-1}

        fwd = model.replay_forward(batch, timestep_idx)
        noise_pred = fwd["noise_pred"]

        # SDE step with log-prob
        result = flow_matching_math.sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            t,
            observations,
            prev_sample=actions,
            return_dt=signal_request.need_kl_intermediates,
            noise_level=self.noise_level,
            sde_type=self.sde_type,
        )

        ref_log_prob = None
        ref_prev_sample_mean = None
        ref_dt = None

        # Reference model forward for KL
        if signal_request.need_ref and ref_model is not None:
            with torch.no_grad():
                # Gap 7: LoRA disable_adapter() — when ref_model IS model,
                # disable LoRA adapter to get base model output.
                # Port from flow_grpo train_wan2_1.py:940:
                #   with transformer.module.disable_adapter():
                use_adapter_disable = ref_model is model and hasattr(
                    model, "disable_adapter",
                )
                ctx = model.disable_adapter() if use_adapter_disable else contextlib.nullcontext()

                with ctx:
                    ref_fwd = ref_model.replay_forward(batch, timestep_idx)
                    ref_noise_pred = ref_fwd["noise_pred"]

                    ref_result = flow_matching_math.sde_step_with_logprob(
                        self.scheduler,
                        ref_noise_pred,
                        t,
                        observations,
                        prev_sample=actions,
                        return_dt=signal_request.need_kl_intermediates,
                        noise_level=self.noise_level,
                        sde_type=self.sde_type,
                    )
                    ref_log_prob = ref_result.log_prob
                    ref_prev_sample_mean = ref_result.prev_sample_mean
                    ref_dt = ref_result.dt

        return SignalBatch(
            log_prob=result.log_prob,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=result.prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            std_dev_t=result.std_dev_t,
            dt=result.dt if result.dt is not None else ref_dt,
            dist_family="flow_matching",
        )
