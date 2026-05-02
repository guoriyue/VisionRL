"""Flow-matching signal extraction for diffusion model training."""

from __future__ import annotations

import contextlib
from typing import Any

from vrl.algorithms.diffusion.sde import (  # noqa: F401
    SDEStepResult,
    compute_kl_divergence,
    sde_step_with_logprob,
)
from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.types import ExperienceBatch

# ------------------------------------------------------------------
# FlowMatchingEvaluator
# ------------------------------------------------------------------

class FlowMatchingEvaluator:
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
        collector: Any,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Replay one diffusion step -> sde_step_with_logprob -> SignalBatch.

        Replay forward ownership lives on the policy (``model.replay_forward``).
        ``collector`` is retained in the signature for trainer-interface
        compatibility; the body never reads it.

        When ref_model is the same object as model (LoRA scenario),
        uses ``disable_adapter()`` to get base-model predictions —
        matching flow_grpo train_wan2_1.py:940.
        """
        import torch

        del collector  # replay ownership now on the policy

        if signal_request is None:
            signal_request = SignalRequest()

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        observations = batch.observations[:, timestep_idx]  # x_t
        actions = batch.actions[:, timestep_idx]             # x_{t-1}

        # Forward pass through current model — policy owns the replay math.
        fwd = model.replay_forward(batch, timestep_idx, model=model)
        noise_pred = fwd["noise_pred"]

        # SDE step with log-prob
        result = sde_step_with_logprob(
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
                use_adapter_disable = (
                    ref_model is model
                    and hasattr(model, "disable_adapter")
                )
                ctx = model.disable_adapter() if use_adapter_disable else contextlib.nullcontext()

                with ctx:
                    ref_fwd = ref_model.replay_forward(
                        batch, timestep_idx, model=ref_model,
                    )
                    ref_noise_pred = ref_fwd["noise_pred"]

                    ref_result = sde_step_with_logprob(
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
