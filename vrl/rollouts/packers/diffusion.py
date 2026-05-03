"""Diffusion OutputBatch to ExperienceBatch packing."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine.generation import OutputBatch
from vrl.rollouts.experience import ExperienceBatch
from vrl.rollouts.packers.base import RolloutPackContext


class DiffusionRolloutPacker:
    """Pack diffusion trajectory tensors into the CEA ExperienceBatch shape."""

    def __init__(self, *, error_prefix: str) -> None:
        self.error_prefix = error_prefix

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any:
        del context
        return _env_extra(output, self.error_prefix)["videos"]

    def reward_prompts(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> list[str]:
        del context
        return [spec.prompt for spec in output.sample_specs]

    async def pack(
        self,
        output: OutputBatch,
        rewards_raw: torch.Tensor,
        context: RolloutPackContext,
    ) -> ExperienceBatch:
        rt = output.rollout_trajectory_data
        if rt is None or rt.dit_trajectory is None:
            raise RuntimeError(
                f"{self.error_prefix} OutputBatch is missing "
                "rollout_trajectory_data / dit_trajectory",
            )

        observations = rt.dit_trajectory.latents
        timesteps_tensor = rt.dit_trajectory.timesteps
        log_probs = rt.rollout_log_probs
        env_extra = _env_extra(output, self.error_prefix)
        actions = env_extra["actions"]
        kl_tensor = env_extra["kl"]
        training_extras: dict[str, Any] = env_extra["training_extras"]
        rollout_context: dict[str, Any] = env_extra["context"]
        video = env_extra["videos"]

        if observations is None or actions is None or log_probs is None:
            raise RuntimeError(
                f"{self.error_prefix} OutputBatch is missing trajectory tensors "
                "(observations/actions/log_probs)",
            )

        batch_size = observations.shape[0]
        device = observations.device
        if len(output.sample_specs) != batch_size:
            raise RuntimeError(
                f"{self.error_prefix} OutputBatch sample_specs length does not match batch size",
            )

        if context.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
            rewards_adjusted = rewards_raw.to(device) - context.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw.to(device)

        extras: dict[str, Any] = {
            "log_probs": log_probs,
            "timesteps": timesteps_tensor,
            "kl": kl_tensor,
            "reward_before_kl": rewards_raw.to(device),
        }
        extras.update(training_extras)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=torch.ones(batch_size, dtype=torch.bool, device=device),
            group_ids=torch.tensor(
                [spec.prompt_index for spec in output.sample_specs],
                dtype=torch.long,
                device=device,
            ),
            extras=extras,
            context=rollout_context,
            videos=video,
            prompts=[spec.prompt for spec in output.sample_specs],
        )


def _env_extra(output: OutputBatch, error_prefix: str) -> dict[str, Any]:
    rt = output.rollout_trajectory_data
    if rt is None or rt.denoising_env is None:
        raise RuntimeError(f"{error_prefix} OutputBatch is missing denoising_env")
    return rt.denoising_env.extra


__all__ = ["DiffusionRolloutPacker"]
