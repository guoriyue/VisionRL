"""Discrete-token AR OutputBatch to ExperienceBatch packing."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine.generation import OutputBatch
from vrl.rollouts.experience import ExperienceBatch
from vrl.rollouts.packers.base import RolloutPackContext


class ARDiscreteRolloutPacker:
    """Pack Janus-style discrete image-token rollouts."""

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any:
        images = output.output
        if context.rescale_to_unit:
            images = (images + 1.0) * 0.5
            images = images.clamp(0.0, 1.0)
        return images

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
        device = context.device or output.extra["prompt_input_ids"].device
        images = output.output
        token_ids = output.extra["token_ids"]
        token_log_probs = output.extra["token_log_probs"]
        token_mask = output.extra["token_mask"]
        prompt_ids = output.extra["prompt_input_ids"]
        prompt_mask = output.extra["prompt_attention_mask"]
        uncond_ids = output.extra["uncond_input_ids"]
        uncond_mask = output.extra["uncond_attention_mask"]

        observations = prompt_ids.unsqueeze(1)
        log_probs_3d = token_log_probs.detach().unsqueeze(1)

        return ExperienceBatch(
            observations=observations,
            actions=token_ids,
            rewards=rewards_raw.to(device),
            dones=torch.ones(len(output.sample_specs), dtype=torch.bool, device=device),
            group_ids=torch.tensor(
                [spec.prompt_index for spec in output.sample_specs],
                dtype=torch.long,
                device=device,
            ),
            extras={
                "log_probs": log_probs_3d,
                "prompt_attention_mask": prompt_mask,
                "uncond_input_ids": uncond_ids,
                "uncond_attention_mask": uncond_mask,
                "token_mask": token_mask,
            },
            context=dict(output.extra.get("context", {})),
            videos=images.unsqueeze(2),
            prompts=[spec.prompt for spec in output.sample_specs],
        )


__all__ = ["ARDiscreteRolloutPacker"]
