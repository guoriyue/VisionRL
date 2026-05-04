"""Continuous-token AR OutputBatch to RolloutBatch packing."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine.generation import OutputBatch
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.packers.base import RolloutPackContext


class ARContinuousRolloutPacker:
    """Pack NextStep-style continuous image-token rollouts."""

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any:
        del context
        return output.extra["images_for_reward"]

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
    ) -> RolloutBatch:
        extra = output.extra
        device = context.device or extra["prompt_input_ids"].device
        tokens = extra["tokens"]
        saved_noise = extra["saved_noise"]
        old_logprobs = extra["log_probs"]
        prompt_ids = extra["prompt_input_ids"]
        prompt_mask = extra["prompt_attention_mask"]
        uncond_ids = extra["uncond_input_ids"]
        uncond_mask = extra["uncond_attention_mask"]
        images = output.output

        token_mask = torch.ones_like(old_logprobs)
        observations = prompt_ids.unsqueeze(1)
        log_probs_3d = old_logprobs.detach().unsqueeze(1)

        return RolloutBatch(
            observations=observations,
            actions=tokens,
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
                "saved_noise": saved_noise,
            },
            context=dict(extra["context"]),
            videos=images.unsqueeze(2),
            prompts=[spec.prompt for spec in output.sample_specs],
        )


__all__ = ["ARContinuousRolloutPacker"]
