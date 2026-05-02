"""Wan 2.1 GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches Wan GRPO configs here.
Pipeline construction, LoRA, collector wiring, training loop, and checkpointing
live in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrl.engine.generation import RolloutBackend


async def train_wan_2_1_grpo(
    cfg: DictConfig,
    *,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    """Run Wan-family GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.algorithms.grpo_token import TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs
    from vrl.engine.generation import build_rollout_backend_from_cfg
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.wan_2_1 import (
        Wan_2_1Collector,
        Wan_2_1CollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]
    if not isinstance(grpo_config, GRPOConfig) or isinstance(grpo_config, TokenGRPOConfig):
        raise TypeError(
            f"Wan-GRPO expects algorithm.kind=grpo, got {type(grpo_config).__name__}",
        )

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    # 1. Build policy bundle (backend construction lives in family builder)
    from vrl.models.families.wan_2_1.builder import build_wan_2_1_runtime_bundle_from_cfg

    bundle = build_wan_2_1_runtime_bundle_from_cfg(cfg, device, weight_dtype)
    wan_model = bundle.policy
    transformer = bundle.trainable_modules["transformer"]

    if trainer_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 2. Reward
    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_fn = MultiReward.from_dict(
        reward_weights, device=str(device), reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)

    # 3. Collector + evaluator + algorithm
    collector_config = Wan_2_1CollectorConfig(
        num_steps=cfg.sampling.num_steps,
        guidance_scale=cfg.sampling.guidance_scale,
        height=cfg.sampling.height,
        width=cfg.sampling.width,
        num_frames=cfg.sampling.num_frames,
        cfg=cfg.sampling.cfg,
        kl_reward=cfg.algorithm.kl_reward,
        sde_window_size=cfg.rollout.sde.window_size,
        sde_window_range=tuple(cfg.rollout.sde.window_range),
        same_latent=cfg.rollout.same_latent,
    )
    collector = Wan_2_1Collector(wan_model, reward_fn, collector_config)
    collector._runtime = build_rollout_backend_from_cfg(
        cfg,
        runtime=rollout_runtime,
        local_runtime_builder=collector._build_runtime,
        driver_bundle=bundle,
    )

    evaluator = FlowMatchingEvaluator(
        bundle.scheduler, noise_level=1.0, sde_type="sde",
    )
    algorithm = GRPO(grpo_config)

    ref_model = transformer if cfg.model.use_lora and grpo_config.init_kl_coef > 0 else None
    stat_tracker = (
        PerPromptStatTracker(global_std=grpo_config.global_std)
        if cfg.algorithm.per_prompt_stat_tracking else None
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=transformer,
        ref_model=ref_model,
        config=trainer_config,
        device=device,
        stat_tracker=stat_tracker,
    )

    # 4. Prompts
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples: list[PromptExample] = load_prompt_manifest(manifest_path)
    logger.info(
        "Starting training — %d epochs, %d examples, n=%d",
        trainer_config.total_epochs, len(examples), trainer_config.n,
    )

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{n}" for n in component_names)
    csv_path.write_text(
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
        "adv_zero_rate,group_size,trained_prompt_num," + component_cols + "\n"
    )

    rng = torch.Generator().manual_seed(trainer_config.seed)

    # 5. Training loop
    for epoch in range(trainer_config.total_epochs):
        idx = torch.randperm(len(examples), generator=rng)[
            : trainer_config.rollout_batch_size
        ].tolist()
        example_batch = [examples[i] for i in idx]

        metrics = await trainer.step(example_batch)

        if epoch % trainer_config.log_freq == 0:
            last = getattr(reward_fn, "last_components", {}) or {}
            component_means = {
                n: (sum(last.get(n, [])) / len(last.get(n, []))) if last.get(n) else float("nan")
                for n in component_names
            }
            component_str = " ".join(
                f"{n}={component_means[n]:.3f}" for n in component_names
            )
            logger.info(
                "Epoch %d | loss=%.4f kl=%.4f reward=%.4f+/-%.4f "
                "grad_norm=%.4f adv_sat=%.3f adv_zero=%.3f "
                "group_size=%.1f seen_prompts=%d | %s",
                epoch, metrics.loss, metrics.kl_penalty,
                metrics.reward_mean, metrics.reward_std,
                metrics.grad_norm, metrics.adv_saturation, metrics.adv_zero_rate,
                metrics.group_size, metrics.trained_prompt_num, component_str,
            )
            with open(csv_path, "a") as f:
                vals = ",".join(f"{component_means[n]:.4f}" for n in component_names)
                f.write(
                    f"{epoch},{metrics.loss:.6f},{metrics.policy_loss:.6f},"
                    f"{metrics.kl_penalty:.6f},{metrics.reward_mean:.4f},"
                    f"{metrics.reward_std:.4f},{metrics.clip_fraction:.4f},"
                    f"{metrics.approx_kl:.6f},{metrics.advantage_mean:.6f},"
                    f"{metrics.grad_norm:.6f},{metrics.adv_saturation:.4f},"
                    f"{metrics.adv_zero_rate:.4f},{metrics.group_size:.2f},"
                    f"{metrics.trained_prompt_num}," + vals + "\n"
                )

        if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch+1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt_path / "lora_weights")
            logger.info("Saved checkpoint to %s", ckpt_path)

    final_path = output_dir / "checkpoint-final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), final_path / "trainer_state.pt")
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(final_path / "lora_weights")
    logger.info("Training complete. Final checkpoint: %s", final_path)
