"""SD 3.5 GRPO training recipe.

Entry-point scripts in this directory delegate to ``train_sd3_5_grpo`` here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


async def train_sd3_5_grpo(cfg: DictConfig) -> None:
    """Run SD 3.5 GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo import GRPO
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.sd3_5 import (
        SD3_5Collector,
        SD3_5CollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer

    from vrl.algorithms.grpo import GRPOConfig as _GRPOConfig
    from vrl.algorithms.grpo_token import TokenGRPOConfig as _TokenGRPOConfig

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]
    if not isinstance(grpo_config, _GRPOConfig) or isinstance(grpo_config, _TokenGRPOConfig):
        raise TypeError(
            f"SD3.5 expects algorithm.kind=grpo, got {type(grpo_config).__name__}",
        )

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    # 1. Build runtime (backend construction lives in family builder)
    from vrl.models.families.sd3_5.builder import build_sd3_5_runtime_bundle_from_cfg

    bundle = build_sd3_5_runtime_bundle_from_cfg(cfg, device, weight_dtype)
    sd3_5_model = bundle.policy
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

    # 3. Collector + algorithm
    collector_config = SD3_5CollectorConfig(
        num_steps=cfg.sampling.num_steps,
        guidance_scale=cfg.sampling.guidance_scale,
        height=cfg.sampling.height,
        width=cfg.sampling.width,
        noise_level=cfg.rollout.get("noise_level", 0.7),
        cfg=cfg.sampling.cfg,
        sample_batch_size=cfg.rollout.get("sample_batch_size", 8),
        kl_reward=cfg.algorithm.kl_reward,
        sde_window_size=cfg.rollout.sde.window_size,
        sde_window_range=tuple(cfg.rollout.sde.window_range),
    )
    collector = SD3_5Collector(sd3_5_model, reward_fn, collector_config)

    evaluator = FlowMatchingEvaluator(
        bundle.scheduler, noise_level=cfg.rollout.get("noise_level", 0.7), sde_type="sde",
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

    # 4. Prompts + loop
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples: list[PromptExample] = load_prompt_manifest(manifest_path)
    logger.info(
        "Starting SD3 GRPO — %d epochs, %d examples, n=%d",
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
                "grad_norm=%.4f adv_sat=%.3f adv_zero=%.3f | %s",
                epoch, metrics.loss, metrics.kl_penalty,
                metrics.reward_mean, metrics.reward_std,
                metrics.grad_norm, metrics.adv_saturation, metrics.adv_zero_rate,
                component_str,
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
