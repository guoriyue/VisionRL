"""Wan 2.1 GRPO training recipe.

Every Wan entry-point in this directory (1.3B basic / multi-reward / OCR,
14B) is a thin wrapper that picks a YAML config and delegates to
``train_wan_grpo`` here. Pipeline construction, LoRA, collector wiring,
training loop, and checkpointing live in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


async def train_wan_grpo(cfg: DictConfig) -> None:
    """Run Wan-family GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo import GRPO
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.wan2_1 import (
        Wan21Collector,
        Wan21CollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    # 1. Pipeline
    from diffusers import WanPipeline

    logger.info("Loading WanPipeline from %s", cfg.model.path)
    pipeline = WanPipeline.from_pretrained(cfg.model.path, torch_dtype=weight_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=weight_dtype)

    # 2. LoRA
    if cfg.model.use_lora:
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer.to(device)
        from peft import LoraConfig, PeftModel, get_peft_model

        lora_path = cfg.model.lora.path
        if lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, lora_path, is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            lora_config = LoraConfig(
                r=cfg.model.lora.rank,
                lora_alpha=cfg.model.lora.alpha,
                init_lora_weights="gaussian",
                target_modules=list(cfg.model.lora.target_modules),
            )
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d)",
            cfg.model.lora.rank, cfg.model.lora.alpha,
        )
    else:
        pipeline.transformer.requires_grad_(True)
        pipeline.transformer.to(device)

    transformer = pipeline.transformer

    if trainer_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if cfg.model.torch_compile.enable:
        logger.info("Compiling transformer with mode=%s", cfg.model.torch_compile.mode)
        pipeline.transformer = torch.compile(
            pipeline.transformer, mode=cfg.model.torch_compile.mode, fullgraph=False,
        )
        transformer = pipeline.transformer

    # 3. Scheduler + reward
    pipeline.scheduler.set_timesteps(cfg.generation.num_steps, device=device)

    reward_weights = {
        name: float(w) for name, w in cfg.reward.components.items() if float(w) > 0
    }
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_kwargs: dict[str, dict] = {}
    if "ocr" in reward_weights and cfg.reward.get("ocr_debug_dir", ""):
        reward_kwargs["ocr"] = {"debug_dir": cfg.reward.ocr_debug_dir}
    reward_fn = MultiReward.from_dict(
        reward_weights, device=str(device), reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)

    # 4. Collector + evaluator + algorithm
    collector_config = Wan21CollectorConfig(
        num_steps=cfg.generation.num_steps,
        guidance_scale=cfg.generation.guidance_scale,
        height=cfg.generation.height,
        width=cfg.generation.width,
        num_frames=cfg.generation.num_frames,
        cfg=cfg.rollout.cfg,
        kl_reward=cfg.rollout.kl_reward,
        sde_window_size=cfg.rollout.sde.window_size,
        sde_window_range=tuple(cfg.rollout.sde.window_range),
        same_latent=cfg.rollout.same_latent,
    )
    from vrl.models.families.wan2_1.diffusers_t2v import DiffusersWanT2VModel

    wan_model = DiffusersWanT2VModel(pipeline=pipeline, device=device)
    collector = Wan21Collector(wan_model, reward_fn, collector_config)

    evaluator = FlowMatchingEvaluator(
        pipeline.scheduler, noise_level=1.0, sde_type="sde",
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

    # 5. Prompts
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

    # 6. Training loop
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
