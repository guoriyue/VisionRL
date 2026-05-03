"""Cosmos Predict2 GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches Cosmos configs here.
Pipeline construction, LoRA, collector wiring, training loop, checkpointing,
paired LoRA-vs-base eval, and middle-frame PNG dumps all live in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrl.engine.generation import RolloutBackend


async def train_cosmos_predict2_grpo(
    cfg: DictConfig,
    *,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    """Run Cosmos Predict2 GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.algorithms.grpo_token import TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs, require
    from vrl.distributed.ray import (
        DistributedRolloutConfig,
        build_family_ray_rollout_runtime_inputs,
    )
    from vrl.engine.generation import build_rollout_backend_from_cfg
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.cosmos_predict2 import (
        CosmosPredict2Collector,
        CosmosPredict2CollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.weight_sync import build_runtime_weight_syncer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]
    if not isinstance(grpo_config, GRPOConfig) or isinstance(grpo_config, TokenGRPOConfig):
        raise TypeError(
            f"Cosmos expects algorithm.kind=grpo, got {type(grpo_config).__name__}",
        )

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    # 1. Build policy bundle (backend construction lives in family builder)
    from vrl.models.families.cosmos.builder import (
        build_cosmos_predict2_runtime_bundle_from_cfg,
    )

    bundle = build_cosmos_predict2_runtime_bundle_from_cfg(
        cfg, device, weight_dtype,
    )
    cosmos_model = bundle.policy
    transformer = cosmos_model.transformer

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

    # 3. Reference image (for Video2World conditioning)
    reference_image = None
    ref_image_path = str(require(cfg, "model.reference_image") or "")
    if ref_image_path:
        from PIL import Image

        reference_image = Image.open(ref_image_path).convert("RGB")
        logger.info("Loaded reference image from %s", ref_image_path)
    else:
        logger.warning(
            "No model.reference_image provided. Video2World will use zero "
            "conditioning. This is degenerate — provide a real image for "
            "valid training."
        )

    # 4. Collector + evaluator + algorithm
    collector_config = CosmosPredict2CollectorConfig(
        num_steps=cfg.sampling.num_steps,
        guidance_scale=cfg.sampling.guidance_scale,
        height=cfg.sampling.height,
        width=cfg.sampling.width,
        num_frames=cfg.sampling.num_frames,
        fps=cfg.sampling.fps,
        cfg=cfg.sampling.cfg,
        kl_reward=cfg.algorithm.kl_reward,
        sde_window_size=cfg.rollout.sde.window_size,
        sde_window_range=tuple(cfg.rollout.sde.window_range),
        same_latent=cfg.rollout.same_latent,
    )
    collector = CosmosPredict2Collector(
        cosmos_model, reward_fn, collector_config,
        reference_image=reference_image,
    )
    rollout_backend_config = DistributedRolloutConfig.from_cfg(cfg)
    ray_rollout_inputs = build_family_ray_rollout_runtime_inputs(
        cfg,
        "cosmos",
        weight_dtype=weight_dtype,
        executor_kwargs={"sample_batch_size": collector_config.sample_batch_size},
    )
    collector._runtime = build_rollout_backend_from_cfg(
        cfg,
        runtime=rollout_runtime,
        local_runtime_builder=collector._build_runtime,
        driver_bundle=None if rollout_backend_config.backend == "ray" else bundle,
        runtime_spec=(
            ray_rollout_inputs.runtime_spec if ray_rollout_inputs is not None else None
        ),
        gatherer=ray_rollout_inputs.gatherer if ray_rollout_inputs is not None else None,
    )

    evaluator = FlowMatchingEvaluator(
        bundle.scheduler, noise_level=1.0, sde_type="sde",
    )
    algorithm = GRPO(grpo_config)

    use_lora_kl = cfg.model.use_lora and grpo_config.init_kl_coef > 0
    ref_model = cosmos_model if use_lora_kl else None
    stat_tracker = (
        PerPromptStatTracker(global_std=grpo_config.global_std)
        if cfg.algorithm.per_prompt_stat_tracking else None
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=cosmos_model,
        ref_model=ref_model,
        weight_syncer=build_runtime_weight_syncer(collector._runtime),
        config=trainer_config,
        device=device,
        stat_tracker=stat_tracker,
    )

    # 5. Prompts (manifest = one prompt per line)
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    prompts = [
        line.strip()
        for line in manifest_path.read_text().strip().splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"Manifest contains no prompts: {manifest_path}")

    # Eval prompts: held-out file or first 4 training prompts
    eval_prompts: list[str] = []
    eval_prompts_file = str(require(cfg, "eval.prompts_file") or "")
    if eval_prompts_file and Path(eval_prompts_file).exists():
        eval_prompts = [
            line.strip()
            for line in Path(eval_prompts_file).read_text().strip().splitlines()
            if line.strip()
        ]
    if not eval_prompts:
        eval_prompts = prompts[:4]

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    # --- eval-only mode: paired base-vs-LoRA comparison, no training ---
    if bool(require(cfg, "eval.eval_only")):
        if not str(require(cfg, "model.lora.path") or ""):
            raise ValueError(
                "eval.eval_only=true requires model.lora.path pointing to a "
                "trained LoRA checkpoint. Without it, 'LoRA' scores would "
                "come from a random adapter."
            )
        eval_seeds = int(require(cfg, "eval.seeds"))
        await _run_eval_only(
            transformer=transformer,
            collector=collector,
            eval_prompts=eval_prompts,
            eval_seeds=eval_seeds,
            reference_image=reference_image,
            output_dir=output_dir,
            torch=torch,
        )
        return

    # --- training loop ---
    logger.info(
        "Starting Cosmos Predict2 GRPO — %d epochs, %d prompts, n=%d",
        trainer_config.total_epochs, len(prompts), trainer_config.n,
    )

    if grpo_config.global_std and trainer_config.rollout_batch_size == 1:
        logger.warning(
            "global_std collapses to per-group std with rollout_batch_size=1; "
            "consider rollout.rollout_batch_size>=2",
        )

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{n}" for n in component_names)
    if not csv_path.exists():
        csv_path.write_text(
            "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
            "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
            "adv_zero_rate,group_size,trained_prompt_num,ref_image,"
            + component_cols + "\n"
        )

    ref_image_flag = "1" if reference_image is not None else "0"
    gate_enable = bool(require(cfg, "eval.sanity_gates.enable"))
    clip_thresh = float(require(cfg, "eval.sanity_gates.clip_fraction_threshold"))
    kl_thresh = float(require(cfg, "eval.sanity_gates.approx_kl_threshold"))

    rng = torch.Generator().manual_seed(trainer_config.seed)
    for epoch in range(trainer_config.total_epochs):
        idx = torch.randperm(len(prompts), generator=rng)[
            : trainer_config.rollout_batch_size
        ].tolist()
        prompt_batch = [prompts[i] for i in idx]

        metrics = await trainer.step(prompt_batch)

        # Epoch-0 on-policy sanity check (preserved from legacy script)
        if epoch == 0 and gate_enable:
            logger.info(
                "Epoch 0 sanity check: approx_kl=%.6f clip_fraction=%.4f",
                metrics.approx_kl, metrics.clip_fraction,
            )
            if metrics.clip_fraction > clip_thresh:
                logger.warning(
                    "HIGH clip_fraction at epoch 0 (%.3f) — likely ratio "
                    "mismatch between collect and forward_step. Training may "
                    "not converge.",
                    metrics.clip_fraction,
                )
            if metrics.approx_kl > kl_thresh:
                logger.warning(
                    "HIGH approx_kl at epoch 0 (%.6f) — log-probs from "
                    "collect and forward_step may not match. Check "
                    "_predict_noise_impl consistency.",
                    metrics.approx_kl,
                )

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
                    f"{metrics.trained_prompt_num},{ref_image_flag},"
                    + vals + "\n"
                )

        if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch+1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt_path / "lora_weights")
            logger.info("Saved checkpoint to %s", ckpt_path)

            # Generate eval samples for visual comparison (middle-frame PNGs)
            logger.info("Generating eval samples at epoch %d...", epoch + 1)
            transformer.eval()
            eval_dir = ckpt_path / "eval_samples"
            eval_dir.mkdir(exist_ok=True)
            eval_scores: list[float] = []
            for i, ep in enumerate(eval_prompts):
                with torch.no_grad():
                    eval_batch = await collector.collect(
                        [ep], reference_image=reference_image, seed=i,
                    )
                score = eval_batch.rewards[0].item()
                eval_scores.append(score)
                if eval_batch.videos is not None:
                    _save_middle_frame(
                        eval_batch.videos[0], eval_dir,
                        f"prompt_{i}_score_{score:.2f}.png", torch,
                    )
            transformer.train()

            avg_eval = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
            logger.info(
                "Checkpoint %d | eval_reward=%.4f (%s) | saved to %s",
                epoch + 1, avg_eval,
                ", ".join(f"{s:.2f}" for s in eval_scores), ckpt_path,
            )

    final_path = output_dir / "checkpoint-final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), final_path / "trainer_state.pt")
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(final_path / "lora_weights")
    logger.info("Training complete. Final checkpoint: %s", final_path)


def _save_middle_frame(
    video: Any, out_dir: Path, filename: str, torch: Any,
) -> None:
    """Extract middle frame from video tensor and save as PNG."""
    try:
        from PIL import Image

        vid = video
        if vid.ndim == 4:
            mid = vid.shape[1] // 2
            frame = vid[:, mid, :, :]  # [C, H, W]
        else:
            frame = vid
        frame = (frame * 255).clamp(0, 255).to(torch.uint8)
        frame = frame.cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray(frame)
        img.save(out_dir / filename)
    except Exception:
        logger.debug("Failed to save frame %s", filename)


async def _run_eval_only(
    *,
    transformer: Any,
    collector: Any,
    eval_prompts: list[str],
    eval_seeds: int,
    reference_image: Any,
    output_dir: Path,
    torch: Any,
) -> None:
    """Eval-only mode: paired base-vs-LoRA comparison.

    Same seed for both LoRA and base runs on each (prompt, seed) pair so
    the comparison is paired / apples-to-apples.
    """
    import csv

    eval_dir = output_dir / "eval_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "eval_results.csv"

    rows: list[dict[str, Any]] = []

    # Evaluate LoRA model
    logger.info(
        "Evaluating LoRA model on %d prompts x %d seeds...",
        len(eval_prompts), eval_seeds,
    )
    transformer.eval()
    lora_scores: list[float] = []
    for i, prompt in enumerate(eval_prompts):
        for seed in range(eval_seeds):
            with torch.no_grad():
                batch = await collector.collect(
                    [prompt], reference_image=reference_image, seed=seed,
                )
            score = batch.rewards[0].item()
            lora_scores.append(score)
            if batch.videos is not None:
                _save_middle_frame(
                    batch.videos[0], eval_dir,
                    f"lora_prompt_{i}_seed_{seed}_score_{score:.2f}.png",
                    torch,
                )
            rows.append({
                "prompt": prompt, "seed": seed,
                "lora_score": f"{score:.4f}",
                "base_score": "", "delta": "",
            })

    # Evaluate base model (disable LoRA adapter) — same seeds for paired comparison
    base_scores: list[float] = []
    if hasattr(transformer, "disable_adapter"):
        logger.info(
            "Evaluating base model (LoRA disabled) on %d prompts x %d seeds...",
            len(eval_prompts), eval_seeds,
        )
        with transformer.disable_adapter():
            row_idx = 0
            for i, prompt in enumerate(eval_prompts):
                for seed in range(eval_seeds):
                    with torch.no_grad():
                        batch = await collector.collect(
                            [prompt], reference_image=reference_image, seed=seed,
                        )
                    score = batch.rewards[0].item()
                    base_scores.append(score)
                    if batch.videos is not None:
                        _save_middle_frame(
                            batch.videos[0], eval_dir,
                            f"base_prompt_{i}_seed_{seed}_score_{score:.2f}.png",
                            torch,
                        )
                    rows[row_idx]["base_score"] = f"{score:.4f}"
                    rows[row_idx]["delta"] = (
                        f"{lora_scores[row_idx] - score:.4f}"
                    )
                    row_idx += 1
    else:
        logger.warning(
            "Model does not support disable_adapter — skipping base comparison.",
        )

    transformer.train()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "seed", "lora_score", "base_score", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_lora = sum(lora_scores) / len(lora_scores) if lora_scores else 0.0
    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0.0
    delta = avg_lora - avg_base if base_scores else float("nan")
    logger.info(
        "Eval-only results: lora=%.4f base=%.4f delta=%.4f | %s",
        avg_lora, avg_base, delta, csv_path,
    )
