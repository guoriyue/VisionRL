"""Cosmos Predict2-2B GRPO training (single-GPU, diffusers).

Uses diffusers.Cosmos2VideoToWorldPipeline with:
  CosmosDiffusersCollector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Usage:
    python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
        --model-path nvidia/Cosmos-Predict2-2B-Video2World \
        --reward-type aesthetic \
        --prompt-file prompts.txt \
        --reference-image ref.png

Eval-only mode (base vs LoRA comparison):
    python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
        --eval-only --lora-path outputs/cosmos_pred2_2b_grpo/checkpoint-100/lora_weights \
        --reference-image ref.png --eval-prompts eval_prompts.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CosmosPred2Config:
    """Configuration for Cosmos Predict2-2B GRPO training."""

    # Model — HuggingFace model path or local path
    model_path: str = "nvidia/Cosmos-Predict2-2B-Video2World"

    # Generation (Cosmos Predict2 defaults)
    width: int = 1280
    height: int = 704
    num_frames: int = 93
    num_steps: int = 35
    guidance_scale: float = 7.0
    fps: int = 16

    # Reference image for Video2World conditioning
    reference_image: str = ""

    # LoRA
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_path: str = ""

    # Training
    lr: float = 1e-5
    num_epochs: int = 10000
    group_size: int = 4
    num_inner_epochs: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    beta: float = 0.004  # KL loss coefficient
    clip_range: float = 1e-3
    adv_clip_max: float = 5.0
    timestep_fraction: float = 0.99
    gradient_checkpointing: bool = True

    # EMA
    ema: bool = True
    ema_decay: float = 0.9
    ema_update_interval: int = 8

    # Sampling
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    global_std: bool = False
    cfg: bool = True

    # Data
    prompt_file: str = ""
    prompts: list[str] = field(default_factory=list)
    prompts_per_step: int = 1

    # Reward
    reward_type: str = "aesthetic"

    # Eval
    eval_prompts_file: str = ""
    eval_seeds: int = 1
    seed: int = 0
    eval_only: bool = False

    # Debug
    debug_first_step: bool = False

    # Logging
    log_interval: int = 1
    save_interval: int = 100
    output_dir: str = "outputs/cosmos_pred2_2b_grpo"


async def train(config: CosmosPred2Config) -> None:
    """Main training loop."""
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.rollouts.collectors.cosmos import (
        CosmosDiffusersCollector,
        CosmosDiffusersCollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rewards.multi import _register_builtins, get_reward
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Cosmos2 Pipeline (bypass safety checker)
    logger.info("Loading Cosmos2VideoToWorldPipeline from %s", config.model_path)

    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _v2w_mod
    from diffusers import Cosmos2VideoToWorldPipeline

    # Bypass safety checker
    _orig_safety = _v2w_mod.CosmosSafetyChecker

    class _PassthroughSafetyChecker:
        def to(self, device):
            return self

        def check_text_safety(self, prompt):
            return True

        def check_video_safety(self, video):
            return video

    _v2w_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
    try:
        pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        )
    finally:
        _v2w_mod.CosmosSafetyChecker = _orig_safety

    # diffusers from_pretrained disables grad globally — re-enable for training
    torch.set_grad_enabled(True)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(
        device,
        dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )

    # 2. Apply LoRA to the transformer
    if config.use_lora:
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer.to(device)

        from peft import LoraConfig, PeftModel, get_peft_model

        if config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.lora_path,
                is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
            ]
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            pipeline.transformer = get_peft_model(
                pipeline.transformer, lora_config,
            )
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to transformer",
            config.lora_rank, config.lora_alpha,
        )
    else:
        pipeline.transformer.requires_grad_(True)
        pipeline.transformer.to(device)

    transformer = pipeline.transformer

    # Gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 3. Get the scheduler for the evaluator
    pipeline.scheduler.set_timesteps(config.num_steps, device=device)

    # 4. Build reward function
    _register_builtins()
    reward_cls = get_reward(config.reward_type)
    reward_fn = reward_cls(device=str(device))

    # 5. Load reference image (for Video2World conditioning)
    reference_image = None
    if config.reference_image:
        from PIL import Image

        reference_image = Image.open(config.reference_image).convert("RGB")
        logger.info("Loaded reference image from %s", config.reference_image)

    # 6. Wire up 4-layer architecture
    collector_config = CosmosDiffusersCollectorConfig(
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        fps=config.fps,
        cfg=config.cfg,
        kl_reward=config.kl_reward,
        sde_window_size=config.sde_window_size,
        sde_window_range=config.sde_window_range,
        same_latent=config.same_latent,
    )
    from vrl.models.families.cosmos.predict2 import DiffusersCosmosPredict2Executor
    from vrl.models.families.cosmos.variants import CosmosVariant

    cosmos_executor = DiffusersCosmosPredict2Executor(
        variant=CosmosVariant.PREDICT2_VIDEO2WORLD,
        model_size="2B",
    )
    cosmos_executor._pipeline = pipeline
    cosmos_executor._load_modules()

    collector = CosmosDiffusersCollector(
        cosmos_executor, reward_fn, collector_config,
        reference_image=reference_image,
    )

    evaluator = FlowMatchingEvaluator(
        pipeline.scheduler,
        noise_level=1.0,
        sde_type="sde",  # Cosmos2 uses FlowMatchEulerDiscreteScheduler — same as Wan
    )

    grpo_config = GRPOConfig(
        clip_eps=config.clip_range,
        kl_coeff=config.beta,
        adv_clip_max=config.adv_clip_max,
        global_std=config.global_std,
    )
    algorithm = GRPO(grpo_config)

    trainer_config = TrainerConfig(
        lr=config.lr,
        max_grad_norm=config.max_grad_norm,
        num_inner_epochs=config.num_inner_epochs,
        group_size=config.group_size,
        clip_range=config.clip_range,
        adv_clip_max=config.adv_clip_max,
        beta=config.beta,
        mixed_precision=config.mixed_precision,
        ema=config.ema,
        ema_decay=config.ema_decay,
        ema_update_interval=config.ema_update_interval,
        timestep_fraction=config.timestep_fraction,
        cfg=config.cfg,
        debug_first_step=config.debug_first_step,
    )

    # Use transformer itself as ref_model (LoRA disable_adapter for ref)
    ref_model = transformer if config.use_lora and config.beta > 0 else None

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=transformer,
        ref_model=ref_model,
        config=trainer_config,
        device=device,
    )

    # 7. Load prompts
    prompts = list(config.prompts)
    if config.prompt_file and Path(config.prompt_file).exists():
        prompts = Path(config.prompt_file).read_text().strip().splitlines()
    if not prompts:
        prompts = [
            "a car driving through a cityscape at sunset",
            "waves crashing on a rocky shoreline",
            "a drone flying over mountain terrain",
            "a person walking through a park in autumn",
        ]

    # Load eval prompts (held-out or first 4 training prompts)
    eval_prompts: list[str] = []
    if config.eval_prompts_file and Path(config.eval_prompts_file).exists():
        eval_prompts = Path(config.eval_prompts_file).read_text().strip().splitlines()
    if not eval_prompts:
        eval_prompts = prompts[:4]

    # Reference image warning
    if not config.reference_image:
        logger.warning(
            "No --reference-image provided. Video2World will use zero conditioning. "
            "This is degenerate — provide a real image for valid training."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --eval-only mode: compare base vs LoRA on eval prompts
    if config.eval_only:
        if not config.lora_path:
            raise ValueError(
                "--eval-only requires --lora-path pointing to a trained LoRA checkpoint. "
                "Without it, 'LoRA' scores would come from a random adapter."
            )
        await _run_eval_only(
            config, transformer, collector, eval_prompts, output_dir, device, torch,
        )
        return

    # 8. Training loop
    logger.info(
        "Starting Cosmos Predict2-2B GRPO training — %d epochs, %d prompts, group_size=%d",
        config.num_epochs, len(prompts), config.group_size,
    )

    # CSV log for easy plotting
    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
            "clip_fraction,approx_kl,advantage_mean,ref_image\n"
        )

    ref_image_flag = "1" if config.reference_image else "0"

    if config.global_std and config.prompts_per_step == 1:
        logger.warning(
            "global_std collapses to per-group std with a single prompt per step; "
            "consider --prompts-per-step >= 2"
        )

    rng = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.num_epochs):
        n = config.prompts_per_step
        idx = torch.randperm(len(prompts), generator=rng)[:n].tolist()
        prompt_batch = [prompts[i] for i in idx]

        metrics = await trainer.step(prompt_batch)

        # Epoch-0 on-policy sanity check
        if epoch == 0:
            logger.info(
                "Epoch 0 sanity check: approx_kl=%.6f clip_fraction=%.4f",
                metrics.approx_kl, metrics.clip_fraction,
            )
            if metrics.clip_fraction > 0.5:
                logger.warning(
                    "HIGH clip_fraction at epoch 0 (%.3f) — likely ratio mismatch "
                    "between collect and forward_step. Training may not converge.",
                    metrics.clip_fraction,
                )
            if metrics.approx_kl > 0.1:
                logger.warning(
                    "HIGH approx_kl at epoch 0 (%.6f) — log-probs from collect and "
                    "forward_step may not match. Check _predict_noise_impl consistency.",
                    metrics.approx_kl,
                )

        if epoch % config.log_interval == 0:
            logger.info(
                "Epoch %d | loss=%.4f policy_loss=%.4f kl=%.4f "
                "reward=%.4f+/-%.4f clip_frac=%.3f",
                epoch,
                metrics.loss,
                metrics.policy_loss,
                metrics.kl_penalty,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.clip_fraction,
            )
            # Append to CSV
            with open(csv_path, "a") as f:
                f.write(
                    f"{epoch},{metrics.loss:.6f},{metrics.policy_loss:.6f},"
                    f"{metrics.kl_penalty:.6f},{metrics.reward_mean:.4f},"
                    f"{metrics.reward_std:.4f},{metrics.clip_fraction:.4f},"
                    f"{metrics.approx_kl:.6f},{metrics.advantage_mean:.4f},"
                    f"{ref_image_flag}\n"
                )

        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch+1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt_path / "lora_weights")

            # Generate eval samples for visual comparison
            logger.info("Generating eval samples at epoch %d...", epoch + 1)
            transformer.eval()
            eval_dir = ckpt_path / "eval_samples"
            eval_dir.mkdir(exist_ok=True)
            eval_scores = []
            for i, ep in enumerate(eval_prompts):
                with torch.no_grad():
                    eval_batch = await collector.collect(
                        [ep], reference_image=reference_image, seed=i,
                    )
                score = eval_batch.rewards[0].item()
                eval_scores.append(score)
                # Save middle frame as PNG
                if eval_batch.videos is not None:
                    _save_middle_frame(
                        eval_batch.videos[0], eval_dir,
                        f"prompt_{i}_score_{score:.2f}.png", torch,
                    )
            transformer.train()

            avg_eval = sum(eval_scores) / len(eval_scores) if eval_scores else 0
            logger.info(
                "Checkpoint %d | eval_reward=%.4f (%s) | saved to %s",
                epoch + 1,
                avg_eval,
                ", ".join(f"{s:.2f}" for s in eval_scores),
                ckpt_path,
            )

    logger.info("Training complete.")


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
    config: CosmosPred2Config,
    transformer: Any,
    collector: Any,
    eval_prompts: list[str],
    output_dir: Path,
    device: Any,
    torch: Any,
) -> None:
    """Eval-only mode: compare base model vs LoRA on eval prompts.

    Uses the same seed for both LoRA and base runs on each (prompt, seed)
    pair so the comparison is paired / apples-to-apples.
    """
    import csv

    reference_image = None
    if config.reference_image:
        from PIL import Image as PILImage

        reference_image = PILImage.open(config.reference_image).convert("RGB")

    eval_dir = output_dir / "eval_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "eval_results.csv"

    rows: list[dict[str, Any]] = []

    # Evaluate LoRA model
    logger.info("Evaluating LoRA model on %d prompts x %d seeds...", len(eval_prompts), config.eval_seeds)
    transformer.eval()
    lora_scores: list[float] = []
    for i, prompt in enumerate(eval_prompts):
        for seed in range(config.eval_seeds):
            with torch.no_grad():
                batch = await collector.collect(
                    [prompt], reference_image=reference_image, seed=seed,
                )
            score = batch.rewards[0].item()
            lora_scores.append(score)
            if batch.videos is not None:
                _save_middle_frame(
                    batch.videos[0], eval_dir,
                    f"lora_prompt_{i}_seed_{seed}_score_{score:.2f}.png", torch,
                )
            rows.append({
                "prompt": prompt, "seed": seed,
                "lora_score": f"{score:.4f}", "base_score": "", "delta": "",
            })

    # Evaluate base model (disable LoRA adapter) — same seeds for paired comparison
    base_scores: list[float] = []
    if hasattr(transformer, "disable_adapter"):
        logger.info("Evaluating base model (LoRA disabled) on %d prompts x %d seeds...", len(eval_prompts), config.eval_seeds)
        with transformer.disable_adapter():
            row_idx = 0
            for i, prompt in enumerate(eval_prompts):
                for seed in range(config.eval_seeds):
                    with torch.no_grad():
                        batch = await collector.collect(
                            [prompt], reference_image=reference_image, seed=seed,
                        )
                    score = batch.rewards[0].item()
                    base_scores.append(score)
                    if batch.videos is not None:
                        _save_middle_frame(
                            batch.videos[0], eval_dir,
                            f"base_prompt_{i}_seed_{seed}_score_{score:.2f}.png", torch,
                        )
                    rows[row_idx]["base_score"] = f"{score:.4f}"
                    rows[row_idx]["delta"] = f"{lora_scores[row_idx] - score:.4f}"
                    row_idx += 1
    else:
        logger.warning("Model does not support disable_adapter — skipping base comparison.")

    transformer.train()

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "seed", "lora_score", "base_score", "delta"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    avg_lora = sum(lora_scores) / len(lora_scores) if lora_scores else 0
    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
    delta = avg_lora - avg_base if base_scores else float("nan")
    logger.info(
        "Eval-only results: lora=%.4f base=%.4f delta=%.4f | %s",
        avg_lora, avg_base, delta, csv_path,
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Cosmos Predict2-2B GRPO Training (single GPU)",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="nvidia/Cosmos-Predict2-2B-Video2World",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--reference-image", type=str, default="")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/cosmos_pred2_2b_grpo")
    parser.add_argument("--reward-type", type=str, default="aesthetic")
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.004)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--clip-range", type=float, default=1e-3)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument("--eval-prompts", type=str, default="", help="Path to held-out eval prompts file")
    parser.add_argument("--eval-seeds", type=int, default=1, help="Number of seeds per eval prompt")
    parser.add_argument("--eval-only", action="store_true", help="Eval-only mode: compare base vs LoRA")
    parser.add_argument("--debug-first-step", action="store_true", help="Log old vs fresh log-prob diff on first training step")

    args = parser.parse_args()

    config = CosmosPred2Config(
        model_path=args.model_path,
        reference_image=args.reference_image,
        lora_path=args.lora_path,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        reward_type=args.reward_type,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        group_size=args.group_size,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        mixed_precision=args.mixed_precision,
        clip_range=args.clip_range,
        use_lora=not args.no_lora,
        ema=not args.no_ema,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        prompts_per_step=args.prompts_per_step,
        eval_prompts_file=args.eval_prompts,
        eval_seeds=args.eval_seeds,
        eval_only=args.eval_only,
        debug_first_step=args.debug_first_step,
    )

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
