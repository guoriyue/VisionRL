"""Wan2.1-1.3B Diffusion-DPO training (single-GPU, diffusers).

Offline DPO over Pick-a-Pic v2 image preference pairs:
    WanPipeline + LoRA   →   OfflineDPOTrainer   →   diffusion_dpo_loss

Pick-a-Pic is image-only; for Wan video model we replicate each image
to ``num_frames`` along the temporal dim before VAE encoding. Use
``num_frames=1`` for the fastest sanity-check loop.

Usage:
    python -m vrl.scripts.wan_2_1.wan_2_1_1_3b_dpo \
        --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --max-train-steps 2000 \
        --beta-dpo 5000 \
        --output-dir outputs/wan_1_3b_dpo
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WanDPOConfig:
    """Defaults track the Diffusion-DPO paper for SDXL, scaled to Wan latents."""

    # --- model ---
    model_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_path: str = ""
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

    # --- DPO ---
    beta_dpo: float = 5000.0
    sft_weight: float = 0.0

    # --- generation shape (sets the "video" the trainer fakes from images) ---
    width: int = 416
    height: int = 240
    num_frames: int = 1                  # 1 for image-style DPO; 33+ for video pairs

    # --- optimizer ---
    lr: float = 1e-8
    scale_lr: bool = True
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    use_adafactor: bool = False

    # --- data ---
    dataset_name: str = "yuvalkirstain/pickapic_v2"
    cache_dir: str = ""
    split: str = "train"
    max_train_samples: int | None = None
    dataloader_num_workers: int = 4
    random_crop: bool = False
    no_hflip: bool = False
    resolution: int = 0   # 0 → derive from height (square)

    # --- training ---
    max_train_steps: int = 2000
    checkpointing_steps: int = 500
    log_interval: int = 10

    # --- output ---
    output_dir: str = "outputs/wan_1_3b_dpo"


# ---------------------------------------------------------------------------
# Pixel & text encoders bound to a loaded WanPipeline
# ---------------------------------------------------------------------------


def _build_encoders(pipeline, num_frames: int, device, dtype):
    """Returns (encode_pixels, encode_text) closures."""
    import torch

    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

    # VAE normalization stats — same as WanT2VDiffusersPolicy.decode_latents
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device, dtype=torch.float32)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device, dtype=torch.float32)
    )

    def encode_pixels(pixels_2bchw: torch.Tensor) -> torch.Tensor:
        """[2B, 3, H, W] images → [2B, C', T', h, w] Wan latents.

        Pixels are already in [-1, 1] (the dataset normalizes via
        ``Normalize([0.5], [0.5])``), matching VAE expectation.
        """
        # Replicate to T frames along a new temporal dim.
        x = pixels_2bchw.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        # VAE encode
        x = x.to(device=device, dtype=vae.dtype)
        latents = vae.encode(x).latent_dist.sample()
        # Normalize like the pipeline does for inference (inverse of decode shift)
        latents = (latents.float() - latents_mean) * latents_std
        return latents.to(dtype)

    @torch.no_grad()
    def encode_text(captions: list[str]) -> torch.Tensor:
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=captions,
            negative_prompt=[""] * len(captions),
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        return prompt_embeds.to(dtype)

    return encode_pixels, encode_text


# ---------------------------------------------------------------------------
# Train entry
# ---------------------------------------------------------------------------


def train(config: WanDPOConfig) -> None:
    import torch
    from torch.utils.data import DataLoader

    from vrl.trainers.pickapic import collate_preference, load_pickapic
    from vrl.trainers.offline_dpo import (
        OfflineDPOTrainer,
        OfflineDPOTrainerConfig,
        wan_forward,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    # 1. Build runtime via the wan_2_1 family builder (no direct diffusers import).
    from vrl.models.families.wan_2_1.builder import build_wan_2_1_runtime_bundle
    from vrl.models.runtime import RuntimeBuildSpec

    lora_cfg = None
    if config.use_lora and not config.lora_path:
        lora_cfg = {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "target_modules": [
                "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
                "to_k", "to_out.0", "to_q", "to_v",
            ],
        }
    spec = RuntimeBuildSpec(
        model_name_or_path=config.model_path,
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="t2v",
        use_lora=bool(config.use_lora),
        lora_path=config.lora_path or None,
        lora_config=lora_cfg,
        scheduler_config=None,  # DPO sets training timesteps directly below
    )
    bundle = build_wan_2_1_runtime_bundle(spec)
    pipeline = bundle.backend_handle
    transformer = bundle.trainable_modules["transformer"]

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 3. Encoders
    encode_pixels, encode_text = _build_encoders(
        pipeline, num_frames=config.num_frames, device=device, dtype=weight_dtype,
    )

    # 4. Data
    resolution = config.resolution or config.height
    logger.info("Loading Pick-a-Pic from %s split=%s", config.dataset_name, config.split)
    ds = load_pickapic(
        split=config.split,
        cache_dir=config.cache_dir or None,
        max_samples=config.max_train_samples,
        resolution=resolution,
        random_crop=config.random_crop,
        no_hflip=config.no_hflip,
        dataset_name=config.dataset_name,
    )
    dataloader = DataLoader(
        ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_preference,
        drop_last=True,
    )
    logger.info("Loaded %d preference pairs", len(ds))

    # 5. Trainer
    effective_bs = config.train_batch_size * config.gradient_accumulation_steps
    lr = config.lr * effective_bs if config.scale_lr else config.lr

    trainer_cfg = OfflineDPOTrainerConfig(
        beta=config.beta_dpo,
        sft_weight=config.sft_weight,
        lr=lr,
        scale_lr=False,                       # already scaled above
        max_grad_norm=config.max_grad_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        prediction_type="flow_matching",
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
        num_frames=config.num_frames,
        mixed_precision=config.mixed_precision,
        use_adafactor=config.use_adafactor,
    )
    pipeline.scheduler.set_timesteps(
        pipeline.scheduler.config.num_train_timesteps, device=device,
    )
    trainer = OfflineDPOTrainer(
        model=transformer,
        ref_model=None,                       # use LoRA disable_adapter for ref
        forward_fn=wan_forward,
        noise_scheduler=pipeline.scheduler,
        encode_pixels=encode_pixels,
        encode_text=encode_text,
        config=trainer_cfg,
        device=device,
    )

    # 6. Output dir + CSV log
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "step,loss,raw_model_loss,raw_ref_loss,model_diff,ref_diff,"
            "implicit_acc,sft_loss,grad_norm\n"
        )

    # 7. Training loop
    logger.info(
        "Starting Wan-1.3B DPO — %d steps, β=%g, lr=%g, num_frames=%d",
        config.max_train_steps, config.beta_dpo, lr, config.num_frames,
    )

    step = 0
    data_iter = iter(dataloader)
    while step < config.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        m = trainer.step(batch)

        if step % config.log_interval == 0:
            logger.info(
                "step %d | loss=%.4f acc=%.3f model_diff=%.4f ref_diff=%.4f "
                "model_mse=%.4f ref_mse=%.4f gn=%.3f",
                step, m.loss, m.implicit_acc, m.model_diff, m.ref_diff,
                m.raw_model_loss, m.raw_ref_loss, m.grad_norm,
            )
            with open(csv_path, "a") as f:
                f.write(
                    f"{step},{m.loss:.6f},{m.raw_model_loss:.6f},{m.raw_ref_loss:.6f},"
                    f"{m.model_diff:.6f},{m.ref_diff:.6f},{m.implicit_acc:.4f},"
                    f"{m.sft_loss:.6f},{m.grad_norm:.4f}\n"
                )

        if config.checkpointing_steps > 0 and (step + 1) % config.checkpointing_steps == 0:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt / "lora_weights")
            logger.info("Saved checkpoint → %s", ckpt)

        step += 1

    logger.info("DPO training complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="Wan 1.3B Diffusion-DPO training")
    p.add_argument("--model-path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--lora-path", type=str, default="")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--mixed-precision", choices=["fp16", "bf16", "no"], default="bf16")
    p.add_argument("--no-gradient-checkpointing", action="store_true")

    p.add_argument("--beta-dpo", type=float, default=5000.0)
    p.add_argument("--sft-weight", type=float, default=0.0)

    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--resolution", type=int, default=0,
                   help="square crop size; 0 → use --height")

    p.add_argument("--lr", type=float, default=1e-8)
    p.add_argument("--no-scale-lr", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--use-adafactor", action="store_true")

    p.add_argument("--dataset-name", type=str, default="yuvalkirstain/pickapic_v2")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--random-crop", action="store_true")
    p.add_argument("--no-hflip", action="store_true")

    p.add_argument("--max-train-steps", type=int, default=2000)
    p.add_argument("--checkpointing-steps", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--output-dir", type=str, default="outputs/wan_1_3b_dpo")

    args = p.parse_args()

    cfg = WanDPOConfig(
        model_path=args.model_path,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_path=args.lora_path,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        beta_dpo=args.beta_dpo,
        sft_weight=args.sft_weight,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        resolution=args.resolution,
        lr=args.lr,
        scale_lr=not args.no_scale_lr,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_batch_size=args.train_batch_size,
        use_adafactor=args.use_adafactor,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        split=args.split,
        max_train_samples=args.max_train_samples,
        dataloader_num_workers=args.dataloader_num_workers,
        random_crop=args.random_crop,
        no_hflip=args.no_hflip,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
    )

    train(cfg)


if __name__ == "__main__":
    main()
