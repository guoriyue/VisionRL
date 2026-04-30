"""Wan 2.1 Diffusion-DPO training recipe (offline, Pick-a-Pic v2).

Each Wan-DPO entry-point in this directory is a thin wrapper that picks
a YAML config and delegates to ``train_wan_2_1_dpo`` here. Pipeline
construction, LoRA, encoders, training loop, and checkpointing live in
this module.

DPO is offline preference learning — no rollout collection, no algorithm
ABC, just a pure functional loss. We therefore drive the synchronous
``OfflineDPOTrainer`` directly rather than going through ``OnlineTrainer``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _build_encoders(pipeline, num_frames: int, device, dtype):
    """Returns ``(encode_pixels, encode_text)`` closures bound to a WanPipeline.

    ``encode_pixels`` replicates each image to ``num_frames`` along the
    temporal dim before VAE encoding — this lets image-only datasets
    (Pick-a-Pic) train a video model with ``num_frames=1`` (image-style)
    or ``num_frames>1`` (video-style).
    """
    import torch

    vae = pipeline.vae
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )

    def encode_pixels(pixels_2bchw: torch.Tensor) -> torch.Tensor:
        # Replicate to T frames along a new temporal dim.
        x = pixels_2bchw.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        x = x.to(device=device, dtype=vae.dtype)
        latents = vae.encode(x).latent_dist.sample()
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


def train_wan_2_1_dpo(cfg: DictConfig) -> None:
    """Run Wan-family Diffusion-DPO training driven by a merged YAML config."""
    import torch
    from torch.utils.data import DataLoader

    from vrl.models.families.wan_2_1.builder import (
        build_wan_2_1_runtime_bundle_from_cfg,
    )
    from vrl.trainers.offline_dpo import (
        OfflineDPOTrainer,
        OfflineDPOTrainerConfig,
        wan_forward,
    )
    from vrl.trainers.pickapic import collate_preference, load_pickapic

    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.config.loader import (
        build_algorithm_config,
        optional_none,
        require,
        validate_training_config,
    )

    # DPO doesn't go through `build_configs()`, so validate explicitly here
    # to keep the YAML-as-source-of-truth contract (SPRINT patch 3 Phase 6).
    validate_training_config(cfg)

    actor = cfg.actor
    trainer_cfg_yaml = cfg.trainer
    sampling = cfg.sampling
    data_cfg = cfg.data

    dpo_config = build_algorithm_config(cfg)
    if not isinstance(dpo_config, DiffusionDPOConfig):
        raise TypeError(
            f"Wan-DPO expects algorithm.kind=diffusion_dpo, got "
            f"{type(dpo_config).__name__}",
        )

    mixed_precision = str(require(cfg, "actor.mixed_precision"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    # 1. Runtime via family builder (no diffusers import here)
    bundle = build_wan_2_1_runtime_bundle_from_cfg(cfg, device, weight_dtype)
    pipeline = bundle.backend_handle
    transformer = bundle.trainable_modules["transformer"]

    if bool(require(cfg, "actor.gradient_checkpointing")):
        transformer.enable_gradient_checkpointing()

    # 2. Encoders bound to the loaded pipeline
    num_frames = int(sampling.num_frames)
    encode_pixels, encode_text = _build_encoders(
        pipeline, num_frames=num_frames, device=device, dtype=weight_dtype,
    )

    # 3. Data — Pick-a-Pic v2 preference pairs
    # `data.resolution: 0` is YAML's signal "fall back to sampling.height";
    # `require` ensures the key is declared, the `or` keeps that semantic.
    resolution = int(require(cfg, "data.resolution")) or int(sampling.height)
    logger.info(
        "Loading Pick-a-Pic from %s split=%s",
        data_cfg.dataset_name, data_cfg.split,
    )
    ds = load_pickapic(
        split=str(data_cfg.split),
        cache_dir=str(data_cfg.cache_dir) or None,
        max_samples=optional_none(cfg, "data.max_train_samples"),
        resolution=resolution,
        random_crop=bool(require(cfg, "data.random_crop")),
        no_hflip=bool(require(cfg, "data.no_hflip")),
        dataset_name=str(data_cfg.dataset_name),
    )
    train_batch_size = int(require(cfg, "actor.train_batch_size"))
    grad_accum = int(require(cfg, "actor.gradient_accumulation_steps"))
    dataloader = DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=int(require(cfg, "data.dataloader_num_workers")),
        collate_fn=collate_preference,
        drop_last=True,
    )
    logger.info("Loaded %d preference pairs", len(ds))

    # 4. Trainer config — bridge YAML slices to OfflineDPOTrainerConfig
    base_lr = float(actor.optim.lr)
    scale_lr = bool(require(cfg, "actor.scale_lr"))
    effective_bs = train_batch_size * grad_accum
    lr = base_lr * effective_bs if scale_lr else base_lr

    trainer_cfg = OfflineDPOTrainerConfig(
        beta=float(dpo_config.beta),
        sft_weight=float(dpo_config.sft_weight),
        lr=lr,
        scale_lr=False,                       # already scaled above
        max_grad_norm=float(require(cfg, "actor.max_norm")),
        gradient_accumulation_steps=grad_accum,
        prediction_type=str(require(cfg, "actor.prediction_type")),
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
        num_frames=num_frames,
        mixed_precision=mixed_precision,
        use_adafactor=bool(require(cfg, "actor.use_adafactor")),
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

    # 5. Output dir + CSV log + resolved config snapshot
    out_dir = Path(str(trainer_cfg_yaml.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "step,loss,raw_model_loss,raw_ref_loss,model_diff,ref_diff,"
            "implicit_acc,sft_loss,grad_norm\n"
        )

    # 6. Training loop
    max_train_steps = int(require(cfg, "trainer.max_train_steps"))
    checkpointing_steps = int(require(cfg, "trainer.checkpointing_steps"))
    log_interval = int(require(cfg, "trainer.log_interval"))

    logger.info(
        "Starting Wan-1.3B DPO — %d steps, beta=%g, lr=%g, num_frames=%d",
        max_train_steps, trainer_cfg.beta, lr, num_frames,
    )

    step = 0
    data_iter = iter(dataloader)
    while step < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        m = trainer.step(batch)

        if step % log_interval == 0:
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

        if checkpointing_steps > 0 and (step + 1) % checkpointing_steps == 0:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt / "lora_weights")
            logger.info("Saved checkpoint -> %s", ckpt)

        step += 1

    logger.info("DPO training complete.")
