"""Wan2.1-1.3B GRPO with multi-reward + flow_grpo pickscore prompts.

Sibling to ``wan2_1_1_3b_grpo.py`` — kept as a *new* file (not an
extension of v1) so the previous single-reward experiments stay
reproducible. This script fixes the three failure modes diagnosed in
``outputs/wan_1_3b_grpo_v2`` (post-mortem in conversation 2026-04-18):

  1. Reward hacking — swap single ``aesthetic`` for a weighted
     combination (aesthetic + pickscore + clipscore). All three are
     prompt-aware except aesthetic, whose weight we keep small.
  2. ``approx_kl ≈ 0`` — tune inner_epochs / lr / clip_range so the
     LoRA adapter actually moves.
  3. Prompt distribution — load 15k SFW Pick-a-Pic prompts shipped
     with the flow_grpo repo instead of the built-in 8-prompt demo
     set.

Usage:
    python -m vrl.scripts.wan.wan2_1_1_3b_multi_reward_grpo \\
        --output-dir outputs/wan_1_3b_multi_v1 \\
        --num-epochs 500

Held-out eval is run against the first ``eval_prompt_count`` prompts
from ``drawbench/test.txt`` (disjoint from training set) every
``save_interval`` epochs — this is the real metric to watch.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"
DEFAULT_TRAIN_PROMPTS = DATASETS_ROOT / "pickscore_sfw" / "train.txt"
DEFAULT_EVAL_PROMPTS = DATASETS_ROOT / "drawbench" / "test.txt"


@dataclass
class Wan1_3BMultiRewardConfig:
    """Config for the multi-reward + tuned-hyperparam Wan 1.3B recipe."""

    # Model
    model_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Generation (same as v1 defaults)
    width: int = 416
    height: int = 240
    num_frames: int = 33
    num_steps: int = 20
    guidance_scale: float = 4.5

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_path: str = ""

    # ---- Training: tuned for a non-zero approx_kl ----
    # v2 had approx_kl ≈ 0 with lr=1e-4, num_inner_epochs=1, clip_range=1e-4.
    # Three changes in lockstep:
    #   * num_inner_epochs 1 → 4  (more PPO updates per rollout)
    #   * lr                1e-4 → 3e-4
    #   * clip_range       1e-4 → 1e-3  (was absurdly small; 1e-3 aligned w/ trl defaults)
    lr: float = 3e-4
    num_epochs: int = 500
    group_size: int = 4
    num_inner_epochs: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    beta: float = 0.004
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
    global_std: bool = True
    cfg: bool = True

    # ---- Data ----
    # Training: 15k SFW prompts from flow_grpo (~5 MB file).
    prompt_file: str = str(DEFAULT_TRAIN_PROMPTS)
    max_train_prompts: int = 0  # 0 = use all
    prompts_per_step: int = 4

    # Held-out eval — drawbench test set, disjoint from pickscore_sfw.
    eval_prompt_file: str = str(DEFAULT_EVAL_PROMPTS)
    eval_prompt_count: int = 8           # small, deterministic, for fast per-ckpt signal
    eval_seed_base: int = 42

    # ---- Multi-reward (the headline change vs v1) ----
    # Keep aesthetic weight SMALL — it's prompt-agnostic and drives hacking.
    # PickScore + CLIPScore are prompt-aware; they dominate the objective.
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "aesthetic": 0.3,
        "pickscore": 0.5,
        "clipscore": 0.2,
    })

    # Logging
    log_interval: int = 1
    save_interval: int = 50
    output_dir: str = "outputs/wan_1_3b_multi_v1"
    seed: int = 0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _load_prompts(path: str, cap: int | None = None) -> list[str]:
    """Read a newline-delimited prompt file, skip empty lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"prompt file not found: {p}. Clone flow_grpo to "
            "~/Desktop/flow_grpo or override --prompt-file."
        )
    prompts = [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"prompt file {p} is empty")
    if cap:
        prompts = prompts[:cap]
    return prompts


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


async def train(config: Wan1_3BMultiRewardConfig) -> None:
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.wan_diffusers import (
        WanDiffusersCollector,
        WanDiffusersCollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Pipeline --------------------------------------------------------
    logger.info("Loading WanPipeline from %s", config.model_path)
    from diffusers import WanPipeline

    pipeline = WanPipeline.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(
        device,
        dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )

    # 2. LoRA -----------------------------------------------------------
    if config.use_lora:
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer.to(device)

        from peft import LoraConfig, PeftModel, get_peft_model

        if config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.lora_path, is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            target_modules = [
                "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
                "to_k", "to_out.0", "to_q", "to_v",
            ]
            lora_cfg = LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian", target_modules=target_modules,
            )
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_cfg)
        logger.info(
            "LoRA applied: r=%d, alpha=%d", config.lora_rank, config.lora_alpha,
        )
    else:
        pipeline.transformer.requires_grad_(True)
        pipeline.transformer.to(device)

    transformer = pipeline.transformer
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    pipeline.scheduler.set_timesteps(config.num_steps, device=device)

    # 3. Multi-reward ---------------------------------------------------
    logger.info("Building multi-reward: %s", config.reward_weights)
    reward_fn = MultiReward.from_dict(config.reward_weights, device=str(device))

    # 4. Wiring ---------------------------------------------------------
    collector_cfg = WanDiffusersCollectorConfig(
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        height=config.height, width=config.width,
        num_frames=config.num_frames,
        cfg=config.cfg,
        kl_reward=config.kl_reward,
        sde_window_size=config.sde_window_size,
        sde_window_range=config.sde_window_range,
        same_latent=config.same_latent,
    )
    wan_model = DiffusersWanT2VModel(pipeline=pipeline, device=device)
    collector = WanDiffusersCollector(wan_model, reward_fn, collector_cfg)
    evaluator = FlowMatchingEvaluator(
        pipeline.scheduler, noise_level=1.0, sde_type="sde",
    )

    grpo_cfg = GRPOConfig(
        clip_eps=config.clip_range,
        kl_coeff=config.beta,
        adv_clip_max=config.adv_clip_max,
        global_std=config.global_std,
    )
    algorithm = GRPO(grpo_cfg)

    trainer_cfg = TrainerConfig(
        lr=config.lr,
        max_grad_norm=config.max_grad_norm,
        num_inner_epochs=config.num_inner_epochs,
        group_size=config.group_size,
        clip_range=config.clip_range,
        adv_clip_max=config.adv_clip_max,
        beta=config.beta,
        mixed_precision=config.mixed_precision,
        ema=config.ema, ema_decay=config.ema_decay,
        ema_update_interval=config.ema_update_interval,
        timestep_fraction=config.timestep_fraction,
        cfg=config.cfg,
    )
    ref_model = transformer if config.use_lora and config.beta > 0 else None
    trainer = OnlineTrainer(
        algorithm=algorithm, collector=collector, evaluator=evaluator,
        model=transformer, ref_model=ref_model,
        config=trainer_cfg, device=device,
    )

    # 5. Prompts --------------------------------------------------------
    train_prompts = _load_prompts(
        config.prompt_file,
        cap=config.max_train_prompts or None,
    )
    eval_prompts = _load_prompts(
        config.eval_prompt_file, cap=config.eval_prompt_count,
    )
    logger.info(
        "Loaded %d train prompts from %s; %d held-out eval prompts from %s",
        len(train_prompts), config.prompt_file,
        len(eval_prompts), config.eval_prompt_file,
    )

    # 6. IO -------------------------------------------------------------
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    # Extra per-component columns — critical for spotting reward hacking
    # (e.g. aesthetic climbs while pickscore flatlines).
    reward_cols = [f"r_{k}" for k in config.reward_weights.keys()]
    if not csv_path.exists():
        header = [
            "epoch", "loss", "policy_loss", "kl_penalty",
            "reward_mean", "reward_std",
            "clip_fraction", "approx_kl", "advantage_mean",
            *reward_cols,
        ]
        csv_path.write_text(",".join(header) + "\n")

    eval_csv_path = out_dir / "eval_metrics.csv"
    if not eval_csv_path.exists():
        eval_csv_path.write_text(
            "epoch,eval_reward_mean,"
            + ",".join(f"eval_r_{k}" for k in config.reward_weights.keys())
            + "\n"
        )

    # 7. Train loop ----------------------------------------------------
    logger.info(
        "Starting Wan 1.3B multi-reward GRPO — %d epochs, "
        "num_inner_epochs=%d, lr=%g, clip_range=%g",
        config.num_epochs, config.num_inner_epochs, config.lr, config.clip_range,
    )

    # i.i.d. sampling — see OCR script for rationale (GRPO per-prompt stat
    # tracking assumes i.i.d. draws; cyclic traversal biases the advantage).
    rng = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.num_epochs):
        n = config.prompts_per_step
        idx = torch.randperm(len(train_prompts), generator=rng)[:n].tolist()
        prompt_batch = [train_prompts[i] for i in idx]

        metrics = await trainer.step(prompt_batch)

        # Pull per-component reward means from the MultiReward's latest call.
        per_component = {
            k: (sum(v) / len(v)) if v else float("nan")
            for k, v in reward_fn.last_components.items()
        }

        if epoch % config.log_interval == 0:
            component_str = " ".join(f"{k}={v:.3f}" for k, v in per_component.items())
            logger.info(
                "Epoch %d | loss=%.4f kl=%.4f reward=%.3f±%.3f | %s | "
                "approx_kl=%.5f clip_frac=%.2f",
                epoch, metrics.loss, metrics.kl_penalty,
                metrics.reward_mean, metrics.reward_std,
                component_str,
                metrics.approx_kl, metrics.clip_fraction,
            )
            row = [
                epoch, f"{metrics.loss:.6f}", f"{metrics.policy_loss:.6f}",
                f"{metrics.kl_penalty:.6f}", f"{metrics.reward_mean:.4f}",
                f"{metrics.reward_std:.4f}", f"{metrics.clip_fraction:.4f}",
                f"{metrics.approx_kl:.6f}", f"{metrics.advantage_mean:.4f}",
            ] + [f"{per_component.get(k, float('nan')):.4f}" for k in config.reward_weights.keys()]
            with open(csv_path, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")

        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            await _save_checkpoint_and_eval(
                epoch=epoch + 1, out_dir=out_dir,
                transformer=transformer, trainer=trainer,
                collector=collector, reward_fn=reward_fn,
                eval_prompts=eval_prompts, eval_seed_base=config.eval_seed_base,
                eval_csv_path=eval_csv_path,
                component_keys=list(config.reward_weights.keys()),
            )

    logger.info("Training complete — metrics at %s", csv_path)


async def _save_checkpoint_and_eval(
    *, epoch, out_dir, transformer, trainer, collector, reward_fn,
    eval_prompts, eval_seed_base, eval_csv_path, component_keys,
) -> None:
    """Save LoRA + trainer state, then run held-out eval on drawbench."""
    import torch

    ckpt_dir = out_dir / f"checkpoint-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), ckpt_dir / "trainer_state.pt")
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(ckpt_dir / "lora_weights")

    logger.info("Running held-out eval at epoch %d on %d prompts…",
                epoch, len(eval_prompts))
    transformer.eval()
    eval_dir = ckpt_dir / "eval_samples"
    eval_dir.mkdir(exist_ok=True)

    total_reward: list[float] = []
    component_totals: dict[str, list[float]] = {k: [] for k in component_keys}

    for i, ep in enumerate(eval_prompts):
        with torch.no_grad():
            eval_batch = await collector.collect([ep], seed=eval_seed_base + i)
        score = eval_batch.rewards[0].item()
        total_reward.append(score)
        for k, vals in reward_fn.last_components.items():
            if vals:
                component_totals.setdefault(k, []).append(vals[0])

        # Save the middle frame for visual inspection.
        if eval_batch.videos is not None:
            vid = eval_batch.videos[0]
            if vid.ndim == 4:
                mid = vid.shape[1] // 2
                frame = vid[:, mid, :, :]
            else:
                frame = vid
            frame = (frame * 255).clamp(0, 255).to(torch.uint8)
            frame = frame.cpu().permute(1, 2, 0).numpy()
            from PIL import Image
            Image.fromarray(frame).save(
                eval_dir / f"prompt_{i}_score_{score:.2f}.png"
            )
    transformer.train()

    mean_reward = sum(total_reward) / len(total_reward) if total_reward else 0.0
    component_means = {
        k: (sum(v) / len(v)) if v else float("nan")
        for k, v in component_totals.items()
    }
    logger.info(
        "Eval @ epoch %d | mean_reward=%.4f | %s",
        epoch, mean_reward,
        " ".join(f"{k}={component_means.get(k, float('nan')):.3f}" for k in component_keys),
    )

    with open(eval_csv_path, "a") as f:
        row = [str(epoch), f"{mean_reward:.4f}"] + [
            f"{component_means.get(k, float('nan')):.4f}" for k in component_keys
        ]
        f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_reward_weights(s: str) -> dict[str, float]:
    """Parse ``"aesthetic:0.3,pickscore:0.5,clipscore:0.2"`` into a dict."""
    out: dict[str, float] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--reward-weights entry {chunk!r} missing ':'"
            )
        name, weight = chunk.split(":", 1)
        out[name.strip()] = float(weight)
    if not out:
        raise argparse.ArgumentTypeError("--reward-weights produced empty dict")
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", default=Wan1_3BMultiRewardConfig.model_path)
    p.add_argument("--output-dir", default=Wan1_3BMultiRewardConfig.output_dir)
    p.add_argument("--prompt-file", default=Wan1_3BMultiRewardConfig.prompt_file)
    p.add_argument("--eval-prompt-file", default=Wan1_3BMultiRewardConfig.eval_prompt_file)
    p.add_argument("--eval-prompt-count", type=int, default=Wan1_3BMultiRewardConfig.eval_prompt_count)
    p.add_argument("--max-train-prompts", type=int, default=0,
                   help="Cap training prompt list (0 = use all). Useful for fast smoke.")
    p.add_argument(
        "--reward-weights", type=_parse_reward_weights,
        default="aesthetic:0.3,pickscore:0.5,clipscore:0.2",
        help="Comma-separated name:weight pairs. Known names: aesthetic, pickscore, clipscore, ocr.",
    )
    p.add_argument("--num-epochs", type=int, default=Wan1_3BMultiRewardConfig.num_epochs)
    p.add_argument("--lr", type=float, default=Wan1_3BMultiRewardConfig.lr)
    p.add_argument("--clip-range", type=float, default=Wan1_3BMultiRewardConfig.clip_range)
    p.add_argument("--num-inner-epochs", type=int, default=Wan1_3BMultiRewardConfig.num_inner_epochs)
    p.add_argument("--group-size", type=int, default=Wan1_3BMultiRewardConfig.group_size)
    p.add_argument("--prompts-per-step", type=int, default=Wan1_3BMultiRewardConfig.prompts_per_step)
    p.add_argument("--save-interval", type=int, default=Wan1_3BMultiRewardConfig.save_interval)
    p.add_argument("--seed", type=int, default=Wan1_3BMultiRewardConfig.seed)
    p.add_argument("--lora-rank", type=int, default=Wan1_3BMultiRewardConfig.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=Wan1_3BMultiRewardConfig.lora_alpha)
    p.add_argument("--lora-path", default=Wan1_3BMultiRewardConfig.lora_path)
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _build_argparser().parse_args()

    # Guardrail: MultiReward with no prompt-aware components will hack again.
    reward_weights = args.reward_weights
    prompt_aware = {"pickscore", "clipscore", "ocr"}
    if not (set(reward_weights) & prompt_aware):
        raise SystemExit(
            "reward_weights contains only prompt-agnostic rewards "
            f"({sorted(reward_weights)}). That is exactly the setup that "
            "produced the reward-hacking failure in outputs/wan_1_3b_grpo_v2. "
            "Include at least one of pickscore / clipscore / ocr."
        )

    cfg = Wan1_3BMultiRewardConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        eval_prompt_file=args.eval_prompt_file,
        eval_prompt_count=args.eval_prompt_count,
        max_train_prompts=args.max_train_prompts,
        reward_weights=reward_weights,
        num_epochs=args.num_epochs,
        lr=args.lr,
        clip_range=args.clip_range,
        num_inner_epochs=args.num_inner_epochs,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        save_interval=args.save_interval,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_path=args.lora_path,
    )

    asyncio.run(train(cfg))


if __name__ == "__main__":
    main()
