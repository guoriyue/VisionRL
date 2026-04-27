"""Noise-floor measurement: fixed checkpoint, repeat rollouts across seeds.

For N prompts, call collector.collect() K times (each with group_size=G and
a distinct rollout seed), yielding N*K groups = N*K*G samples. Logs each
sample's reward + per-component r_* so we can decompose total variance into:
  * within-group std (fixed prompt, fixed seed stream, parallel samples)
  * between-group same-prompt std (fixed prompt, different seeds)
  * between-prompt mean std (prompt difficulty distribution)

No gradient updates. Same pipeline/LoRA/reward wiring as the training script.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "datasets" / "ocr" / "train.txt"
)


async def run_eval(args: argparse.Namespace) -> None:
    import torch
    from diffusers import WanPipeline
    from peft import LoraConfig, PeftModel, get_peft_model

    from vrl.models.families.wan2_1.diffusers_t2v import DiffusersWanT2VModel
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collectors.wan2_1 import (
        Wan21Collector,
        Wan21CollectorConfig,
    )
    from vrl.trainers.data import load_prompt_manifest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading WanPipeline from %s", args.model_path)
    pipeline = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=torch.bfloat16)

    pipeline.transformer.requires_grad_(False)
    pipeline.transformer.to(device)

    if args.lora_path:
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, args.lora_path, is_trainable=False,
        )
        pipeline.transformer.set_adapter("default")
        logger.info("Loaded LoRA checkpoint from %s", args.lora_path)
    else:
        target_modules = [
            "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
            "to_k", "to_out.0", "to_q", "to_v",
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
        logger.info(
            "Initialized fresh gaussian LoRA (rank=%d, alpha=%d) — baseline/untrained",
            args.lora_rank, args.lora_alpha,
        )

    pipeline.scheduler.set_timesteps(args.num_steps, device=device)

    reward_fn = MultiReward.from_dict({"ocr": 1.0}, device=str(device))

    # Per-sample component tracker: collector invokes score() per sample and
    # last_components gets overwritten. Wrap score() to accumulate across calls.
    batch_components: dict[str, list[float]] = {}
    orig_score = reward_fn.score

    async def tracked_score(rollout):  # type: ignore[no-untyped-def]
        r = await orig_score(rollout)
        for name, vals in reward_fn.last_components.items():
            batch_components.setdefault(name, []).extend(vals)
        return r

    reward_fn.score = tracked_score  # type: ignore[method-assign]

    collector_config = Wan21CollectorConfig(
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg=True,
        kl_reward=0.0,
    )
    wan_model = DiffusersWanT2VModel(pipeline=pipeline, device=device)
    collector = Wan21Collector(wan_model, reward_fn, collector_config)

    examples = load_prompt_manifest(args.manifest)
    rng = torch.Generator().manual_seed(args.prompt_seed)
    prompt_idx = torch.randperm(len(examples), generator=rng)[: args.n_prompts].tolist()
    selected = [examples[i] for i in prompt_idx]

    logger.info("Selected %d prompts (seed=%d):", len(selected), args.prompt_seed)
    for i, ex in enumerate(selected):
        logger.info("  [%d] idx=%d target=%r", i, prompt_idx[i], ex.target_text)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"

    with csv_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "prompt_orig_idx", "target_text",
            "group_idx", "rollout_seed", "sample_idx",
            "reward", "r_ocr",
        ])

        for p_i, example in enumerate(selected):
            for g_i in range(args.n_seeds):
                seed = args.seed_base + p_i * 1000 + g_i
                batch_components.clear()
                batch = await collector.collect(
                    [example.prompt],
                    seed=seed,
                    group_size=args.group_size,
                    target_text=example.target_text,
                    references=example.references,
                    task_type=example.task_type,
                )
                rewards = batch.rewards.detach().cpu().tolist()
                r_ocr = list(batch_components.get("ocr", []))

                logger.info(
                    "p[%d] g[%d] seed=%d | reward=%s | r_ocr=%s",
                    p_i, g_i, seed,
                    [f"{r:.3f}" for r in rewards],
                    [f"{r:.3f}" for r in r_ocr],
                )
                for s_i, r in enumerate(rewards):
                    ro = r_ocr[s_i] if s_i < len(r_ocr) else float("nan")
                    writer.writerow([
                        p_i, prompt_idx[p_i], example.target_text,
                        g_i, seed, s_i, f"{r:.4f}", f"{ro:.4f}",
                    ])
                f.flush()

    logger.info("Done. Samples CSV at %s", csv_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Noise-floor eval (no training).")
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument(
        "--lora-path", default="",
        help="LoRA checkpoint dir. Empty = fresh gaussian-init LoRA (baseline).",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="outputs/noise_floor")
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
