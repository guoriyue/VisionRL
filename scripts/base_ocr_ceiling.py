"""Base Wan 1.3B OCR ceiling test.

Generates videos for paper-style "A sign that says X" prompts using the
base Wan model (no LoRA). Scores each with PaddleOCR + Levenshtein,
matching the production OCRReward (vrl/rewards/ocr.py) so scores are
directly comparable to training reward_mean and the paper's 0.92 target.

Usage:
    python scripts/base_ocr_ceiling.py --num-prompts 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

WORDS = [
    "HELLO", "STOP", "EXIT", "OPEN", "CLOSED", "WELCOME", "CAFE", "BANK",
    "HOTEL", "PIZZA", "COFFEE", "WIFI", "SALE", "WARNING", "DANGER",
    "POLICE", "TAXI", "METRO", "BUS", "PARK", "NORTH", "SOUTH", "EAST",
    "WEST", "ENTER", "PUSH", "PULL", "FIRE", "WATER", "LIGHT", "MUSIC",
    "PEACE", "LOVE", "HOPE", "DREAM", "SPEED", "POWER", "SMART", "QUICK",
    "FRESH", "GREEN", "BLACK", "WHITE", "GOLD", "SILVER", "MOON", "STAR",
    "OCEAN", "RIVER",
]


def _frame_score(frame_np, target: str, engine) -> tuple[float, str]:
    from Levenshtein import distance
    target_norm = target.replace(" ", "").lower()
    target_len = len(target_norm)
    if target_len == 0:
        return 0.0, ""
    try:
        result = engine.ocr(frame_np, cls=False)
        if result and result[0]:
            text_raw = "".join(
                res[1][0] if res[1][1] > 0 else "" for res in result[0]
            )
        else:
            text_raw = ""
    except Exception as e:
        logger.warning("OCR failed on frame: %s", e)
        return 0.0, ""
    text_norm = text_raw.replace(" ", "").lower()
    dist = min(distance(text_norm, target_norm), target_len)
    return 1.0 - dist / target_len, text_raw


def _video_score(video_tensor, target: str, engine, frame_interval: int = 4) -> tuple[float, str, int]:
    """Match OCRReward.score: mean over non-zero per-frame rewards."""
    import numpy as np
    import torch

    raw = (video_tensor * 255).round().clamp(0, 255).to(torch.uint8)
    if raw.ndim == 4 and raw.shape[0] <= 4:
        # [C, T, H, W] → [T, H, W, C]
        video = raw.permute(1, 2, 3, 0).cpu().numpy()
    elif raw.ndim == 4:
        video = raw.permute(0, 2, 3, 1).cpu().numpy()
    else:
        raise ValueError(f"Unexpected video shape: {raw.shape}")

    frames = list(video[::frame_interval])
    rewards: list[float] = []
    best_reward = 0.0
    best_text = ""
    for frame in frames:
        r, t = _frame_score(frame, target, engine)
        if r > 0:
            rewards.append(r)
        if r > best_reward:
            best_reward = r
            best_text = t

    if not rewards:
        return 0.0, best_text, len(frames)
    return sum(rewards) / len(rewards), best_text, len(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--lora-path", default="", help="empty = base model only")
    parser.add_argument("--output-dir", default="outputs/base_ceiling")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = WORDS[: args.num_prompts]
    prompts = [f'A sign that says "{w}"' for w in words]

    logger.info("Loading WanPipeline (base, no LoRA)…")
    import torch
    from diffusers import WanPipeline
    from PIL import Image

    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(args.num_steps, device="cuda")

    if args.lora_path:
        from peft import PeftModel
        logger.info("Applying LoRA from %s", args.lora_path)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path, is_trainable=False)

    logger.info("Loading PaddleOCR…")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

    results = []
    for idx, (word, prompt) in enumerate(zip(words, prompts)):
        logger.info("[%d/%d] target=%r", idx + 1, len(words), word)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        with torch.no_grad():
            out_pipe = pipe(
                prompt=prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                output_type="pt",
            )
        video = out_pipe.frames[0]
        score, best_text, n_frames = _video_score(video, word, ocr)

        # save middle frame for spot-check
        raw = (video * 255).round().clamp(0, 255).to(torch.uint8)
        if raw.ndim == 4 and raw.shape[0] <= 4:
            mid = raw[:, raw.shape[1] // 2].permute(1, 2, 0).cpu().numpy()
        else:
            mid = raw[raw.shape[0] // 2].permute(1, 2, 0).cpu().numpy()
        png = out / f"{idx:03d}_{word}.png"
        Image.fromarray(mid).save(png)

        results.append({
            "idx": idx, "target": word, "prompt": prompt,
            "score": float(score), "best_ocr_text": best_text,
            "n_frames_scored": n_frames, "image": str(png),
        })
        logger.info("  score=%.3f  ocr=%r", score, best_text)

    # ---- summary ----
    scores = [r["score"] for r in results]
    nonzero = [s for s in scores if s > 0]
    import statistics

    summary_lines = [
        "=" * 60,
        f"Base Wan 1.3B OCR ceiling — {len(results)} prompts, paper-style",
        "=" * 60,
        f"mean:                 {statistics.mean(scores):.4f}",
        f"median:               {statistics.median(scores):.4f}",
        f"max:                  {max(scores):.4f}",
        f"non-zero rate:        {len(nonzero)}/{len(scores)} = {len(nonzero)/len(scores):.2%}",
        f"mean (non-zero only): {statistics.mean(nonzero) if nonzero else 0:.4f}",
        "",
        "Top 10 by score:",
    ]
    for r in sorted(results, key=lambda x: -x["score"])[:10]:
        summary_lines.append(
            f"  {r['target']:10s}  score={r['score']:.3f}  ocr={r['best_ocr_text']!r}"
        )

    summary = "\n".join(summary_lines)
    print(summary)
    (out / "summary.txt").write_text(summary + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Wrote summary to %s", out / "summary.txt")


if __name__ == "__main__":
    main()
