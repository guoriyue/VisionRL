"""Multi-checkpoint OCR ceiling: base + LoRA at multiple epochs, same prompts, same OCR engine.

Apples-to-apples: every variant gets the SAME 49 paper-style prompts,
SAME seeds (42 + idx), and is scored with PaddleOCR using the
production OCRReward formula (mean of non-zero per-frame edit-distance scores).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
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


def _frame_score(frame_np, target: str, engine):
    from Levenshtein import distance
    target_norm = target.replace(" ", "").lower()
    target_len = len(target_norm)
    if target_len == 0:
        return 0.0, ""
    try:
        result = engine.ocr(frame_np, cls=False)
        if result and result[0]:
            text_raw = "".join(res[1][0] if res[1][1] > 0 else "" for res in result[0])
        else:
            text_raw = ""
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return 0.0, ""
    text_norm = text_raw.replace(" ", "").lower()
    dist = min(distance(text_norm, target_norm), target_len)
    return 1.0 - dist / target_len, text_raw


def _video_score(video_tensor, target, engine, frame_interval=4):
    import torch

    raw = (video_tensor * 255).round().clamp(0, 255).to(torch.uint8)
    if raw.ndim == 4 and raw.shape[0] <= 4:
        video = raw.permute(1, 2, 3, 0).cpu().numpy()
    elif raw.ndim == 4:
        video = raw.permute(0, 2, 3, 1).cpu().numpy()
    else:
        raise ValueError(f"shape: {raw.shape}")

    frames = list(video[::frame_interval])
    rewards = []
    best_r, best_t = 0.0, ""
    for frame in frames:
        r, t = _frame_score(frame, target, engine)
        if r > 0:
            rewards.append(r)
        if r > best_r:
            best_r, best_t = r, t
    return (sum(rewards) / len(rewards) if rewards else 0.0, best_t)


def run_variant(name: str, lora_path: str | None, words, args, ocr) -> list[dict]:
    import torch
    from diffusers import WanPipeline

    logger.info("===== Variant: %s =====", name)
    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(args.num_steps, device="cuda")

    if lora_path:
        from peft import PeftModel
        logger.info("Applying LoRA %s", lora_path)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, is_trainable=False)

    results = []
    for idx, word in enumerate(words):
        prompt = f'A sign that says "{word}"'
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        with torch.no_grad():
            out = pipe(
                prompt=prompt, num_frames=args.num_frames,
                height=args.height, width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=gen, output_type="pt",
            )
        score, text = _video_score(out.frames[0], word, ocr)
        results.append({"idx": idx, "target": word, "score": float(score), "ocr": text})
        logger.info("  [%s] %d/%d %s score=%.3f ocr=%r", name, idx + 1, len(words), word, score, text)

    del pipe
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--ckpt-dir", default="outputs/tracker_gradaccum_4p_1000ep")
    parser.add_argument("--ckpt-epochs", default="50,150,300")
    parser.add_argument("--output-dir", default="outputs/multi_ckpt_ceiling")
    parser.add_argument("--num-prompts", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = WORDS[: args.num_prompts]

    logger.info("Loading PaddleOCR…")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

    variants = [("base", None)]
    for ep in args.ckpt_epochs.split(","):
        ep = ep.strip()
        if ep:
            variants.append((f"ckpt-{ep}", str(Path(args.ckpt_dir) / f"checkpoint-{ep}/lora_weights")))

    all_results = {}
    for name, lora in variants:
        all_results[name] = run_variant(name, lora, words, args, ocr)
        # save partial in case of crash
        (out_dir / "partial_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # ---- summary ----
    lines = [
        "=" * 72,
        "Apples-to-apples OCR ceiling — same 49 paper-style prompts, PaddleOCR, seed=42+idx",
        "=" * 72,
        f"{'variant':12s}  {'mean':>7s}  {'median':>7s}  {'max':>5s}  {'nonzero%':>9s}  {'mean(>0)':>9s}",
    ]
    summary_rows = {}
    for name in all_results:
        scores = [r["score"] for r in all_results[name]]
        nonzero = [s for s in scores if s > 0]
        m = statistics.mean(scores)
        med = statistics.median(scores)
        mx = max(scores)
        nz_rate = len(nonzero) / len(scores)
        nz_mean = statistics.mean(nonzero) if nonzero else 0
        summary_rows[name] = {"mean": m, "median": med, "max": mx, "nz_rate": nz_rate, "nz_mean": nz_mean}
        lines.append(f"{name:12s}  {m:>7.4f}  {med:>7.4f}  {mx:>5.3f}  {nz_rate:>8.2%}  {nz_mean:>9.4f}")

    # delta vs base
    if "base" in summary_rows:
        base_m = summary_rows["base"]["mean"]
        lines.append("")
        lines.append("Delta vs base:")
        for name, s in summary_rows.items():
            if name == "base":
                continue
            d = s["mean"] - base_m
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            lines.append(f"  {name:12s}  {arrow} {d:+.4f}  ({d/base_m*100:+.1f}% relative)")

    # per-prompt comparison: which prompts each variant won
    lines.append("")
    lines.append("Per-prompt: best variant for each target")
    for i in range(len(words)):
        per = [(name, all_results[name][i]["score"]) for name in all_results]
        winner = max(per, key=lambda x: x[1])
        lines.append(f"  {words[i]:10s}  best={winner[0]}({winner[1]:.2f})  " + " ".join(f"{n}={s:.2f}" for n, s in per))

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("DONE — summary at %s", out_dir / "summary.txt")


if __name__ == "__main__":
    main()
