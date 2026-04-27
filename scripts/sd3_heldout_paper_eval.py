"""SD3.5 held-out eval on PAPER-STYLE prompts (datasets/ocr/test.txt).

Replaces the toy 'A sign that says "X"' eval with the real held-out
distribution: long, quoted, scene-context prompts the model never saw
during training. This is the apples-to-apples test for whether RL
generalizes beyond the train distribution.

Same seed scheme (42+idx) across variants → noise-matched comparison.
T=40 inference (paper).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _image_score(image_tensor, target, engine):
    import torch
    raw = (image_tensor * 255).round().clamp(0, 255).to(torch.uint8)
    if raw.ndim == 3:
        frame = raw.permute(1, 2, 0).cpu().numpy()
    elif raw.ndim == 4:
        frame = raw[0].permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"shape: {raw.shape}")
    return _frame_score(frame, target, engine)


def load_heldout(path: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Read paper-style prompts; return [(prompt, target), ...]."""
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    out = []
    for line in lines:
        parts = line.split('"')
        if len(parts) >= 3 and parts[1].strip():
            out.append((line, parts[1].strip()))
        if len(out) >= n:
            break
    return out


def run_variant(name, lora_path, items, args, ocr, out_dir):
    import torch
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image

    logger.info("===== Variant: %s =====", name)
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    if lora_path:
        from peft import PeftModel
        logger.info("Applying LoRA %s", lora_path)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, is_trainable=False)

    results = []
    for idx, (prompt, target) in enumerate(items):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                height=args.height, width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=gen, output_type="pt",
            )
        image = out.images[0]
        score, ocr_text = _image_score(image, target, ocr)

        raw = (image * 255).round().clamp(0, 255).to(torch.uint8)
        if raw.ndim == 3:
            arr = raw.permute(1, 2, 0).cpu().numpy()
        else:
            arr = raw[0].permute(1, 2, 0).cpu().numpy()
        safe_target = target.replace(" ", "_").replace("/", "_")[:30]
        png = out_dir / f"{name}_{idx:02d}_{safe_target}.png"
        Image.fromarray(arr).save(png)

        results.append({
            "idx": idx,
            "target": target,
            "prompt": prompt,
            "score": float(score),
            "ocr": ocr_text,
            "image": str(png),
        })
        logger.info("  [%s] %d/%d target=%r score=%.3f ocr=%r",
                    name, idx + 1, len(items), target, score, ocr_text)

    del pipe
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--ckpt-dir", default="outputs/sd3_5_paper_200ep")
    parser.add_argument("--ckpt-epochs", default="25,100,200")
    parser.add_argument("--heldout-path", default="datasets/ocr/test.txt")
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--output-dir", default="outputs/sd3_heldout_paper_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_heldout(args.heldout_path, args.num_prompts, args.shuffle_seed)
    logger.info("Loaded %d held-out prompts (seed=%d) from %s",
                len(items), args.shuffle_seed, args.heldout_path)
    (out_dir / "prompts.json").write_text(
        json.dumps([{"prompt": p, "target": t} for p, t in items],
                   indent=2, ensure_ascii=False))

    logger.info("Loading PaddleOCR…")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

    variants = [("base", None)]
    for ep in args.ckpt_epochs.split(","):
        ep = ep.strip()
        if ep:
            variants.append((f"ckpt-{ep}",
                             str(Path(args.ckpt_dir) / f"checkpoint-{ep}/lora_weights")))

    all_results = {}
    for name, lora in variants:
        all_results[name] = run_variant(name, lora, items, args, ocr, out_dir)
        (out_dir / "partial_results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False))

    # ---- summary ----
    lines = [
        "=" * 80,
        f"SD3.5 OCR HELD-OUT eval on PAPER-style prompts ({args.num_prompts} from test.txt, T={args.num_steps})",
        "=" * 80,
        f"{'variant':12s}  {'mean':>7s}  {'median':>7s}  {'max':>5s}  {'nonzero%':>9s}  {'mean(>0)':>9s}",
    ]
    rows = {}
    for name in all_results:
        scores = [r["score"] for r in all_results[name]]
        nz = [s for s in scores if s > 0]
        rows[name] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "max": max(scores),
            "nz_rate": len(nz) / len(scores),
            "nz_mean": statistics.mean(nz) if nz else 0,
        }
        s = rows[name]
        lines.append(
            f"{name:12s}  {s['mean']:>7.4f}  {s['median']:>7.4f}  "
            f"{s['max']:>5.3f}  {s['nz_rate']:>8.2%}  {s['nz_mean']:>9.4f}"
        )

    if "base" in rows:
        base_m = rows["base"]["mean"]
        lines.append("")
        lines.append("Delta vs base:")
        for name, s in rows.items():
            if name == "base":
                continue
            d = s["mean"] - base_m
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            rel = f"({d/base_m*100:+.1f}% relative)" if base_m else ""
            lines.append(f"  {name:12s}  {arrow} {d:+.4f}  {rel}")

    lines.append("")
    lines.append("Per-prompt breakdown:")
    for i in range(len(items)):
        target = items[i][1]
        per = [(n, all_results[n][i]["score"], all_results[n][i]["ocr"])
               for n in all_results]
        winner = max(per, key=lambda x: x[1])
        scores_str = " ".join(f"{n}={s:.2f}" for n, s, _ in per)
        lines.append(f"  [{i:2d}] {target!r:30s}  best={winner[0]}({winner[1]:.2f})  {scores_str}")

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    (out_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("DONE — %s", out_dir / "summary.txt")


if __name__ == "__main__":
    main()
