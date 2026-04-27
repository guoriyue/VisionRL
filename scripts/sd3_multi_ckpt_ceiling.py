"""SD3.5 multi-checkpoint OCR ceiling: base + LoRA at multiple epochs.

Same 8 paper-style prompts ("A sign that says X"), same seeds (42+idx),
PaddleOCR scoring matching production OCRReward formula.
Inference uses paper's T=40 (longer than T=10 training, per Denoising Reduction).
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
    "Hello World",
    "Take With Food",
    "Zero Percent APR",
    "First Place Winner",
    "System Override Active",
    "Coffee And Donuts",
    "Sale Ends Sunday",
    "Page 42",
    "Year 2024",
    "Room 305",
    "Gate B12",
    "Highway 101",
    "The Quick Brown Fox",
    "Welcome To Paradise",
    "Knowledge Is Power",
    "Keep Calm Carry On",
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


def _image_score(image_tensor, target, engine):
    """SD3 returns [C, H, W] image. Score once."""
    import torch

    raw = (image_tensor * 255).round().clamp(0, 255).to(torch.uint8)
    if raw.ndim == 3:
        frame = raw.permute(1, 2, 0).cpu().numpy()
    elif raw.ndim == 4:
        frame = raw[0].permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"shape: {raw.shape}")
    return _frame_score(frame, target, engine)


def run_variant(name: str, lora_path: str | None, words, args, ocr, out_dir: Path) -> list[dict]:
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
    for idx, word in enumerate(words):
        prompt = f'A sign that says "{word}"'
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
        score, text = _image_score(image, word, ocr)

        # save image
        raw = (image * 255).round().clamp(0, 255).to(torch.uint8)
        if raw.ndim == 3:
            arr = raw.permute(1, 2, 0).cpu().numpy()
        else:
            arr = raw[0].permute(1, 2, 0).cpu().numpy()
        png = out_dir / f"{name}_{idx:02d}_{word.replace(' ', '_')}.png"
        Image.fromarray(arr).save(png)

        results.append({"idx": idx, "target": word, "score": float(score), "ocr": text, "image": str(png)})
        logger.info("  [%s] %d/%d %s score=%.3f ocr=%r", name, idx + 1, len(words), word, score, text)

    del pipe
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--ckpt-dir", default="outputs/sd3_5_paper_200ep")
    parser.add_argument("--ckpt-epochs", default="25,100,200")
    parser.add_argument("--output-dir", default="outputs/sd3_multi_ckpt_ceiling")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=40, help="paper uses 40 for eval")
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
        all_results[name] = run_variant(name, lora, words, args, ocr, out_dir)
        (out_dir / "partial_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # ---- summary ----
    lines = [
        "=" * 80,
        "SD3.5 OCR ceiling apples-to-apples — same 8 prompts, PaddleOCR, T=40 inference",
        "=" * 80,
        f"{'variant':12s}  {'mean':>7s}  {'median':>7s}  {'max':>5s}  {'nonzero%':>9s}  {'mean(>0)':>9s}",
    ]
    rows = {}
    for name in all_results:
        scores = [r["score"] for r in all_results[name]]
        nz = [s for s in scores if s > 0]
        rows[name] = {"mean": statistics.mean(scores), "median": statistics.median(scores),
                      "max": max(scores), "nz_rate": len(nz) / len(scores),
                      "nz_mean": statistics.mean(nz) if nz else 0}
        s = rows[name]
        lines.append(f"{name:12s}  {s['mean']:>7.4f}  {s['median']:>7.4f}  {s['max']:>5.3f}  {s['nz_rate']:>8.2%}  {s['nz_mean']:>9.4f}")

    if "base" in rows:
        base_m = rows["base"]["mean"]
        lines.append("")
        lines.append("Delta vs base:")
        for name, s in rows.items():
            if name == "base":
                continue
            d = s["mean"] - base_m
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            lines.append(f"  {name:12s}  {arrow} {d:+.4f}  ({d/base_m*100:+.1f}% relative)" if base_m else f"  {name:12s}  {arrow} {d:+.4f}")

    lines.append("")
    lines.append("Per-prompt: best variant for each target")
    for i in range(len(words)):
        per = [(name, all_results[name][i]["score"], all_results[name][i]["ocr"]) for name in all_results]
        winner = max(per, key=lambda x: x[1])
        lines.append(f"  {words[i]:20s}  best={winner[0]}({winner[1]:.2f},{winner[2]!r})  " + " ".join(f"{n}={s:.2f}" for n, s, _ in per))

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("DONE — %s", out_dir / "summary.txt")

    # ---- side-by-side composites (only when exactly 2 variants) ----
    if len(all_results) == 2:
        from PIL import Image, ImageDraw, ImageFont
        sbs_dir = out_dir / "side_by_side"
        sbs_dir.mkdir(exist_ok=True)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_s = font
        names = list(all_results.keys())
        for i in range(len(words)):
            paths = [all_results[n][i]["image"] for n in names]
            scores = [all_results[n][i]["score"] for n in names]
            ocrs = [all_results[n][i]["ocr"] for n in names]
            imgs = [Image.open(p).convert("RGB") for p in paths]
            w, h = imgs[0].size
            pad = 20
            header = 80
            combined = Image.new("RGB", (w * 2 + pad * 3, h + header + pad), (245, 245, 245))
            combined.paste(imgs[0], (pad, header))
            combined.paste(imgs[1], (pad * 2 + w, header))
            d = ImageDraw.Draw(combined)
            delta = scores[1] - scores[0]
            color = (0, 140, 0) if delta > 0 else (200, 0, 0) if delta < 0 else (80, 80, 80)
            d.text((pad, 8), f'#{i}: target = "{words[i]}"', fill=(0, 0, 0), font=font)
            d.text((pad, 36), f"delta = {delta:+.3f}", fill=color, font=font)
            d.text((pad, header - 22), f"{names[0]}: {scores[0]:.2f}  ocr={ocrs[0]!r}", fill=(80, 80, 80), font=font_s)
            d.text((pad * 2 + w, header - 22), f"{names[1]}: {scores[1]:.2f}  ocr={ocrs[1]!r}", fill=(0, 80, 180), font=font_s)
            combined.save(sbs_dir / f"sbs_{i:02d}_{words[i].replace(' ', '_')}.png")
        # grid
        files = sorted(sbs_dir.glob("sbs_*.png"))
        if files:
            first = Image.open(files[0])
            gw, gh = first.size
            cols = 2
            rows = (len(files) + cols - 1) // cols
            grid = Image.new("RGB", (gw * cols, gh * rows), (255, 255, 255))
            for k, p in enumerate(files):
                r, c = divmod(k, cols)
                grid.paste(Image.open(p), (c * gw, r * gh))
            grid.save(sbs_dir / "ALL_prompts_grid.png")
            logger.info("Side-by-side grid: %s", sbs_dir / "ALL_prompts_grid.png")


if __name__ == "__main__":
    main()
