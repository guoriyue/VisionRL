"""A/B inference: Wan2.1-1.3B base vs trained LoRA, same seed, same prompts.

Generates one video per (variant, prompt) pair, dumps the middle frame and
runs OCR scoring to produce a side-by-side quantitative comparison.

Usage:
    python -m vrl.scripts.wan.wan_t2v_ab_infer \
        --lora-path outputs/wan_1_3b_ocr_50ep_save/checkpoint-final/lora_weights \
        --manifest datasets/ocr/train.txt \
        --output-dir outputs/ab_infer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vrl.trainers.data import load_prompt_manifest

logger = logging.getLogger(__name__)


def _load_prompts(manifest_path: Path) -> list[tuple[str, str]]:
    """Return list of (prompt, target_text) pairs."""
    return [(ex.prompt, ex.target_text) for ex in load_prompt_manifest(manifest_path)]


def _generate(pipeline, prompt: str, seed: int, cfg: dict) -> "torch.Tensor":
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        out = pipeline(
            prompt=prompt,
            height=cfg["height"],
            width=cfg["width"],
            num_frames=cfg["num_frames"],
            num_inference_steps=cfg["num_steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=generator,
            output_type="pt",
        )
    # out.frames is typically [B, T, C, H, W] or [B, C, T, H, W]
    return out.frames[0]


def _middle_frame_uint8(video_tensor: "torch.Tensor") -> "np.ndarray":
    """Extract middle frame as numpy uint8 [H, W, C]."""
    import numpy as np
    import torch

    v = video_tensor
    # Expected [T, C, H, W] from diffusers Wan; fall back to other common shapes
    if v.ndim == 4 and v.shape[0] > v.shape[1]:
        # [T, C, H, W]
        mid = v[v.shape[0] // 2]
    elif v.ndim == 4:
        # [C, T, H, W]
        mid = v[:, v.shape[1] // 2]
    else:
        raise ValueError(f"Unexpected video shape: {v.shape}")
    mid = mid.clamp(0, 1) if mid.dtype.is_floating_point else mid
    arr = (mid * 255).round().clamp(0, 255).to(torch.uint8)
    return arr.cpu().numpy().transpose(1, 2, 0)


def _ocr_score(frame_np: "np.ndarray", target: str, engine) -> tuple[float, str]:
    """Return (score, detected_text)."""
    import re
    import string
    from difflib import SequenceMatcher

    def normalize(t: str) -> str:
        t = t.lower().strip()
        t = t.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", t).strip()

    if not target:
        return 0.0, ""

    result, _ = engine(frame_np)
    if not result:
        return 0.0, ""

    ocr_raw = " ".join(item[1] for item in result)
    ocr_norm = normalize(ocr_raw)
    target_norm = normalize(target)
    if len(ocr_norm) < 2:
        return 0.0, ocr_raw
    ratio = SequenceMatcher(None, ocr_norm, target_norm).ratio()
    return ratio, ocr_raw


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Wan 1.3B base vs LoRA A/B inference")
    parser.add_argument("--model-path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to LoRA weights dir (from PeftModel.save_pretrained)")
    parser.add_argument(
        "--manifest", type=str,
        default=str(Path(__file__).resolve().parents[3] / "datasets" / "ocr" / "train.txt"),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/ab_infer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-prompts", type=int, default=0,
                        help="Cap number of prompts (0 = all)")
    args = parser.parse_args()

    import torch
    from diffusers import WanPipeline
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    pairs = _load_prompts(manifest_path)
    if args.num_prompts > 0:
        pairs = pairs[: args.num_prompts]
    logger.info("Loaded %d prompts from %s", len(pairs), manifest_path)

    gen_cfg = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
    }

    ocr_engine = RapidOCR()
    results: dict[str, list[dict]] = {"base": [], "lora": []}

    for variant in ["base", "lora"]:
        logger.info("========== Variant: %s ==========", variant)

        logger.info("Loading WanPipeline…")
        pipeline = WanPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )
        pipeline.to("cuda")

        if variant == "lora":
            from peft import PeftModel
            logger.info("Applying LoRA from %s", args.lora_path)
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, args.lora_path, is_trainable=False
            )

        pipeline.scheduler.set_timesteps(args.num_steps, device="cuda")

        for idx, (prompt, target) in enumerate(pairs):
            logger.info("[%s] %d/%d: target=%r", variant, idx + 1, len(pairs), target)
            video = _generate(pipeline, prompt, args.seed, gen_cfg)
            frame_np = _middle_frame_uint8(video)
            score, ocr_text = _ocr_score(frame_np, target, ocr_engine)

            tag = f"{idx:03d}_{target.replace(' ', '_')[:20]}"
            img_path = output_dir / f"{variant}_{tag}.png"
            Image.fromarray(frame_np).save(img_path)

            results[variant].append({
                "idx": idx, "prompt": prompt, "target": target,
                "score": score, "ocr": ocr_text, "image": str(img_path),
            })
            logger.info("  → score=%.3f ocr=%r", score, ocr_text)

        # Free the pipeline before loading the next variant
        del pipeline
        torch.cuda.empty_cache()

    # --- summary ---
    import statistics
    base_scores = [r["score"] for r in results["base"]]
    lora_scores = [r["score"] for r in results["lora"]]

    summary_lines = [
        "=" * 60,
        "A/B Summary (same seed, same prompts)",
        "=" * 60,
        f"Seed: {args.seed}",
        f"Prompts: {len(pairs)}",
        "",
        f"{'target':20s} {'base':>8s} {'lora':>8s} {'delta':>8s}",
    ]
    delta_total = 0.0
    for b, l in zip(results["base"], results["lora"]):
        delta = l["score"] - b["score"]
        delta_total += delta
        summary_lines.append(
            f"{b['target']:20s} {b['score']:>8.3f} {l['score']:>8.3f} {delta:>+8.3f}"
        )

    summary_lines += [
        "",
        f"base mean:  {statistics.mean(base_scores):.3f}  (median {statistics.median(base_scores):.3f})",
        f"lora mean:  {statistics.mean(lora_scores):.3f}  (median {statistics.median(lora_scores):.3f})",
        f"mean delta: {delta_total / len(pairs):+.3f}",
    ]

    summary = "\n".join(summary_lines)
    print(summary)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
