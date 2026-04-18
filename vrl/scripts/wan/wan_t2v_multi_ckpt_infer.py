"""Multi-checkpoint A/B inference: base vs N LoRA checkpoints.

Generates one video per (variant, prompt) pair, dumps the middle frame and
runs OCR scoring. Produces a side-by-side table across all variants so you
can see the training trajectory (checkpoint-25 → 50 → final) instead of
only comparing base vs one final LoRA.

Usage:
    python -m vrl.scripts.wan.wan_t2v_multi_ckpt_infer \
        --lora ep25=outputs/wan_1_3b_ocr_50ep_save/checkpoint-25/lora_weights \
        --lora ep50=outputs/wan_1_3b_ocr_50ep_save/checkpoint-50/lora_weights \
        --lora final=outputs/wan_1_3b_ocr_50ep_save/checkpoint-final/lora_weights \
        --manifest vrl/scripts/examples/ocr_prompts.jsonl \
        --output-dir outputs/multi_ckpt_ab
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_prompts(manifest_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pairs.append((row["prompt"], row.get("target_text", "")))
    return pairs


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
    return out.frames[0]


def _middle_frame_uint8(video_tensor: "torch.Tensor") -> "np.ndarray":
    import torch

    v = video_tensor
    if v.ndim == 4 and v.shape[0] > v.shape[1]:
        mid = v[v.shape[0] // 2]
    elif v.ndim == 4:
        mid = v[:, v.shape[1] // 2]
    else:
        raise ValueError(f"Unexpected video shape: {v.shape}")
    mid = mid.clamp(0, 1) if mid.dtype.is_floating_point else mid
    arr = (mid * 255).round().clamp(0, 255).to(torch.uint8)
    return arr.cpu().numpy().transpose(1, 2, 0)


def _ocr_score(frame_np: "np.ndarray", target: str, engine) -> tuple[float, str]:
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


def _parse_lora_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--lora must be name=path, got {spec!r}"
        )
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"invalid --lora spec {spec!r}")
    return name, path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Wan 1.3B base vs N LoRA checkpoints")
    parser.add_argument("--model-path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument(
        "--lora",
        type=_parse_lora_spec,
        action="append",
        required=True,
        help="name=path entry for a LoRA checkpoint. Repeatable.",
    )
    parser.add_argument("--manifest", type=str,
                        default="vrl/scripts/examples/ocr_prompts.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/multi_ckpt_ab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-prompts", type=int, default=0,
                        help="Cap number of prompts (0 = all)")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the base variant (LoRA-only comparison)")
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

    # Build variant list: base (unless skipped) + all --lora entries in order
    variants: list[tuple[str, str | None]] = []
    if not args.skip_base:
        variants.append(("base", None))
    for name, path in args.lora:
        variants.append((name, path))

    logger.info("Variants to run: %s", [v[0] for v in variants])

    ocr_engine = RapidOCR()
    results: dict[str, list[dict]] = {name: [] for name, _ in variants}

    for variant_name, lora_path in variants:
        logger.info("========== Variant: %s ==========", variant_name)

        logger.info("Loading WanPipeline…")
        pipeline = WanPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )
        pipeline.to("cuda")

        if lora_path is not None:
            from peft import PeftModel
            logger.info("Applying LoRA from %s", lora_path)
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, lora_path, is_trainable=False
            )

        pipeline.scheduler.set_timesteps(args.num_steps, device="cuda")

        for idx, (prompt, target) in enumerate(pairs):
            logger.info("[%s] %d/%d: target=%r", variant_name, idx + 1, len(pairs), target)
            video = _generate(pipeline, prompt, args.seed, gen_cfg)
            frame_np = _middle_frame_uint8(video)
            score, ocr_text = _ocr_score(frame_np, target, ocr_engine)

            tag = f"{idx:03d}_{target.replace(' ', '_')[:20]}"
            img_path = output_dir / f"{variant_name}_{tag}.png"
            Image.fromarray(frame_np).save(img_path)

            results[variant_name].append({
                "idx": idx, "prompt": prompt, "target": target,
                "score": score, "ocr": ocr_text, "image": str(img_path),
            })
            logger.info("  → score=%.3f ocr=%r", score, ocr_text)

        del pipeline
        torch.cuda.empty_cache()

    # ---- summary ----
    import statistics

    variant_names = [v[0] for v in variants]
    header_cols = [f"{n:>9s}" for n in variant_names]
    summary_lines = [
        "=" * (22 + 10 * len(variant_names)),
        "Multi-checkpoint A/B (same seed, same prompts)",
        "=" * (22 + 10 * len(variant_names)),
        f"Seed: {args.seed}",
        f"Prompts: {len(pairs)}",
        f"Variants: {variant_names}",
        "",
        f"{'target':20s} " + " ".join(header_cols),
    ]

    n = len(pairs)
    for i in range(n):
        target = results[variant_names[0]][i]["target"]
        score_cells = [f"{results[v][i]['score']:>9.3f}" for v in variant_names]
        summary_lines.append(f"{target:20s} " + " ".join(score_cells))

    summary_lines.append("")
    mean_cells = [f"{statistics.mean([r['score'] for r in results[v]]):>9.3f}" for v in variant_names]
    median_cells = [f"{statistics.median([r['score'] for r in results[v]]):>9.3f}" for v in variant_names]
    summary_lines.append(f"{'mean':20s} " + " ".join(mean_cells))
    summary_lines.append(f"{'median':20s} " + " ".join(median_cells))

    # Deltas vs first variant (usually base)
    base_name = variant_names[0]
    base_scores = [r["score"] for r in results[base_name]]
    summary_lines.append("")
    summary_lines.append(f"Delta vs {base_name!r}:")
    for v in variant_names[1:]:
        v_scores = [r["score"] for r in results[v]]
        deltas = [a - b for a, b in zip(v_scores, base_scores)]
        mean_delta = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0.01)
        losses = sum(1 for d in deltas if d < -0.01)
        summary_lines.append(
            f"  {v:>12s}: mean_delta={mean_delta:+.3f}  wins={wins}  losses={losses}  ties={len(deltas) - wins - losses}"
        )

    summary = "\n".join(summary_lines)
    print(summary)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
