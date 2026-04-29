"""NextStep-1 OCR GRPO — continuous-token AR image RL with OCR reward.

Reads ``PromptExample`` manifests (JSONL) with per-sample ``target_text``
and trains NextStep-1's LoRA + flow-head to generate images whose
rendered text matches the target when run through ``rapidocr_onnxruntime``.

Mirror of ``vrl.scripts.janus_pro.janus_pro_1b_ocr_grpo`` — only the
model wrapper, collector, and evaluator change. The ``OnlineTrainer``
+ ``TokenGRPO`` machinery is identical.

Usage:
    python -m vrl.scripts.nextstep_1.nextstep_1_ocr_grpo \\
        --manifest datasets/ocr/train.txt \\
        --max-train-steps 50 \\
        --n-samples-per-prompt 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "datasets" / "ocr" / "train.txt"
)


@dataclass
class NextStep1OCRConfig:
    # model
    model_path: str = "stepfun-ai/NextStep-1.1"
    vae_path: str = "stepfun-ai/NextStep-1-f8ch16-Tokenizer"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    dtype: str = "bfloat16"

    # rollout
    n_samples_per_prompt: int = 4
    cfg_scale: float = 4.5
    num_flow_steps: int = 20
    noise_level: float = 1.0
    image_token_num: int = 1024
    image_size: int = 256

    # reward
    manifest: str = str(_DEFAULT_MANIFEST)
    ocr_debug_dir: str = ""

    # optimizer / training
    lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_train_steps: int = 50
    batch_prompts: int = 1
    num_batches_per_epoch: int = 1

    # algorithm
    clip_eps: float = 0.2
    kl_coeff: float = 0.0
    adv_clip_max: float = 5.0
    global_std: bool = False
    kl_estimator: str = "k3"

    # IO
    output_dir: str = "outputs/nextstep_1_ocr_grpo"
    checkpointing_steps: int = 999
    log_every: int = 1
    seed: int = 0


async def _train(config: NextStep1OCRConfig) -> None:
    import torch

    from vrl.algorithms.grpo_token import TokenGRPO, TokenGRPOConfig
    from vrl.models.families.nextstep_1 import NextStep1Config, NextStep1Policy
    from vrl.rewards.ocr import OCRReward
    from vrl.rollouts.collectors.nextstep_1 import (
        NextStep1Collector,
        NextStep1CollectorConfig,
    )
    from vrl.rollouts.evaluators.lm import ContinuousTokenLogProbEvaluator
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import OptimConfig, TrainerConfig

    torch.manual_seed(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading NextStep-1 from %s ...", config.model_path)
    model = NextStep1Policy(NextStep1Config(
        model_path=config.model_path,
        vae_path=config.vae_path,
        dtype=config.dtype,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        cfg_scale=config.cfg_scale,
        num_flow_steps=config.num_flow_steps,
        noise_level=config.noise_level,
        image_token_num=config.image_token_num,
        image_size=config.image_size,
    ))
    logger.info("Trainable params: %.2f M", model.trainable_param_count() / 1e6)

    reward = OCRReward(debug_dir=config.ocr_debug_dir or None)
    if config.ocr_debug_dir:
        logger.info("OCR debug frames → %s", config.ocr_debug_dir)

    collector = NextStep1Collector(
        model=model, reward_fn=reward,
        config=NextStep1CollectorConfig(
            n_samples_per_prompt=config.n_samples_per_prompt,
            cfg_scale=config.cfg_scale,
            num_flow_steps=config.num_flow_steps,
            noise_level=config.noise_level,
            image_token_num=config.image_token_num,
            image_size=config.image_size,
        ),
    )
    evaluator = ContinuousTokenLogProbEvaluator()
    algorithm = TokenGRPO(TokenGRPOConfig(
        eps_clip=config.clip_eps,
        init_kl_coef=config.kl_coeff,
        adv_clip_max=config.adv_clip_max,
        global_std=config.global_std,
        kl_estimator=config.kl_estimator,
    ))

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=model,
        config=TrainerConfig(
            optim=OptimConfig(lr=config.lr, weight_decay=config.weight_decay),
            max_norm=config.max_grad_norm,
            n=config.n_samples_per_prompt,
            bf16=(config.mixed_precision == "bf16"),
            ppo_epochs=1,
            timestep_fraction=1.0,
        ),
        device=model.device,
    )

    from vrl.trainers.data import load_prompt_manifest
    examples = load_prompt_manifest(config.manifest)
    logger.info("Loaded %d OCR prompt examples from %s", len(examples), config.manifest)
    rng = torch.Generator().manual_seed(config.seed)

    csv_path = out / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "loss", "reward_mean", "reward_std",
            "approx_kl", "clip_fraction", "target_text", "prompt",
        ])

        for step in range(1, config.max_train_steps + 1):
            n = config.num_batches_per_epoch * config.batch_prompts
            idx = torch.randperm(len(examples), generator=rng)[:n].tolist()
            batch_examples = [examples[i] for i in idx]

            metrics = await trainer.step(batch_examples)

            if step % config.log_every == 0:
                writer.writerow([
                    step, metrics.loss, metrics.reward_mean, metrics.reward_std,
                    metrics.approx_kl, metrics.clip_fraction,
                    batch_examples[0].target_text, batch_examples[0].prompt[:60],
                ])
                f.flush()
                logger.info(
                    "step=%d target=%r reward=%.3f±%.3f loss=%.4f clip=%.2f kl=%.4f",
                    step, batch_examples[0].target_text,
                    metrics.reward_mean, metrics.reward_std,
                    metrics.loss, metrics.clip_fraction, metrics.approx_kl,
                )

            if step % config.checkpointing_steps == 0:
                ckpt = out / f"checkpoint-{step}"
                ckpt.mkdir(exist_ok=True)
                if config.use_lora:
                    model.language_model.save_pretrained(ckpt / "lora_weights")
                logger.info("Saved checkpoint at step %d → %s", step, ckpt)

    logger.info("Training complete — metrics at %s", csv_path)


def main() -> None:
    p = argparse.ArgumentParser(description="NextStep-1 OCR GRPO trainer")
    p.add_argument("--manifest", default=str(_DEFAULT_MANIFEST))
    p.add_argument("--model-path", default=NextStep1OCRConfig.model_path)
    p.add_argument("--vae-path", default=NextStep1OCRConfig.vae_path)
    p.add_argument("--output-dir", default=NextStep1OCRConfig.output_dir)
    p.add_argument("--max-train-steps", type=int, default=NextStep1OCRConfig.max_train_steps)
    p.add_argument("--lr", type=float, default=NextStep1OCRConfig.lr)
    p.add_argument("--n-samples-per-prompt", type=int,
                   default=NextStep1OCRConfig.n_samples_per_prompt)
    p.add_argument("--batch-prompts", type=int, default=NextStep1OCRConfig.batch_prompts)
    p.add_argument("--num-batches-per-epoch", type=int,
                   default=NextStep1OCRConfig.num_batches_per_epoch)
    p.add_argument("--cfg-scale", type=float, default=NextStep1OCRConfig.cfg_scale)
    p.add_argument("--num-flow-steps", type=int, default=NextStep1OCRConfig.num_flow_steps)
    p.add_argument("--noise-level", type=float, default=NextStep1OCRConfig.noise_level)
    p.add_argument("--kl-coeff", type=float, default=NextStep1OCRConfig.kl_coeff)
    p.add_argument("--clip-eps", type=float, default=NextStep1OCRConfig.clip_eps)
    p.add_argument("--lora-rank", type=int, default=NextStep1OCRConfig.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=NextStep1OCRConfig.lora_alpha)
    p.add_argument("--ocr-debug-dir", default=NextStep1OCRConfig.ocr_debug_dir)
    p.add_argument("--checkpointing-steps", type=int,
                   default=NextStep1OCRConfig.checkpointing_steps)
    p.add_argument("--seed", type=int, default=NextStep1OCRConfig.seed)
    p.add_argument(
        "--global-std", action="store_true",
        help="Use global-batch std for advantage normalization (default: per-prompt std).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = NextStep1OCRConfig(
        manifest=args.manifest,
        model_path=args.model_path,
        vae_path=args.vae_path,
        output_dir=args.output_dir,
        max_train_steps=args.max_train_steps,
        lr=args.lr,
        n_samples_per_prompt=args.n_samples_per_prompt,
        batch_prompts=args.batch_prompts,
        num_batches_per_epoch=args.num_batches_per_epoch,
        cfg_scale=args.cfg_scale,
        num_flow_steps=args.num_flow_steps,
        noise_level=args.noise_level,
        kl_coeff=args.kl_coeff,
        clip_eps=args.clip_eps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        ocr_debug_dir=args.ocr_debug_dir,
        checkpointing_steps=args.checkpointing_steps,
        seed=args.seed,
        global_std=args.global_std,
    )
    asyncio.run(_train(cfg))


if __name__ == "__main__":
    main()
