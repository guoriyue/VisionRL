"""Janus-Pro-1B GRPO training script — autoregressive image RL.

Single-GPU H100 reference recipe. Wraps the four building blocks into
one CLI entry point:

    JanusProT2I  --(rollout)-->  JanusCollector
                                       │
                                       ▼
                          TokenLogProbEvaluator
                                       │
                                       ▼
                                  TokenGRPO
                                       │
                                       ▼
                                OnlineTrainer

Usage:
    python -m vrl.scripts.janus.janus_pro_1b_grpo \\
        --model-path deepseek-ai/Janus-Pro-1B \\
        --output-dir outputs/janus_1b_grpo \\
        --max-train-steps 500

Note: Requires the upstream ``deepseek-ai/Janus`` package (not on PyPI).
See ``vrl/models/families/janus/model.py`` for install instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JanusGRPOConfig:
    """Hyper-parameters for the Janus-Pro-1B GRPO recipe."""

    # ---- model ----
    model_path: str = "deepseek-ai/Janus-Pro-1B"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    dtype: str = "bfloat16"

    # ---- generation / rollout ----
    n_samples_per_prompt: int = 8
    cfg_weight: float = 5.0
    temperature: float = 1.0
    image_token_num: int = 576
    image_size: int = 384

    # ---- reward ----
    # YAML loaded by ``MultiReward.from_yaml``. Default = aesthetic only,
    # which is known to reward-hack — for serious runs combine with
    # PickScore or CLIP-Score (see vrl/rewards/multi.py).
    reward_config: str = ""
    reward_name: str = "aesthetic"

    # ---- optimizer / training ----
    lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_train_steps: int = 1000

    # ---- algorithm (TokenGRPO) ----
    clip_eps: float = 0.2
    kl_coeff: float = 0.04
    adv_clip_max: float = 5.0
    global_std: bool = True
    kl_estimator: str = "k3"

    # ---- prompts ----
    prompts_file: str = ""        # one prompt per line; empty → use built-in demo set
    batch_prompts: int = 4         # number of distinct prompts per training step

    # ---- IO ----
    output_dir: str = "outputs/janus_1b_grpo"
    checkpointing_steps: int = 200
    log_every: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEMO_PROMPTS: tuple[str, ...] = (
    "A photo of a corgi wearing sunglasses on a beach",
    "A watercolor painting of a quiet Japanese garden in autumn",
    "A neon-lit cyberpunk street with rain reflections",
    "A close-up portrait of a snow leopard, soft natural light",
    "A cozy reading nook with a stack of books and a window seat",
    "A vintage motorcycle on a desert highway at sunset",
)


def _load_prompts(path: str) -> list[str]:
    if not path:
        return list(_DEMO_PROMPTS)
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_reward(config: JanusGRPOConfig) -> Any:
    """Construct the reward function from YAML or fall back to aesthetic."""
    if config.reward_config:
        from vrl.rewards.multi import MultiReward
        return MultiReward.from_yaml(config.reward_config)

    # Single-reward fall-back. Keep imports lazy so the script remains
    # importable on machines that don't have the reward weights.
    if config.reward_name == "aesthetic":
        from vrl.rewards.aesthetic import AestheticReward
        return AestheticReward()
    raise ValueError(
        f"Unknown reward_name={config.reward_name!r}. Use --reward-config "
        "to point at a YAML for multi-reward."
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


async def _train_async(config: JanusGRPOConfig) -> None:
    import torch

    from vrl.algorithms.grpo_lm import TokenGRPO, TokenGRPOConfig
    from vrl.models.families.janus import JanusProConfig, JanusProT2I
    from vrl.rollouts.collectors.janus import JanusCollector, JanusCollectorConfig
    from vrl.rollouts.evaluators.lm import TokenLogProbEvaluator
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    torch.manual_seed(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Model -----------------------------------------------------------
    logger.info("Loading Janus-Pro from %s ...", config.model_path)
    model = JanusProT2I(JanusProConfig(
        model_path=config.model_path,
        dtype=config.dtype,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        cfg_weight=config.cfg_weight,
        temperature=config.temperature,
        image_token_num=config.image_token_num,
    ))
    n_train = model.trainable_param_count()
    logger.info("Trainable params: %.2f M", n_train / 1e6)

    # 2. Reward ----------------------------------------------------------
    reward = _build_reward(config)

    # 3. Collector / evaluator / algorithm ------------------------------
    collector = JanusCollector(
        model=model,
        reward_fn=reward,
        config=JanusCollectorConfig(
            n_samples_per_prompt=config.n_samples_per_prompt,
            cfg_weight=config.cfg_weight,
            temperature=config.temperature,
            image_token_num=config.image_token_num,
            image_size=config.image_size,
        ),
    )
    evaluator = TokenLogProbEvaluator()
    algorithm = TokenGRPO(TokenGRPOConfig(
        clip_eps=config.clip_eps,
        kl_coeff=config.kl_coeff,
        adv_clip_max=config.adv_clip_max,
        global_std=config.global_std,
        kl_estimator=config.kl_estimator,
    ))

    # 4. Trainer — CEA pipeline orchestrator ----------------------------
    # beta > 0 makes OnlineTrainer request ref log-probs from the evaluator;
    # TokenGRPO uses its own kl_coeff for the actual KL penalty weighting.
    trainer_cfg = TrainerConfig(
        lr=config.lr,
        adam_weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        batch_size=1,
        group_size=config.n_samples_per_prompt,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        num_inner_epochs=1,
        timestep_fraction=1.0,
        beta=config.kl_coeff,
    )
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=model,
        config=trainer_cfg,
        device=model.device,
    )

    # 5. Prompt source --------------------------------------------------
    prompts_pool = _load_prompts(config.prompts_file)
    rng = torch.Generator().manual_seed(config.seed)

    # 6. Train loop -----------------------------------------------------
    csv_path = out / "metrics.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "step", "loss", "policy_loss", "kl",
        "reward_mean", "reward_std", "approx_kl", "clip_fraction", "lr",
    ])

    for step in range(1, config.max_train_steps + 1):
        idx = torch.randperm(len(prompts_pool), generator=rng)[: config.batch_prompts]
        prompts = [prompts_pool[i] for i in idx.tolist()]

        metrics = await trainer.step(prompts)

        if step % config.log_every == 0:
            writer.writerow([
                step, metrics.loss, metrics.policy_loss, metrics.kl_penalty,
                metrics.reward_mean, metrics.reward_std,
                metrics.approx_kl, metrics.clip_fraction, config.lr,
            ])
            csv_file.flush()
            logger.info(
                "step=%d loss=%.4f reward_mean=%.3f kl=%.4f clip_frac=%.2f",
                step, metrics.loss, metrics.reward_mean,
                metrics.kl_penalty, metrics.clip_fraction,
            )

        if step % config.checkpointing_steps == 0:
            ckpt_dir = out / f"checkpoint-{step}"
            ckpt_dir.mkdir(exist_ok=True)
            if config.use_lora:
                model.language_model.save_pretrained(ckpt_dir / "lora_weights")
            torch.save({"step": step, "config": config.__dict__}, ckpt_dir / "state.pt")
            logger.info("Saved checkpoint at step %d → %s", step, ckpt_dir)

    csv_file.close()
    logger.info("Training complete — metrics at %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Janus-Pro-1B GRPO trainer")
    p.add_argument("--model-path", default=JanusGRPOConfig.model_path)
    p.add_argument("--output-dir", default=JanusGRPOConfig.output_dir)
    p.add_argument("--max-train-steps", type=int, default=JanusGRPOConfig.max_train_steps)
    p.add_argument("--lr", type=float, default=JanusGRPOConfig.lr)
    p.add_argument("--n-samples-per-prompt", type=int, default=JanusGRPOConfig.n_samples_per_prompt)
    p.add_argument("--batch-prompts", type=int, default=JanusGRPOConfig.batch_prompts)
    p.add_argument("--cfg-weight", type=float, default=JanusGRPOConfig.cfg_weight)
    p.add_argument("--kl-coeff", type=float, default=JanusGRPOConfig.kl_coeff)
    p.add_argument("--clip-eps", type=float, default=JanusGRPOConfig.clip_eps)
    p.add_argument("--lora-rank", type=int, default=JanusGRPOConfig.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=JanusGRPOConfig.lora_alpha)
    p.add_argument("--reward-config", default=JanusGRPOConfig.reward_config,
                   help="YAML path for MultiReward; falls back to single aesthetic if empty")
    p.add_argument("--reward-name", default=JanusGRPOConfig.reward_name)
    p.add_argument("--prompts-file", default=JanusGRPOConfig.prompts_file)
    p.add_argument("--checkpointing-steps", type=int, default=JanusGRPOConfig.checkpointing_steps)
    p.add_argument("--seed", type=int, default=JanusGRPOConfig.seed)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = JanusGRPOConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_train_steps=args.max_train_steps,
        lr=args.lr,
        n_samples_per_prompt=args.n_samples_per_prompt,
        batch_prompts=args.batch_prompts,
        cfg_weight=args.cfg_weight,
        kl_coeff=args.kl_coeff,
        clip_eps=args.clip_eps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        reward_config=args.reward_config,
        reward_name=args.reward_name,
        prompts_file=args.prompts_file,
        checkpointing_steps=args.checkpointing_steps,
        seed=args.seed,
    )
    asyncio.run(_train_async(cfg))


if __name__ == "__main__":
    main()
