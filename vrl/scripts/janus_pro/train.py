"""Janus-Pro GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches Janus-Pro configs here:

    JanusProPolicy  --(rollout)-->  JanusProCollector
                                       |
                                       v
                          TokenLogProbEvaluator
                                       |
                                       v
                                  TokenGRPO
                                       |
                                       v
                                OnlineTrainer

Two public coroutines:
  * ``train_janus_pro_grpo``      — general-purpose mix (PickScore + Aesthetic).
  * ``train_janus_pro_ocr_grpo``  — OCR reward + ``target_text`` manifest.

Both share construction logic; the only differences are reward selection
and manifest format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrl.engine.generation import RolloutBackend


async def train_janus_pro_grpo(
    cfg: DictConfig,
    *,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    """Run Janus-Pro GRPO with a multi-component image-quality reward."""
    await _train_janus_pro(
        cfg,
        ocr_mode=False,
        rollout_runtime=rollout_runtime,
    )


async def train_janus_pro_ocr_grpo(
    cfg: DictConfig,
    *,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    """Run Janus-Pro GRPO with the OCR edit-distance reward.

    Manifest entries must carry ``target_text`` so ``OCRReward`` knows what
    string to look for in each rendered image.
    """
    await _train_janus_pro(
        cfg,
        ocr_mode=True,
        rollout_runtime=rollout_runtime,
    )


async def _train_janus_pro(
    cfg: DictConfig,
    *,
    ocr_mode: bool,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    import csv

    import torch

    from vrl.algorithms.grpo_token import TokenGRPO, TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs, require
    from vrl.engine.generation import build_rollout_backend_from_cfg
    from vrl.models.families.janus_pro import JanusProConfig, JanusProPolicy
    from vrl.rollouts.collectors.janus_pro import (
        JanusProCollector,
        JanusProCollectorConfig,
    )
    from vrl.rollouts.evaluators.lm import TokenLogProbEvaluator
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    # AR rollout uses ``n_samples_per_prompt``; mirror it onto the runtime
    # ``TrainerConfig.n`` (which the OnlineTrainer hands as ``group_size`` to
    # the collector). ``rollout_batch_size`` similarly drives the per-step
    # prompt count below.
    trainer_config.n = int(cfg.rollout.n_samples_per_prompt)
    trainer_config.rollout_batch_size = int(cfg.rollout.rollout_batch_size)

    torch.manual_seed(trainer_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Model -----------------------------------------------------------
    model_cfg = cfg.model
    use_lora = bool(require(cfg, "model.use_lora"))
    logger.info("Loading Janus-Pro from %s ...", model_cfg.path)
    policy = JanusProPolicy(JanusProConfig(
        model_path=str(model_cfg.path),
        dtype=str(require(cfg, "model.dtype")),
        use_lora=use_lora,
        lora_rank=int(require(cfg, "model.lora.rank")),
        lora_alpha=int(require(cfg, "model.lora.alpha")),
        lora_dropout=float(require(cfg, "model.lora.dropout")),
        lora_target_modules=tuple(require(cfg, "model.lora.target_modules")),
        lora_init=str(require(cfg, "model.lora.init")),
        cfg_weight=float(cfg.sampling.cfg_weight),
        temperature=float(cfg.sampling.temperature),
        image_token_num=int(cfg.sampling.image_token_num),
        device=str(device),
    ))
    logger.info("Trainable params: %.2f M", policy.trainable_param_count() / 1e6)

    # 2. Reward ----------------------------------------------------------
    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    from vrl.rewards.multi import MultiReward
    reward_fn = MultiReward.from_dict(
        reward_weights, device=str(device), reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)

    # 3. Collector + evaluator + algorithm -------------------------------
    collector = JanusProCollector(
        model=policy, reward_fn=reward_fn,
        config=JanusProCollectorConfig(
            n_samples_per_prompt=trainer_config.n,
            cfg_weight=float(cfg.sampling.cfg_weight),
            temperature=float(cfg.sampling.temperature),
            image_token_num=int(cfg.sampling.image_token_num),
            image_size=int(cfg.sampling.image_size),
            rescale_to_unit=bool(require(cfg, "rollout.rescale_to_unit")),
            max_text_length=int(require(cfg, "rollout.max_text_length")),
        ),
    )
    collector._runtime = build_rollout_backend_from_cfg(
        cfg,
        runtime=rollout_runtime,
        local_runtime_builder=collector._build_runtime,
        driver_policy=policy,
    )
    evaluator = TokenLogProbEvaluator()
    algo_section = cfg.algorithm
    algorithm_config = built["algorithm"]
    if not isinstance(algorithm_config, TokenGRPOConfig):
        raise TypeError(
            f"Janus expects algorithm.kind=token_grpo, got {type(algorithm_config).__name__}",
        )
    algorithm = TokenGRPO(algorithm_config)

    stat_tracker = (
        PerPromptStatTracker(global_std=algorithm.config.global_std)
        if algo_section.get("per_prompt_stat_tracking", True)
        else None
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=policy,
        config=trainer_config,
        device=policy.device,
        stat_tracker=stat_tracker,
    )

    # 4. Prompts ---------------------------------------------------------
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples: list[PromptExample] = load_prompt_manifest(manifest_path)
    logger.info(
        "Starting Janus-Pro GRPO (%s) — %d epochs, %d examples, n=%d",
        "ocr" if ocr_mode else "general",
        trainer_config.total_epochs, len(examples), trainer_config.n,
    )

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "loss", "policy_loss", "kl_penalty",
             "reward_mean", "reward_std", "approx_kl", "clip_fraction",
             "advantage_mean", "grad_norm", "adv_saturation",
             "adv_zero_rate", "group_size", "trained_prompt_num"]
            + [f"r_{n}" for n in component_names]
        )

        rng = torch.Generator().manual_seed(trainer_config.seed)
        for epoch in range(trainer_config.total_epochs):
            idx = torch.randperm(len(examples), generator=rng)[
                : trainer_config.rollout_batch_size
            ].tolist()
            example_batch = [examples[i] for i in idx]
            metrics = await trainer.step(example_batch)

            if epoch % trainer_config.log_freq == 0:
                last = getattr(reward_fn, "last_components", {}) or {}
                component_means = {
                    n: (sum(last.get(n, [])) / len(last.get(n, [])))
                       if last.get(n) else float("nan")
                    for n in component_names
                }
                component_str = " ".join(
                    f"{n}={component_means[n]:.3f}" for n in component_names
                )
                logger.info(
                    "Epoch %d | loss=%.4f kl=%.4f reward=%.4f+/-%.4f "
                    "clip_frac=%.3f approx_kl=%.4f | %s",
                    epoch, metrics.loss, metrics.kl_penalty,
                    metrics.reward_mean, metrics.reward_std,
                    metrics.clip_fraction, metrics.approx_kl,
                    component_str,
                )
                writer.writerow([
                    epoch, metrics.loss, metrics.policy_loss, metrics.kl_penalty,
                    metrics.reward_mean, metrics.reward_std,
                    metrics.approx_kl, metrics.clip_fraction,
                    metrics.advantage_mean, metrics.grad_norm,
                    metrics.adv_saturation, metrics.adv_zero_rate,
                    metrics.group_size, metrics.trained_prompt_num,
                    *(component_means[n] for n in component_names),
                ])
                f.flush()

            if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
                ckpt_path = output_dir / f"checkpoint-{epoch+1}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                if use_lora:
                    policy.language_model.save_pretrained(ckpt_path / "lora_weights")
                torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
                logger.info("Saved checkpoint to %s", ckpt_path)

    final_path = output_dir / "checkpoint-final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), final_path / "trainer_state.pt")
    if use_lora:
        policy.language_model.save_pretrained(final_path / "lora_weights")
    logger.info("Training complete. Final checkpoint: %s", final_path)
