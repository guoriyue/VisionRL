"""NextStep-1 OCR GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches NextStep-1 configs to
``train_nextstep_1_ocr_grpo`` here. Continuous-token AR image RL —
TokenGRPO + ContinuousTokenLogProbEvaluator + NextStep1Collector.

Mirrors the Janus-Pro pattern (TokenGRPO + AR) — only the model wrapper,
collector, and evaluator differ from ``vrl.scripts.janus_pro``. The
``OnlineTrainer`` machinery is identical.

Note: NextStep-1 is pre-smoke-test — see ``# TODO(nextstep-binding)``
markers in ``vrl/models/families/nextstep_1/policy.py``,
``vrl/models/families/nextstep_1/flow_step.py``, and
``vrl/rollouts/collectors/nextstep_1.py``. The driver here is YAML-driven
scaffolding; binding work is tracked separately.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrl.engine.generation import RolloutBackend


async def train_nextstep_1_ocr_grpo(
    cfg: DictConfig,
    *,
    rollout_runtime: RolloutBackend | None = None,
) -> None:
    """Run NextStep-1 OCR GRPO training driven by a merged YAML config."""
    import torch

    from vrl.algorithms.grpo_token import TokenGRPO, TokenGRPOConfig
    from vrl.config.loader import build_configs, require
    from vrl.models.families.nextstep_1 import NextStep1Config, NextStep1Policy
    from vrl.rewards.ocr import OCRReward
    from vrl.rollouts.backend import build_rollout_backend_from_cfg
    from vrl.rollouts.backend_config import RolloutBackendConfig
    from vrl.rollouts.collectors import (
        NextStep1CollectorConfig,
        build_rollout_collector,
    )
    from vrl.rollouts.evaluators.ar import ContinuousTokenLogProbEvaluator
    from vrl.rollouts.runtime_inputs import build_rollout_runtime_inputs
    from vrl.trainers.data import load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.weight_sync import build_runtime_weight_syncer

    built = build_configs(cfg)
    trainer_config = built["trainer"]

    # ------------------------------------------------------------------
    # 1. Build NextStep-1 policy from cfg.model + cfg.sampling slices.
    # ------------------------------------------------------------------
    model_cfg = cfg.model
    sampling = cfg.sampling
    use_lora = bool(require(cfg, "model.use_lora"))
    lora_targets_tuple = tuple(require(cfg, "model.lora.target_modules"))

    torch.manual_seed(trainer_config.seed)

    logger.info("Loading NextStep-1 from %s ...", model_cfg.path)
    model = NextStep1Policy(
        NextStep1Config(
            model_path=model_cfg.path,
            vae_path=model_cfg.vae_path,
            dtype=str(require(cfg, "model.dtype")),
            use_lora=use_lora,
            lora_rank=int(require(cfg, "model.lora.rank")),
            lora_alpha=int(require(cfg, "model.lora.alpha")),
            lora_dropout=float(require(cfg, "model.lora.dropout")),
            lora_target_modules=lora_targets_tuple,
            lora_init=str(require(cfg, "model.lora.init")),
            cfg_scale=float(sampling.cfg_scale),
            num_flow_steps=int(sampling.num_flow_steps),
            noise_level=float(sampling.noise_level),
            image_token_num=int(sampling.image_token_num),
            image_size=int(sampling.image_size),
            freeze_vae=bool(require(cfg, "model.freeze_vae")),
            freeze_image_head=bool(require(cfg, "model.freeze_image_head")),
            gradient_checkpointing=bool(trainer_config.gradient_checkpointing),
        )
    )
    logger.info("Trainable params: %.2f M", model.trainable_param_count() / 1e6)

    # ------------------------------------------------------------------
    # 2. Reward — OCR only for this recipe (Phase 4 will generalise).
    # ------------------------------------------------------------------
    reward_weights, reward_kwargs = built["reward"]
    if float(reward_weights.get("ocr", 0.0)) <= 0.0:
        raise ValueError(
            f"nextstep_1_ocr_grpo requires reward.components.ocr > 0; got {reward_weights}",
        )
    debug_dir = reward_kwargs.get("ocr", {}).get("debug_dir") or None
    reward = OCRReward(debug_dir=debug_dir)
    if debug_dir:
        logger.info("OCR debug frames -> %s", debug_dir)

    # ------------------------------------------------------------------
    # 3. Collector + evaluator + algorithm.
    # ------------------------------------------------------------------
    collector_config = NextStep1CollectorConfig(
        n_samples_per_prompt=int(require(cfg, "rollout.n_samples_per_prompt")),
        cfg_scale=float(sampling.cfg_scale),
        num_flow_steps=int(sampling.num_flow_steps),
        noise_level=float(require(cfg, "rollout.noise_level")),
        image_token_num=int(sampling.image_token_num),
        image_size=int(sampling.image_size),
        rescale_to_unit=bool(require(cfg, "rollout.rescale_to_unit")),
        max_text_length=int(require(cfg, "rollout.max_text_length")),
    )
    collector = build_rollout_collector(
        "nextstep_1",
        model=model,
        reward_fn=reward,
        config=collector_config,
    )
    rollout_backend_config = RolloutBackendConfig.from_cfg(cfg)
    rollout_runtime_inputs = build_rollout_runtime_inputs(
        cfg,
        "nextstep_1",
        weight_dtype=str(require(cfg, "model.dtype")),
    )
    collector.set_runtime(
        build_rollout_backend_from_cfg(
            cfg,
            runtime=rollout_runtime,
            local_runtime_builder=collector.build_runtime,
            driver_policy=None if rollout_backend_config.backend == "ray" else model,
            runtime_spec=(
                rollout_runtime_inputs.runtime_spec if rollout_runtime_inputs is not None else None
            ),
            gatherer=(
                rollout_runtime_inputs.gatherer if rollout_runtime_inputs is not None else None
            ),
        ),
    )

    evaluator = ContinuousTokenLogProbEvaluator()

    algorithm_config = built["algorithm"]
    if not isinstance(algorithm_config, TokenGRPOConfig):
        raise TypeError(
            f"NextStep expects algorithm.kind=token_grpo, got {type(algorithm_config).__name__}",
        )
    algorithm = TokenGRPO(algorithm_config)

    # Align trainer.n with collector group size so OnlineTrainer's CEA
    # bookkeeping matches the rollout layout.
    trainer_config.n = collector_config.n_samples_per_prompt

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=model,
        weight_syncer=build_runtime_weight_syncer(collector.runtime),
        config=trainer_config,
        device=model.device,
    )

    # ------------------------------------------------------------------
    # 4. Prompts + loop.
    # ------------------------------------------------------------------
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples = load_prompt_manifest(manifest_path)
    logger.info(
        "Loaded %d OCR prompt examples from %s",
        len(examples),
        manifest_path,
    )

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    rollout_batch_size = int(require(cfg, "rollout.rollout_batch_size"))
    rng = torch.Generator().manual_seed(trainer_config.seed)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "reward_mean",
                "reward_std",
                "approx_kl",
                "clip_fraction",
                "target_text",
                "prompt",
            ]
        )

        for step in range(1, trainer_config.total_epochs + 1):
            idx = torch.randperm(len(examples), generator=rng)[:rollout_batch_size].tolist()
            batch_examples = [examples[i] for i in idx]

            metrics = await trainer.step(batch_examples)

            if step % trainer_config.log_freq == 0:
                writer.writerow(
                    [
                        step,
                        metrics.loss,
                        metrics.reward_mean,
                        metrics.reward_std,
                        metrics.approx_kl,
                        metrics.clip_fraction,
                        batch_examples[0].target_text,
                        batch_examples[0].prompt[:60],
                    ]
                )
                f.flush()
                logger.info(
                    "step=%d target=%r reward=%.3f+/-%.3f loss=%.4f clip=%.2f kl=%.4f",
                    step,
                    batch_examples[0].target_text,
                    metrics.reward_mean,
                    metrics.reward_std,
                    metrics.loss,
                    metrics.clip_fraction,
                    metrics.approx_kl,
                )

            if trainer_config.save_freq > 0 and step % trainer_config.save_freq == 0:
                ckpt = output_dir / f"checkpoint-{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                if use_lora:
                    model.language_model.save_pretrained(ckpt / "lora_weights")
                logger.info("Saved checkpoint at step %d -> %s", step, ckpt)

    logger.info("Training complete -- metrics at %s", csv_path)
