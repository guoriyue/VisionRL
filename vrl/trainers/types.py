"""Trainer configuration and training state.

Schema aligned with verl (`actor_rollout_ref.actor.*`) and OpenRLHF
(`actor_learning_rate`, `n_samples_per_prompt`, `init_kl_coef`,
`max_norm`). Diffusion-specific knobs (`timestep_fraction`) live alongside
the rollout block.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OptimConfig:
    """Optimizer hyper-parameters (verl: `actor.optim.*`)."""

    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    eps: float = 1e-8
    use_8bit_adam: bool = False
    allow_tf32: bool = True


@dataclass(slots=True)
class EMAConfig:
    """Exponential moving average of model weights."""

    enable: bool = False
    decay: float = 0.9999
    update_interval: int = 1


@dataclass(slots=True)
class DebugConfig:
    """Diagnostic toggles consumed by the trainer."""

    # First-step log-prob round-trip check (collected old_lp vs fresh_lp).
    first_step: bool = False
    # One-shot ||grad(policy)|| vs ||grad(beta*kl)|| split.
    grad_split: bool = False


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the online RL training loop.

    Field naming follows verl + OpenRLHF conventions:
      - ``optim.lr``                 (was ``lr``)
      - ``ppo_epochs``               (was ``num_inner_epochs``)
      - ``max_norm``                 (was ``max_grad_norm``)
      - ``bf16``                     (was ``mixed_precision == "bf16"``)
      - ``n``                        (was ``group_size`` — verl: ``rollout.n``)
      - ``rollout_batch_size``       (was ``prompts_per_step``)
    """

    # --- nested groups ---
    optim: OptimConfig = field(default_factory=OptimConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # --- gradient ---
    max_norm: float = 1.0

    # --- PPO/GRPO loop ---
    ppo_epochs: int = 1

    # --- precision ---
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- rollout knobs the trainer drives ---
    n: int = 4
    rollout_batch_size: int = 4
    timestep_fraction: float = 1.0

    # --- lifecycle ---
    total_epochs: int = 10000
    save_freq: int = 50
    log_freq: int = 1
    output_dir: str = "outputs/"
    seed: int = 0

    # --- profiling ---
    profile: bool = False


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
