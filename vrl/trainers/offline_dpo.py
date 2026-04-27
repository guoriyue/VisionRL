"""Offline Diffusion-DPO trainer.

Generic over diffusion model family — works with any model whose
forward signature is ``model(hidden_states, timestep, encoder_hidden_states)``.
The two batteries-included paths are:

  * SD UNet (epsilon prediction, ``DDPMScheduler.add_noise``)
  * Wan / SD3 transformer (flow-matching velocity, ``scheduler.scale_noise``)

For Wan video models, image-only datasets (Pick-a-Pic) are handled by
replicating each image to ``num_frames`` along the temporal dim before
VAE encoding. Set ``num_frames=1`` for true image-style training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from vrl.algorithms.dpo import DiffusionDPOConfig, diffusion_dpo_loss, diffusion_sft_loss
from vrl.trainers.pickapic import PreferenceBatch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OfflineDPOTrainerConfig:
    """Configuration for the offline DPO training loop."""

    # --- DPO ---
    beta: float = 5000.0
    sft_weight: float = 0.0

    # --- optimizer ---
    lr: float = 1e-8
    scale_lr: bool = True            # scale lr by effective batch size
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_adafactor: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # --- noise / schedule ---
    prediction_type: str = "flow_matching"   # "epsilon" | "v_prediction" | "flow_matching"
    num_train_timesteps: int = 1000          # for epsilon schedulers
    timestep_subset: tuple[int, int] | None = None  # restrict sampling, e.g. (0, 200)

    # --- video: replicate images to T frames for Wan ---
    num_frames: int = 1
    vae_scale_factor_temporal: int = 1   # for logging only

    # --- mixed precision ---
    mixed_precision: str = "bf16"        # "fp16" | "bf16" | "no"

    # --- logging ---
    log_every: int = 10


@dataclass(slots=True)
class DPOStepMetrics:
    """One training-step's metrics."""

    loss: float = 0.0
    raw_model_loss: float = 0.0
    raw_ref_loss: float = 0.0
    model_diff: float = 0.0
    ref_diff: float = 0.0
    implicit_acc: float = 0.0
    sft_loss: float = 0.0
    grad_norm: float = 0.0


def _autocast(precision: str, device: torch.device) -> Any:
    import contextlib
    if precision == "fp16":
        return torch.amp.autocast(str(device), dtype=torch.float16)
    if precision == "bf16":
        return torch.amp.autocast(str(device), dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: OfflineDPOTrainerConfig,
) -> torch.optim.Optimizer:
    if cfg.use_adafactor:
        try:
            from transformers.optimization import Adafactor
        except ImportError as e:
            raise ImportError("Install transformers for Adafactor: pip install transformers") from e
        return Adafactor(
            list(parameters), lr=cfg.lr,
            scale_parameter=False, relative_step=False, warmup_init=False,
            weight_decay=cfg.adam_weight_decay,
        )
    return torch.optim.AdamW(
        list(parameters), lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon,
    )


# ---------------------------------------------------------------------------
# Forward adapters — caller plugs in a model-specific forward function.
# ---------------------------------------------------------------------------

# A ForwardFn takes ``(model, noisy_latents, timesteps, encoder_hidden_states,
# extra_kwargs)`` and returns the prediction tensor.
ForwardFn = Callable[..., torch.Tensor]


def wan_forward(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Wan transformer forward — matches DiffusersWanT2VModel signature."""
    out = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
    return out


def sd_unet_forward(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> torch.Tensor:
    """SD 1.5 UNet forward."""
    return model(
        sample=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=(extra or {}).get("added_cond_kwargs"),
        return_dict=False,
    )[0]


# ---------------------------------------------------------------------------
# OfflineDPOTrainer
# ---------------------------------------------------------------------------


class OfflineDPOTrainer:
    """Offline DPO trainer over preference pairs.

    The trainer is **synchronous** — DPO has no rollout collection so we
    avoid the OnlineTrainer's async machinery.

    Caller responsibilities:
      * Build the policy ``model`` (typically a LoRA-wrapped backbone).
      * Build the frozen ``ref_model`` (or pass ``None`` and rely on
        LoRA adapter-disable inside ``forward_fn``).
      * Provide ``encode_pixels`` — turns ``[2B, 3, H, W]`` pixels into
        latents in the shape the model expects (handles VAE + temporal
        replication for video models).
      * Provide ``encode_text`` — turns a list of captions into the
        ``encoder_hidden_states`` tensor shaped ``[2B, ..., D]``
        (winner-then-loser convention).
      * Provide ``forward_fn`` — model-family-specific forward
        (``wan_forward`` / ``sd_unet_forward`` provided here).
      * Provide ``noise_scheduler`` for sampling timesteps + injecting
        noise. Flow-matching schedulers use ``scale_noise``; epsilon
        schedulers use ``add_noise``.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        forward_fn: ForwardFn,
        noise_scheduler: Any,
        encode_pixels: Callable[[torch.Tensor], torch.Tensor],
        encode_text: Callable[[list[str]], torch.Tensor],
        config: OfflineDPOTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.forward_fn = forward_fn
        self.noise_scheduler = noise_scheduler
        self.encode_pixels = encode_pixels
        self.encode_text = encode_text
        self.config = config or OfflineDPOTrainerConfig()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.global_step = 0

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("model has no trainable parameters — wire up LoRA / unfreeze first")
        self._optimizer = _build_optimizer(trainable, self.config)

    # ------------------------------------------------------------------
    # Noise injection — branch on prediction_type
    # ------------------------------------------------------------------

    def _sample_timesteps(self, bsz: int) -> torch.Tensor:
        if self.config.timestep_subset is not None:
            lo, hi = self.config.timestep_subset
        else:
            # Resolve the timestep range explicitly. Silently using
            # ``num_train_timesteps`` when ``scheduler.timesteps`` is empty
            # would mask the common bug of forgetting to call
            # ``scheduler.set_timesteps(...)`` before training, which can
            # silently change the sampling distribution.
            ts = getattr(self.noise_scheduler, "timesteps", None)
            if ts is None or len(ts) == 0:
                raise RuntimeError(
                    "noise_scheduler.timesteps is empty/missing. Either "
                    "call scheduler.set_timesteps(num_inference_steps) "
                    "before training, or pass an explicit "
                    "``OfflineDPOTrainerConfig.timestep_subset=(lo, hi)``."
                )
            lo, hi = 0, len(ts)
        return torch.randint(lo, hi, (bsz,), device=self.device).long()

    def _inject_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (noisy_latents, target).

        For epsilon prediction:    target = noise
        For flow-matching:         target = noise - latents (velocity)
        """
        pt = self.config.prediction_type
        if pt == "epsilon":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            return noisy, noise
        if pt == "v_prediction":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            return noisy, target
        if pt == "flow_matching":
            # Flow-matching schedulers expose ``scale_noise(sample, timestep, noise)``.
            # Forward process: x_t = (1 - σ) * x_0 + σ * noise, target velocity = noise - x_0.
            if hasattr(self.noise_scheduler, "scale_noise"):
                noisy = self.noise_scheduler.scale_noise(latents, timesteps, noise)
            else:
                # Fallback: derive from sigmas tensor directly.
                sigmas = self.noise_scheduler.sigmas.to(latents.device)
                # timesteps here index into scheduler.timesteps; map to sigma.
                # Use nearest timestep match.
                ts_idx = torch.searchsorted(
                    self.noise_scheduler.timesteps.to(self.device).flip(0),
                    timesteps.to(self.device),
                ).clamp(0, len(sigmas) - 1)
                view = (-1,) + (1,) * (latents.ndim - 1)
                sigma = sigmas[ts_idx].view(*view)
                noisy = (1.0 - sigma) * latents + sigma * noise
            target = noise - latents
            return noisy, target
        raise ValueError(f"unknown prediction_type: {pt}")

    # ------------------------------------------------------------------
    # One training step
    # ------------------------------------------------------------------

    def step(self, batch: PreferenceBatch) -> DPOStepMetrics:
        """Single optimizer step over one preference batch."""
        cfg = self.config
        self.model.train()

        # 1. Stack winner-then-loser → [2B, 3, H, W]
        pixels = batch.stacked_winner_then_loser().to(self.device)

        # 2. Pixels → latents (caller handles VAE + temporal replication)
        with torch.no_grad():
            latents = self.encode_pixels(pixels)
            # 3. Text encoding — duplicated to match 2B layout
            encoder_hidden_states = self.encode_text(batch.captions)
            if encoder_hidden_states.shape[0] != latents.shape[0]:
                # caller may return [B, ...] — repeat to [2B, ...]
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                    latents.shape[0] // encoder_hidden_states.shape[0], dim=0,
                )

        # 4. Sample shared noise + timestep across each pair
        bsz_pair = latents.shape[0] // 2
        noise = torch.randn(
            (bsz_pair,) + tuple(latents.shape[1:]),
            device=latents.device, dtype=latents.dtype,
        ).repeat(2, *([1] * (latents.ndim - 1)))
        ts_pair = self._sample_timesteps(bsz_pair)
        timesteps = ts_pair.repeat(2)

        noisy_latents, target = self._inject_noise(latents, noise, timesteps)

        # 5. Forward — policy + frozen reference
        with _autocast(cfg.mixed_precision, self.device):
            model_pred = self.forward_fn(
                self.model, noisy_latents, timesteps, encoder_hidden_states,
            )
            with torch.no_grad():
                ref_pred = self._reference_forward(
                    noisy_latents, timesteps, encoder_hidden_states,
                ).detach()

            stats = diffusion_dpo_loss(
                model_pred=model_pred.float(),
                ref_pred=ref_pred.float(),
                target=target.float(),
                beta=cfg.beta,
            )
            loss = stats["loss"]

            sft_loss_val = torch.tensor(0.0, device=self.device)
            if cfg.sft_weight > 0:
                # Compute MSE on winner only
                bsz = model_pred.shape[0] // 2
                sft_loss_val = diffusion_sft_loss(
                    model_pred[:bsz].float(), target[:bsz].float(),
                )
                loss = loss + cfg.sft_weight * sft_loss_val

        # 6. Backward + step (with optional accumulation)
        loss_scaled = loss / max(1, cfg.gradient_accumulation_steps)
        loss_scaled.backward()

        grad_norm = 0.0
        if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
            if cfg.max_grad_norm > 0:
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                grad_norm = float(gn) if isinstance(gn, torch.Tensor) else gn
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

        self.global_step += 1

        return DPOStepMetrics(
            loss=float(stats["loss"].detach()),
            raw_model_loss=float(stats["raw_model_loss"].detach()),
            raw_ref_loss=float(stats["raw_ref_loss"].detach()),
            model_diff=float(stats["model_diff"].detach()),
            ref_diff=float(stats["ref_diff"].detach()),
            implicit_acc=float(stats["implicit_acc"].detach()),
            sft_loss=float(sft_loss_val.detach()),
            grad_norm=grad_norm,
        )

    def _reference_forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the reference policy.

        If a separate ``ref_model`` was provided, use it directly. Otherwise
        assume the policy is a LoRA-wrapped backbone and disable adapters
        for the reference pass (PEFT convention).
        """
        if self.ref_model is not None:
            return self.forward_fn(
                self.ref_model, noisy_latents, timesteps, encoder_hidden_states,
            )
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return self.forward_fn(
                    self.model, noisy_latents, timesteps, encoder_hidden_states,
                )
        raise RuntimeError(
            "no ref_model and policy has no ``disable_adapter`` — "
            "cannot compute reference prediction"
        )
