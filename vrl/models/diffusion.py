"""Family adapter contract — the single protocol for diffusion RL.

The RL loop owner is ``vrl/rollouts/`` (collectors). The adapter exposes
inference primitives + opaque state projection helpers; the collector
owns the loop, the SDE step, the log-prob accounting, the reward
scoring, and the ExperienceBatch packing. There is NO default loop in
this contract — ``inference()`` was removed deliberately so we don't
maintain two parallel loops (one in ``models``, one in ``rollouts``).

Inference primitives:

    encode_prompt(prompt, neg, **kw)        -> dict[str, Tensor]
    prepare_sampling(request, encoded, **kw) -> SamplingState        (per-family local)
    forward_step(state, step_idx) -> dict[str, Tensor] (one fwd + CFG; no scheduler step)
    decode_latents(latents)                  -> Tensor

Boundary helpers — make ``SamplingState`` opaque to the collector:

    export_batch_context(state)            -> dict   (scalar/shared metadata for ExperienceBatch.context)
    export_training_extras(state)            -> dict   (per-sample tensors for ExperienceBatch.extras)
    restore_eval_state(extras, context, latents, step_idx) -> SamplingState  (rebuild for the eval forward path)

Backend ownership (called by the family builder, not the collector):

    @classmethod from_spec(spec)             -> Self
    apply_lora(spec)                         -> None
    enable_full_finetune()                   -> None
    torch_compile_transformer(mode)          -> None
    set_num_steps(n)                         -> None
    @property trainable_modules / scheduler / backend_handle

This is the sole model contract; the engine consumes it directly.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass(slots=True)
class VideoGenerationRequest:
    """Per-request inference parameters. Backend-agnostic; family-specific
    knobs go in ``extra``.
    """

    prompt: str = ""
    negative_prompt: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    width: int = 1024
    height: int = 640
    frame_count: int = 16
    num_steps: int = 35
    guidance_scale: float = 5.0
    high_noise_guidance_scale: float | None = None
    seed: int | None = None
    model_name: str = ""
    model_size: str = "A14B"
    ckpt_dir: str | None = None
    fps: int = 16
    sample_solver: str = "dpmpp"
    shift: float = 1.0
    t5_cpu: bool = True
    convert_model_dtype: bool = True
    offload_model: bool = False
    action_sequence: list[list[float]] | None = None
    action_dim: int | None = None
    action_conditioning_mode: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


class DiffusionPolicy(nn.Module, ABC):
    """Single protocol for visual-generation diffusion families on the RL path."""

    family: str = "diffusion"

    async def load(self) -> None:
        """Load heavy modules. Default no-op for adapters constructed eagerly."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"family": self.family}

    # -- inference primitives -----------------------------------------------

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt (and optional negative) into embedding tensors."""

    @abstractmethod
    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Build the per-family ``SamplingState`` for a denoise loop.

        The returned object is a private dataclass defined inside the adapter
        file. Collectors MUST NOT introspect it — they go through the
        ``export_*`` / ``restore_eval_state`` helpers below.
        """

    @abstractmethod
    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one transformer forward (with optional CFG batch concat).

        Returns at least ``{"noise_pred": Tensor}``. NO scheduler step happens
        here — the collector owns scheduler.step / SDE / log_prob.
        """

    def forward(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one trainable denoise transformer step.

        This is not a full rollout loop. Executors and collectors own sampling,
        scheduler stepping, reward, and rollout artifact packing.
        """

        return self.forward_step(state, step_idx)

    @abstractmethod
    def decode_latents(self, latents: Any) -> Any:
        """Decode latents to a frame tensor (image: [B,C,H,W]; video: [B,C,T,H,W])."""

    # -- collector boundary -------------------------------------------------
    # These are non-abstract on the base so callers can instantiate stub
    # adapters; concrete RL adapters MUST override all three.

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        """Project SamplingState into the read-only dict packed into
        ``ExperienceBatch.context``.

        Implementations return scalar / shared metadata — guidance_scale,
        do_cfg flag, model_family, plus family-specific shared tensors
        (e.g. cosmos masks) the eval path needs to rebuild state.
        """
        raise NotImplementedError

    def export_training_extras(self, state: Any) -> dict[str, Any]:
        """Project SamplingState into the per-sample tensor dict packed
        into ``ExperienceBatch.extras``.

        Implementations return prompt embeds and any family-specific
        per-sample tensors (e.g. cosmos init_latents). The collector
        merges this dict with its own per-step tensors (log_probs,
        timesteps, kl) when assembling the batch.
        """
        raise NotImplementedError

    def restore_eval_state(
        self,
        batch_extras: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        """Rebuild a ``SamplingState`` from a batch slice for the eval
        forward path.

        The replay_forward default impl calls this; collectors do not call
        it directly anymore.
        """
        raise NotImplementedError

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Train-time replay: rebuild SamplingState + run one transformer fwd.

        Default impl: ``restore_eval_state`` -> ``forward_step(state, 0)``.

        SD3 and Wan diffusers pack their eval-path timesteps as ``[1, B]`` and
        index with ``step_idx=0`` inside ``forward_step``, so the default's
        hard-coded 0 is correct for them.

        Cosmos overrides this to forward the real ``timestep_idx`` because
        its ``forward_step`` indexes ``state.scheduler.sigmas[step_idx]``.

        Wan official has no eval path today (``restore_eval_state`` raises).
        """
        state = self.restore_eval_state(
            batch.extras,
            batch.context,
            batch.observations[:, timestep_idx],
            timestep_idx,
        )
        return self.forward(state, 0)

    # -- trainer-facing module helpers -------------------------------------

    def _require_transformer(self) -> Any:
        """Return the registered trainable transformer."""

        transformer = getattr(self, "transformer", None)
        if transformer is None:
            raise RuntimeError(
                f"{type(self).__name__} has no registered trainable transformer",
            )
        return transformer

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable LoRA/adapters on the registered transformer, when available."""

        transformer = self._require_transformer()
        disable = getattr(transformer, "disable_adapter", None)
        if not callable(disable):
            raise RuntimeError(
                f"{type(transformer).__name__} does not expose disable_adapter()",
            )
        return disable()

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load trainable transformer weights from policy-prefixed keys."""

        transformer = self._require_transformer()
        state = dict(state_dict)
        if not state:
            raise ValueError("load_trainable_state received an empty state dict")
        prefix = "transformer."
        bad_keys = [key for key in state if not key.startswith(prefix)]
        if bad_keys:
            raise ValueError(
                "load_trainable_state only accepts policy keys prefixed with "
                f"{prefix!r}; got {bad_keys}",
            )
        state = {
            key[len(prefix):]: value
            for key, value in state.items()
        }
        if not state:
            raise ValueError("load_trainable_state requires transformer.* keys")
        return transformer.load_state_dict(state, strict=True)

    # -- backend ownership (called by builder, NOT by collectors) ----------

    @classmethod
    def from_spec(cls, spec: Any) -> DiffusionPolicy:  # pragma: no cover (abstract)
        """Load the backend (pipeline / native modules) from a spec."""
        raise NotImplementedError

    def apply_lora(self, spec: Any) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def enable_full_finetune(self) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def torch_compile_transformer(self, mode: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def set_num_steps(self, n: int) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def trainable_modules(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @property
    def scheduler(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def backend_handle(self) -> Any:  # pragma: no cover
        raise NotImplementedError
