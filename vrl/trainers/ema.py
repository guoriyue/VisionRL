"""Exponential Moving Average for model parameters.

Ported from flow_grpo/ema.py.  Maintains a shadow copy of trainable
parameters and supports eval-time weight swap via copy_ema_to/copy_temp_to.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class EMAModuleWrapper:
    """EMA wrapper for any set of torch parameters.

    Typical usage::

        ema = EMAModuleWrapper(model.parameters(), decay=0.9, update_step_interval=8)
        # after each optimizer step:
        ema.step(model.parameters(), global_step)
        # for evaluation:
        ema.copy_ema_to(model.parameters(), store_temp=True)
        evaluate(model)
        ema.copy_temp_to(model.parameters())
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ) -> None:
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters: list[torch.Tensor] | None = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        """Warmup decay: ramps from ~0.1 to ``self.decay``."""
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay,
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int) -> None:
        """Update EMA parameters."""
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
                if param.requires_grad:
                    if ema_param.device == param.device:
                        ema_param.add_(one_minus_decay * (param - ema_param))
                    else:
                        param_copy = param.detach().to(ema_param.device)
                        param_copy.sub_(ema_param)
                        param_copy.mul_(one_minus_decay)
                        ema_param.add_(param_copy)
                        del param_copy

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Move EMA parameters to a device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        """Replace model parameters with EMA values; optionally save originals."""
        if store_temp:
            self.temp_stored_parameters = [p.detach().cpu() for p in parameters]

        parameters = list(parameters)
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            param.data.copy_(ema_param.to(param.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore original parameters from temporary storage."""
        assert self.temp_stored_parameters is not None
        for temp_param, param in zip(self.temp_stored_parameters, parameters, strict=True):
            param.data.copy_(temp_param.data)
        self.temp_stored_parameters = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", self.ema_parameters)
        self.to(self.device)
