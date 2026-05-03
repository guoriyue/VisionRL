"""Tests for diffusion policies as trainer-facing modules."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.diffusion import DiffusionPolicy, VideoGenerationRequest
from vrl.models.families.cosmos.predict2_policy import CosmosPredict2Policy
from vrl.models.families.sd3_5.policy import SD3_5Policy
from vrl.models.families.wan_2_1.diffusers_policy import WanT2VDiffusersPolicy


class _AdapterTransformer(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class _PolicyStub(DiffusionPolicy):
    family = "stub"

    def __init__(self) -> None:
        super().__init__()
        pipeline = SimpleNamespace(transformer=_AdapterTransformer())
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self.forward_models: list[Any] = []

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, negative_prompt, kwargs
        return {}

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        del request, encoded, kwargs
        return object()

    def forward_step(
        self,
        state: Any,
        step_idx: int,
        *,
        model: Any = None,
    ) -> dict[str, Any]:
        del state, step_idx
        self.forward_models.append(model)
        return {"noise_pred": torch.ones(1)}

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def restore_eval_state(
        self,
        batch_extras: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del batch_extras, batch_context, latents, step_idx
        return object()

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return None

    @property
    def backend_handle(self) -> Any:
        return self.pipeline


class _BackendPipelineStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.device = torch.device("cpu")


def test_diffusion_policy_registers_only_transformer_child() -> None:
    policy = _PolicyStub()

    assert isinstance(policy, nn.Module)
    assert dict(policy.named_children()) == {"transformer": policy.transformer}
    keys = set(policy.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith("pipeline.") for key in keys)


@pytest.mark.parametrize(
    "policy_cls",
    [SD3_5Policy, WanT2VDiffusersPolicy, CosmosPredict2Policy],
)
def test_concrete_diffusers_policies_register_only_transformer(
    policy_cls: type[DiffusionPolicy],
) -> None:
    pipeline = _BackendPipelineStub()
    policy = policy_cls(pipeline=pipeline, device=torch.device("cpu"))

    assert isinstance(policy, nn.Module)
    assert policy.pipeline is pipeline
    assert policy.transformer is pipeline.transformer
    assert dict(policy.named_children()) == {"transformer": pipeline.transformer}
    keys = set(policy.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith(("vae.", "text_encoder.", "_pipeline.")) for key in keys)


@pytest.mark.parametrize(
    "policy_cls",
    [SD3_5Policy, WanT2VDiffusersPolicy, CosmosPredict2Policy],
)
def test_concrete_diffusers_policies_keep_pipeline_transformer_in_sync(
    policy_cls: type[DiffusionPolicy],
) -> None:
    pipeline = _BackendPipelineStub()
    policy = policy_cls(pipeline=pipeline, device=torch.device("cpu"))
    replacement = nn.Linear(2, 2)

    policy._set_transformer(replacement)

    assert policy.transformer is replacement
    assert policy.pipeline.transformer is replacement
    assert dict(policy.named_children()) == {"transformer": replacement}


def test_forward_resolves_policy_self_to_registered_transformer() -> None:
    policy = _PolicyStub()

    policy.forward(object(), 0, model=policy)
    policy.forward(object(), 0)

    assert policy.forward_models == [policy.transformer, policy.transformer]


def test_replay_forward_uses_registered_transformer_for_policy_model() -> None:
    policy = _PolicyStub()
    batch = SimpleNamespace(
        observations=torch.zeros(1, 1, 2),
        extras={},
        context={},
    )

    policy.replay_forward(batch, 0, model=policy)

    assert policy.forward_models == [policy.transformer]


def test_disable_adapter_forwards_to_transformer_context() -> None:
    policy = _PolicyStub()

    with policy.disable_adapter():
        assert policy.transformer.disabled is True

    assert policy.transformer.disabled is False


@pytest.mark.parametrize("prefixed", [False, True])
def test_load_trainable_state_accepts_legacy_and_policy_keys(prefixed: bool) -> None:
    policy = _PolicyStub()
    replacement = {
        "weight": torch.full_like(policy.transformer.weight, 2.0),
        "bias": torch.full_like(policy.transformer.bias, 3.0),
    }
    state = (
        {f"transformer.{key}": value for key, value in replacement.items()}
        if prefixed
        else replacement
    )

    policy.load_trainable_state(state)

    assert torch.equal(policy.transformer.weight, replacement["weight"])
    assert torch.equal(policy.transformer.bias, replacement["bias"])


def test_load_trainable_state_rejects_all_unmatched_keys() -> None:
    policy = _PolicyStub()

    with pytest.raises(RuntimeError, match="did not match any transformer keys"):
        policy.load_trainable_state({"unknown": torch.ones(1)})
