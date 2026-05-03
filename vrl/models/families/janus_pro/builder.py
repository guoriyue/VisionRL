"""Janus-Pro runtime builder for Ray rollout workers."""

from __future__ import annotations

from typing import Any

from vrl.models.families.janus_pro.policy import JanusProConfig, JanusProPolicy
from vrl.models.runtime import RuntimeBuildSpec, RuntimeBundle


def build_janus_pro_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the Janus-Pro policy from a serializable runtime spec."""

    config = _janus_config_from_runtime_spec(spec)
    policy = JanusProPolicy(JanusProConfig(**config))
    return RuntimeBundle(
        policy=policy,
        trainable_modules={"policy": policy},
        scheduler=None,
        backend_kind="janus_pro",
        backend_handle=policy,
        runtime_caps={
            "supports_chunked_execution": True,
            "supports_token_logprobs": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "use_lora": spec.use_lora,
        },
    )


def _janus_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_path": spec.model_name_or_path,
        "dtype": _dtype_to_config_string(spec.dtype),
        "device": str(spec.device),
        "use_lora": bool(spec.use_lora),
    }

    if spec.lora_config:
        config.update(
            {
                "lora_rank": int(spec.lora_config["rank"]),
                "lora_alpha": int(spec.lora_config["alpha"]),
                "lora_target_modules": tuple(spec.lora_config["target_modules"]),
            },
        )
        if "dropout" in spec.lora_config:
            config["lora_dropout"] = float(spec.lora_config["dropout"])
        if "init" in spec.lora_config:
            config["lora_init"] = str(spec.lora_config["init"])

    if spec.scheduler_config:
        for key in ("cfg_weight", "temperature", "image_token_num"):
            if key in spec.scheduler_config:
                config[key] = spec.scheduler_config[key]

    for key in (
        "trust_remote_code",
        "freeze_vq",
        "freeze_vision_encoder",
        "freeze_aligner",
        "vq_latent_channels",
    ):
        if key in spec.extra:
            config[key] = spec.extra[key]

    return config


def _dtype_to_config_string(value: Any) -> str:
    text = str(value).removeprefix("torch.")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    return aliases.get(text.lower(), text)


__all__ = ["build_janus_pro_runtime_bundle"]
