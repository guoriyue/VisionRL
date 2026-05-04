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


def extract_janus_pro_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice Janus-Pro runtime construction fields out of a whole RL cfg."""

    dtype = _cfg_path(cfg, "model.dtype", weight_dtype or "bfloat16")
    return RuntimeBuildSpec(
        model_name_or_path=str(
            _cfg_path(cfg, "model.path", "deepseek-ai/Janus-Pro-1B"),
        ),
        device=device,
        dtype=_dtype_to_config_string(dtype),
        backend_preference=("native",),
        task_variant="ar_t2i",
        use_lora=bool(_cfg_path(cfg, "model.use_lora", True)),
        lora_config={
            "rank": int(_cfg_path(cfg, "model.lora.rank", 32)),
            "alpha": int(_cfg_path(cfg, "model.lora.alpha", 64)),
            "dropout": float(_cfg_path(cfg, "model.lora.dropout", 0.0)),
            "target_modules": list(
                _cfg_path(cfg, "model.lora.target_modules", ("q_proj", "v_proj")),
            ),
            "init": str(_cfg_path(cfg, "model.lora.init", "gaussian")),
        },
        scheduler_config={
            "cfg_weight": float(_cfg_path(cfg, "sampling.cfg_weight", 5.0)),
            "temperature": float(_cfg_path(cfg, "sampling.temperature", 1.0)),
            "image_token_num": int(_cfg_path(cfg, "sampling.image_token_num", 576)),
        },
        extra={
            "freeze_vq": bool(_cfg_path(cfg, "model.freeze_vq", True)),
            "freeze_vision_encoder": bool(
                _cfg_path(cfg, "model.freeze_vision_encoder", True),
            ),
            "freeze_aligner": bool(_cfg_path(cfg, "model.freeze_aligner", True)),
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


_MISSING = object()


def _cfg_path(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


__all__ = ["build_janus_pro_runtime_bundle", "extract_janus_pro_runtime_spec"]
