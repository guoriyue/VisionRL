"""Family-specific Ray rollout runtime construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vrl.distributed.ray.config import DistributedRolloutConfig
from vrl.distributed.ray.spec import RolloutRuntimeSpec
from vrl.engine.generation.gather import ChunkGatherer, DiffusionChunkGatherer


@dataclass(frozen=True, slots=True)
class RayRolloutRuntimeInputs:
    """Serializable worker spec plus driver-side pure gatherer."""

    runtime_spec: RolloutRuntimeSpec
    gatherer: ChunkGatherer


def build_family_ray_rollout_runtime_inputs(
    cfg: Any,
    family: str,
    *,
    weight_dtype: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
    policy_version: int = 0,
) -> RayRolloutRuntimeInputs | None:
    """Build Ray launcher inputs for a known model family.

    Returns ``None`` for non-Ray rollout configs so training scripts can share
    one call site for local and Ray backends.
    """

    rollout_config = DistributedRolloutConfig.from_cfg(cfg)
    if rollout_config.backend != "ray":
        return None

    normalized_family = _normalize_family(family)
    kwargs = dict(executor_kwargs or {})
    rollout_device = "cuda" if rollout_config.gpus_per_worker > 0 else "cpu"
    dtype_name = _dtype_to_string(weight_dtype)

    if normalized_family == "sd3_5":
        from vrl.models.families.sd3_5.builder import extract_sd3_5_runtime_spec

        return RayRolloutRuntimeInputs(
            runtime_spec=RolloutRuntimeSpec(
                family="sd3_5",
                task="t2i",
                build_spec=_runtime_build_spec_payload(
                    extract_sd3_5_runtime_spec(cfg, rollout_device, dtype_name),
                ),
                executor_kwargs=kwargs,
                policy_version=policy_version,
                runtime_builder=(
                    "vrl.models.families.sd3_5.builder:"
                    "build_sd3_5_runtime_bundle"
                ),
                executor_cls=(
                    "vrl.models.families.sd3_5.executor:"
                    "SD3_5PipelineExecutor"
                ),
            ),
            gatherer=DiffusionChunkGatherer(model_family="sd3_5"),
        )

    if normalized_family == "wan_2_1":
        from vrl.models.families.wan_2_1.builder import extract_wan_2_1_runtime_spec

        return RayRolloutRuntimeInputs(
            runtime_spec=RolloutRuntimeSpec(
                family="wan_2_1",
                task="t2v",
                build_spec=_runtime_build_spec_payload(
                    extract_wan_2_1_runtime_spec(cfg, rollout_device, dtype_name),
                ),
                executor_kwargs=kwargs,
                policy_version=policy_version,
                runtime_builder=(
                    "vrl.models.families.wan_2_1.builder:"
                    "build_wan_2_1_runtime_bundle"
                ),
                executor_cls=(
                    "vrl.models.families.wan_2_1.executor:"
                    "Wan_2_1PipelineExecutor"
                ),
            ),
            gatherer=DiffusionChunkGatherer(model_family="wan_2_1"),
        )

    if normalized_family == "cosmos":
        from vrl.models.families.cosmos.builder import (
            extract_cosmos_predict2_runtime_spec,
        )

        reference_image = _cfg_path(cfg, "model.reference_image", None)
        if reference_image:
            kwargs.setdefault("reference_image", str(reference_image))
        return RayRolloutRuntimeInputs(
            runtime_spec=RolloutRuntimeSpec(
                family="cosmos",
                task="v2w",
                build_spec=_runtime_build_spec_payload(
                    extract_cosmos_predict2_runtime_spec(
                        cfg, rollout_device, dtype_name,
                    ),
                ),
                executor_kwargs=kwargs,
                policy_version=policy_version,
                runtime_builder=(
                    "vrl.models.families.cosmos.builder:"
                    "build_cosmos_predict2_runtime_bundle"
                ),
                executor_cls=(
                    "vrl.models.families.cosmos.executor:"
                    "CosmosPipelineExecutor"
                ),
            ),
            gatherer=DiffusionChunkGatherer(
                model_family="cosmos",
                respect_cfg_flag=False,
            ),
        )

    if normalized_family == "janus_pro":
        from vrl.models.families.janus_pro.executor import JanusProChunkGatherer

        return RayRolloutRuntimeInputs(
            runtime_spec=RolloutRuntimeSpec(
                family="janus_pro",
                task="ar_t2i",
                model_config=_janus_model_config(cfg, device=rollout_device),
                executor_kwargs=kwargs,
                policy_version=policy_version,
                executor_factory=(
                    "vrl.models.families.janus_pro.executor:"
                    "build_janus_pro_executor_from_runtime_spec"
                ),
            ),
            gatherer=JanusProChunkGatherer(),
        )

    if normalized_family == "nextstep_1":
        from vrl.models.families.nextstep_1.executor import NextStep1ChunkGatherer

        return RayRolloutRuntimeInputs(
            runtime_spec=RolloutRuntimeSpec(
                family="nextstep_1",
                task="ar_t2i",
                model_config=_nextstep_1_model_config(cfg, device=rollout_device),
                executor_kwargs=kwargs,
                policy_version=policy_version,
                executor_factory=(
                    "vrl.models.families.nextstep_1.executor:"
                    "build_nextstep_1_executor_from_runtime_spec"
                ),
            ),
            gatherer=NextStep1ChunkGatherer(),
        )

    raise ValueError(f"unsupported Ray rollout family: {family!r}")


def _runtime_build_spec_payload(spec: Any) -> dict[str, Any]:
    payload = asdict(spec)
    payload["device"] = _device_to_string(payload["device"])
    payload["dtype"] = _dtype_to_string(payload["dtype"])
    return payload


def _janus_model_config(cfg: Any, *, device: str) -> dict[str, Any]:
    return {
        "model_path": str(_cfg_path(cfg, "model.path", "deepseek-ai/Janus-Pro-1B")),
        "dtype": str(_cfg_path(cfg, "model.dtype", "bfloat16")),
        "use_lora": bool(_cfg_path(cfg, "model.use_lora", True)),
        "lora_rank": int(_cfg_path(cfg, "model.lora.rank", 32)),
        "lora_alpha": int(_cfg_path(cfg, "model.lora.alpha", 64)),
        "lora_dropout": float(_cfg_path(cfg, "model.lora.dropout", 0.0)),
        "lora_target_modules": list(
            _cfg_path(cfg, "model.lora.target_modules", ("q_proj", "v_proj")),
        ),
        "lora_init": str(_cfg_path(cfg, "model.lora.init", "gaussian")),
        "cfg_weight": float(_cfg_path(cfg, "sampling.cfg_weight", 5.0)),
        "temperature": float(_cfg_path(cfg, "sampling.temperature", 1.0)),
        "image_token_num": int(_cfg_path(cfg, "sampling.image_token_num", 576)),
        "device": device,
        "freeze_vq": bool(_cfg_path(cfg, "model.freeze_vq", True)),
        "freeze_vision_encoder": bool(
            _cfg_path(cfg, "model.freeze_vision_encoder", True),
        ),
        "freeze_aligner": bool(_cfg_path(cfg, "model.freeze_aligner", True)),
    }


def _nextstep_1_model_config(cfg: Any, *, device: str) -> dict[str, Any]:
    return {
        "model_path": str(_cfg_path(cfg, "model.path", "stepfun-ai/NextStep-1.1")),
        "vae_path": str(
            _cfg_path(cfg, "model.vae_path", "stepfun-ai/NextStep-1-f8ch16-Tokenizer"),
        ),
        "dtype": str(_cfg_path(cfg, "model.dtype", "bfloat16")),
        "device": device,
        "use_lora": bool(_cfg_path(cfg, "model.use_lora", True)),
        "lora_rank": int(_cfg_path(cfg, "model.lora.rank", 32)),
        "lora_alpha": int(_cfg_path(cfg, "model.lora.alpha", 64)),
        "lora_dropout": float(_cfg_path(cfg, "model.lora.dropout", 0.0)),
        "lora_target_modules": list(
            _cfg_path(cfg, "model.lora.target_modules", ("q_proj", "v_proj")),
        ),
        "lora_init": str(_cfg_path(cfg, "model.lora.init", "gaussian")),
        "cfg_scale": float(_cfg_path(cfg, "sampling.cfg_scale", 4.5)),
        "num_flow_steps": int(_cfg_path(cfg, "sampling.num_flow_steps", 20)),
        "noise_level": float(_cfg_path(cfg, "sampling.noise_level", 1.0)),
        "image_token_num": int(_cfg_path(cfg, "sampling.image_token_num", 1024)),
        "image_size": int(_cfg_path(cfg, "sampling.image_size", 256)),
        "freeze_vae": bool(_cfg_path(cfg, "model.freeze_vae", True)),
        "freeze_image_head": bool(_cfg_path(cfg, "model.freeze_image_head", False)),
        "gradient_checkpointing": bool(
            _cfg_path(cfg, "actor.gradient_checkpointing", True),
        ),
    }


def _normalize_family(family: str) -> str:
    aliases = {
        "wan": "wan_2_1",
        "cosmos_predict2": "cosmos",
        "nextstep": "nextstep_1",
    }
    return aliases.get(str(family), str(family))


def _device_to_string(value: Any) -> str:
    return str(value)


def _dtype_to_string(value: Any) -> str:
    text = str(value)
    return text.removeprefix("torch.")


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


__all__ = [
    "RayRolloutRuntimeInputs",
    "build_family_ray_rollout_runtime_inputs",
]
