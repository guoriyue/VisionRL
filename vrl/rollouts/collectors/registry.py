"""Explicit rollout collector registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vrl.engine.generation import RolloutBackend
from vrl.rollouts.collect import RolloutCollector
from vrl.rollouts.collectors.configs import (
    CosmosPredict2CollectorConfig,
    JanusProCollectorConfig,
    NextStep1CollectorConfig,
    SD3_5CollectorConfig,
    Wan_2_1CollectorConfig,
)
from vrl.rollouts.packers.ar_continuous import ARContinuousRolloutPacker
from vrl.rollouts.packers.ar_discrete import ARDiscreteRolloutPacker
from vrl.rollouts.packers.diffusion import DiffusionRolloutPacker
from vrl.rollouts.request_builders import (
    ARRequestBuilder,
    DiffusionRequestBuilder,
    RolloutRequestBuilder,
)
from vrl.rollouts.rewards import RewardScorer

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous"]

LAST_COLLECT_PHASES: dict[str, float] = {}


@dataclass(frozen=True, slots=True)
class CollectorRegistryEntry:
    """Declarative binding from explicit family name to collector components."""

    family: str
    task: str
    kind: CollectorKind
    config_cls: type
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    include_fps: bool = False
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


COLLECTOR_REGISTRY: dict[str, CollectorRegistryEntry] = {
    "sd3_5": CollectorRegistryEntry(
        family="sd3_5",
        task="t2i",
        kind="diffusion",
        config_cls=SD3_5CollectorConfig,
        request_prefix="sd3_5",
        default_task_type="text_to_image",
        error_prefix="SD3.5",
    ),
    "wan_2_1": CollectorRegistryEntry(
        family="wan_2_1",
        task="t2v",
        kind="diffusion",
        config_cls=Wan_2_1CollectorConfig,
        request_prefix="wan_2_1",
        default_task_type="text_to_video",
        error_prefix="Wan 2.1",
    ),
    "cosmos": CollectorRegistryEntry(
        family="cosmos",
        task="v2w",
        kind="diffusion",
        config_cls=CosmosPredict2CollectorConfig,
        request_prefix="cosmos",
        default_task_type="video2world",
        error_prefix="Cosmos",
        include_fps=True,
    ),
    "janus_pro": CollectorRegistryEntry(
        family="janus_pro",
        task="ar_t2i",
        kind="ar_discrete",
        config_cls=JanusProCollectorConfig,
        request_prefix="janus_pro",
        sampling_fields=(
            "cfg_weight",
            "temperature",
            "image_token_num",
            "image_size",
            "max_text_length",
        ),
        return_artifacts=("output", "token_ids", "token_log_probs"),
    ),
    "nextstep_1": CollectorRegistryEntry(
        family="nextstep_1",
        task="ar_t2i",
        kind="ar_continuous",
        config_cls=NextStep1CollectorConfig,
        request_prefix="nextstep_1",
        sampling_fields=(
            "cfg_scale",
            "num_flow_steps",
            "noise_level",
            "image_token_num",
            "image_size",
            "max_text_length",
            "rescale_to_unit",
        ),
        return_artifacts=("output", "rollout_trajectory_data"),
        metadata_key="rollout_metadata",
    ),
}


def build_rollout_collector(
    family: str,
    *,
    model: Any | None,
    reward_fn: Any | None,
    config: Any | None = None,
    runtime: RolloutBackend | None = None,
    reference_image: Any = None,
) -> RolloutCollector:
    """Build a rollout collector from an explicit family registry key."""

    entry = _entry_for(family)
    collector_config = _resolve_config(entry, config)
    request_builder = _build_request_builder(entry, collector_config)
    packer = _build_packer(entry)
    executor_kwargs = _build_executor_kwargs(entry, collector_config, reference_image)

    return RolloutCollector(
        model=model,
        config=collector_config,
        family=entry.family,
        task=entry.task,
        request_builder=request_builder,
        packer=packer,
        reward_scorer=RewardScorer(reward_fn),
        default_group_size=_default_group_size(entry, collector_config),
        runtime=runtime,
        executor_kwargs=executor_kwargs,
        phase_sink=LAST_COLLECT_PHASES,
    )


def collector_config_cls(family: str) -> type:
    """Return the config schema for an explicit family registry key."""

    return _entry_for(family).config_cls


def _entry_for(family: str) -> CollectorRegistryEntry:
    try:
        return COLLECTOR_REGISTRY[family]
    except KeyError as exc:
        raise NotImplementedError(
            f"no rollout collector registered for family={family!r}; "
            f"registered={sorted(COLLECTOR_REGISTRY)}",
        ) from exc


def _resolve_config(entry: CollectorRegistryEntry, config: Any | None) -> Any:
    if config is None:
        return entry.config_cls()
    if not isinstance(config, entry.config_cls):
        raise TypeError(
            f"{entry.family} collector requires {entry.config_cls.__name__}, "
            f"got {type(config).__name__}",
        )
    return config


def _build_request_builder(
    entry: CollectorRegistryEntry,
    config: Any,
) -> RolloutRequestBuilder:
    if entry.kind == "diffusion":
        if entry.request_prefix is None or entry.default_task_type is None:
            raise ValueError(f"{entry.family} diffusion registry entry is incomplete")
        return DiffusionRequestBuilder(
            family=entry.family,
            task=entry.task,
            request_prefix=entry.request_prefix,
            config=config,
            default_task_type=entry.default_task_type,
            include_fps=entry.include_fps,
        )
    if entry.kind in {"ar_discrete", "ar_continuous"}:
        if entry.request_prefix is None or not entry.sampling_fields:
            raise ValueError(f"{entry.family} AR registry entry is incomplete")
        return ARRequestBuilder(
            family=entry.family,
            task=entry.task,
            request_prefix=entry.request_prefix,
            config=config,
            sampling_fields=entry.sampling_fields,
            return_artifacts=entry.return_artifacts,
            metadata_key=entry.metadata_key,
        )
    raise AssertionError(f"unhandled collector kind: {entry.kind}")


def _build_packer(entry: CollectorRegistryEntry) -> Any:
    if entry.kind == "diffusion":
        if entry.error_prefix is None:
            raise ValueError(f"{entry.family} diffusion registry entry is incomplete")
        return DiffusionRolloutPacker(error_prefix=entry.error_prefix)
    if entry.kind == "ar_discrete":
        return ARDiscreteRolloutPacker()
    if entry.kind == "ar_continuous":
        return ARContinuousRolloutPacker()
    raise AssertionError(f"unhandled collector kind: {entry.kind}")


def _build_executor_kwargs(
    entry: CollectorRegistryEntry,
    config: Any,
    reference_image: Any,
) -> dict[str, Any]:
    if entry.kind != "diffusion":
        return {}
    kwargs: dict[str, Any] = {"sample_batch_size": config.sample_batch_size}
    if reference_image is not None:
        kwargs["reference_image"] = reference_image
    return kwargs


def _default_group_size(entry: CollectorRegistryEntry, config: Any) -> int:
    if entry.kind == "diffusion":
        return 1
    return int(config.n_samples_per_prompt)


__all__ = [
    "COLLECTOR_REGISTRY",
    "LAST_COLLECT_PHASES",
    "CollectorRegistryEntry",
    "build_rollout_collector",
    "collector_config_cls",
]
