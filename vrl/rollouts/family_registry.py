"""Canonical rollout family registry.

The registry lives in ``vrl.rollouts`` because it wires training-time rollout
components: collector metadata, executor import paths, runtime builders, and
driver-side chunk gatherers. Distributed backends should consume the resolved
entry instead of branching on concrete model families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous"]


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Collector-facing family metadata shared by collector builders."""

    kind: CollectorKind
    config_cls: str
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutorKwargsMetadata:
    """Runtime executor kwargs that can be derived from a full rollout cfg."""

    include_sample_batch_size: bool = False
    include_reference_image: bool = False


@dataclass(frozen=True, slots=True)
class GathererMetadata:
    """Driver-side chunk gatherer construction metadata."""

    import_path: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutFamilyEntry:
    """Declarative binding for one canonical rollout family."""

    family: str
    task: str
    collector: CollectorMetadata
    local_executor_cls: str
    runtime_builder: str
    runtime_spec_extractor: str
    gatherer: GathererMetadata
    executor_kwargs: ExecutorKwargsMetadata = field(
        default_factory=ExecutorKwargsMetadata,
    )
    aliases: tuple[str, ...] = ()

    @property
    def executor_cls(self) -> str:
        """Import path consumed by serializable runtime specs."""

        return self.local_executor_cls


DIFFUSION_RETURN_ARTIFACTS = (
    "output",
    "rollout_trajectory_data",
    "trajectory_timesteps",
    "trajectory_latents",
    "denoising_env",
)

DIFFUSION_COMMON_SAMPLING_FIELDS = (
    "num_steps",
    "guidance_scale",
    "height",
    "width",
    "cfg",
    "sample_batch_size",
    "sde_window_size",
    "sde_window_range",
    "same_latent",
    "max_sequence_length",
    "noise_level",
    "return_kl",
)

DIFFUSION_VIDEO_SAMPLING_FIELDS = (
    *DIFFUSION_COMMON_SAMPLING_FIELDS,
    "num_frames",
)


FAMILY_REGISTRY: dict[str, RolloutFamilyEntry] = {
    "sd3_5": RolloutFamilyEntry(
        family="sd3_5",
        task="t2i",
        aliases=("sd3.5", "sd35"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collectors.configs:SD3_5CollectorConfig",
            request_prefix="sd3_5",
            default_task_type="text_to_image",
            error_prefix="SD3.5",
            sampling_fields=DIFFUSION_COMMON_SAMPLING_FIELDS,
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        local_executor_cls="vrl.models.families.sd3_5.executor:SD3_5PipelineExecutor",
        runtime_builder="vrl.models.families.sd3_5.builder:build_sd3_5_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.families.sd3_5.builder:extract_sd3_5_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.generation.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "sd3_5"},
        ),
        executor_kwargs=ExecutorKwargsMetadata(include_sample_batch_size=True),
    ),
    "wan_2_1": RolloutFamilyEntry(
        family="wan_2_1",
        task="t2v",
        aliases=("wan", "wan_2_1_1_3b", "wan_2_1_14b"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collectors.configs:Wan_2_1CollectorConfig",
            request_prefix="wan_2_1",
            default_task_type="text_to_video",
            error_prefix="Wan 2.1",
            sampling_fields=DIFFUSION_VIDEO_SAMPLING_FIELDS,
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        local_executor_cls="vrl.models.families.wan_2_1.executor:Wan_2_1PipelineExecutor",
        runtime_builder="vrl.models.families.wan_2_1.builder:build_wan_2_1_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.families.wan_2_1.builder:extract_wan_2_1_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.generation.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "wan_2_1"},
        ),
        executor_kwargs=ExecutorKwargsMetadata(include_sample_batch_size=True),
    ),
    "cosmos": RolloutFamilyEntry(
        family="cosmos",
        task="v2w",
        aliases=("cosmos_predict2", "cosmos_predict2_2b"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collectors.configs:CosmosPredict2CollectorConfig",
            request_prefix="cosmos",
            default_task_type="video2world",
            error_prefix="Cosmos",
            sampling_fields=(
                *DIFFUSION_VIDEO_SAMPLING_FIELDS,
                "fps",
            ),
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        local_executor_cls="vrl.models.families.cosmos.executor:CosmosPipelineExecutor",
        runtime_builder=(
            "vrl.models.families.cosmos.builder:build_cosmos_predict2_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.families.cosmos.builder:extract_cosmos_predict2_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.generation.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "cosmos", "respect_cfg_flag": False},
        ),
        executor_kwargs=ExecutorKwargsMetadata(
            include_sample_batch_size=True,
            include_reference_image=True,
        ),
    ),
    "janus_pro": RolloutFamilyEntry(
        family="janus_pro",
        task="ar_t2i",
        aliases=("janus", "janus_pro_1b"),
        collector=CollectorMetadata(
            kind="ar_discrete",
            config_cls="vrl.rollouts.collectors.configs:JanusProCollectorConfig",
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
        local_executor_cls="vrl.models.families.janus_pro.executor:JanusProPipelineExecutor",
        runtime_builder=(
            "vrl.models.families.janus_pro.builder:build_janus_pro_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.families.janus_pro.builder:extract_janus_pro_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.families.janus_pro.executor:JanusProChunkGatherer",
        ),
    ),
    "nextstep_1": RolloutFamilyEntry(
        family="nextstep_1",
        task="ar_t2i",
        aliases=("nextstep", "nextstep_1_1"),
        collector=CollectorMetadata(
            kind="ar_continuous",
            config_cls="vrl.rollouts.collectors.configs:NextStep1CollectorConfig",
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
        local_executor_cls="vrl.models.families.nextstep_1.executor:NextStep1PipelineExecutor",
        runtime_builder=(
            "vrl.models.families.nextstep_1.builder:build_nextstep_1_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.families.nextstep_1.builder:extract_nextstep_1_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.families.nextstep_1.executor:NextStep1ChunkGatherer",
        ),
    ),
}

_FAMILY_ALIASES: dict[str, str] = {
    alias: family
    for family, entry in FAMILY_REGISTRY.items()
    for alias in (family, *entry.aliases)
}


def normalize_rollout_family(family: str) -> str:
    """Return the canonical registry key for a rollout family or alias."""

    text = str(family)
    return _FAMILY_ALIASES.get(text, text)


def get_rollout_family_entry(family: str) -> RolloutFamilyEntry:
    """Return the canonical rollout family entry for ``family``."""

    normalized = normalize_rollout_family(family)
    try:
        return FAMILY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported rollout family: {family!r}; "
            f"registered={sorted(FAMILY_REGISTRY)}",
        ) from exc


def registered_rollout_families() -> tuple[str, ...]:
    """Return canonical rollout family keys."""

    return tuple(FAMILY_REGISTRY)


__all__ = [
    "DIFFUSION_COMMON_SAMPLING_FIELDS",
    "DIFFUSION_RETURN_ARTIFACTS",
    "DIFFUSION_VIDEO_SAMPLING_FIELDS",
    "FAMILY_REGISTRY",
    "CollectorKind",
    "CollectorMetadata",
    "ExecutorKwargsMetadata",
    "GathererMetadata",
    "RolloutFamilyEntry",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "registered_rollout_families",
]
