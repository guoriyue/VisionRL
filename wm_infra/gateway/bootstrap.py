"""Gateway bootstrap helpers for assembling runtime dependencies."""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from wm_infra.backends import BackendRegistry, CosmosJobQueue, CosmosPredictBackend, MatrixGameBackend, RolloutBackend, WanJobQueue, WanVideoBackend
from wm_infra.config import EngineConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore
from wm_infra.engine.engine import create_async_engine
from wm_infra.engine.types import RolloutJob
from wm_infra.gateway.state import GatewayRuntime
from wm_infra.workloads.reinforcement_learning.runtime import ReinforcementLearningEnvManager

logger = logging.getLogger("wm_infra")

def build_rollout_job(request, config: EngineConfig) -> RolloutJob:
    """Translate the northbound rollout request into the engine-native job."""
    job = RolloutJob(
        job_id="",
        num_steps=request.num_steps,
        return_frames=request.return_frames,
        return_latents=request.return_latents,
        stream=request.stream,
    )

    if request.initial_latent is not None:
        job.initial_latent = torch.tensor(request.initial_latent, dtype=torch.float32)
    elif request.initial_observation_b64 is not None:
        img_bytes = base64.b64decode(request.initial_observation_b64)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        job.initial_observation = torch.from_numpy(img_np).permute(2, 0, 1)
    else:
        n = config.state_cache.num_latent_tokens
        d = config.dynamics.latent_token_dim
        job.initial_latent = torch.randn(n, d)

    if request.actions is not None:
        job.actions = torch.tensor(request.actions, dtype=torch.float32)

    return job


def build_default_store(config: EngineConfig) -> SampleManifestStore:
    """Create the sample manifest store used by Gateway-managed backends."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return SampleManifestStore(root)


def build_temporal_store(config: EngineConfig) -> TemporalStore:
    """Create the temporal entity store used by Gateway control-plane routes."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return TemporalStore(Path(root) / "temporal")


def create_gateway_runtime(
    config: EngineConfig,
    *,
    sample_store: SampleManifestStore | None = None,
    backend_registry: BackendRegistry | None = None,
    temporal_store: TemporalStore | None = None,
) -> GatewayRuntime:
    """Assemble static Gateway dependencies that do not require a running engine."""
    resolved_sample_store = sample_store or build_default_store(config)
    resolved_temporal_store = temporal_store or build_temporal_store(config)
    resolved_registry = backend_registry or BackendRegistry()
    temporal_env_manager = ReinforcementLearningEnvManager(resolved_temporal_store)
    return GatewayRuntime(
        config=config,
        sample_store=resolved_sample_store,
        temporal_store=resolved_temporal_store,
        backend_registry=resolved_registry,
        temporal_env_manager=temporal_env_manager,
    )


def build_gateway_lifespan(runtime: GatewayRuntime, *, execution_mode: str = "chunked"):
    """Create the FastAPI lifespan that starts/stops Gateway-managed resources."""

    @asynccontextmanager
    async def lifespan(_app):
        logger.info("Initializing temporal runtime engine...")
        runtime.engine = create_async_engine(runtime.config, execution_mode=execution_mode)
        runtime.engine.start()

        registry = runtime.backend_registry
        config = runtime.config

        if registry.get("rollout-engine") is None:
            registry.register(RolloutBackend(runtime.engine))
        if registry.get("matrix-game") is None:
            registry.register(MatrixGameBackend(runtime.engine))
        if registry.get("wan-video") is None:
            wan_root = config.controlplane.wan_output_root or str(Path(tempfile.gettempdir()) / "wm_infra_wan")
            registry.register(
                WanVideoBackend(
                    wan_root,
                    shell_runner=config.controlplane.wan_shell_runner,
                    shell_runner_timeout_s=config.controlplane.wan_shell_runner_timeout_s,
                    wan_admission_max_units=config.controlplane.wan_admission_max_units,
                    wan_admission_max_vram_gb=config.controlplane.wan_admission_max_vram_gb,
                    max_batch_size=config.controlplane.wan_max_batch_size,
                    batch_wait_ms=config.controlplane.wan_batch_wait_ms,
                    warm_pool_size=config.controlplane.wan_warm_pool_size,
                    prewarm_common_signatures=config.controlplane.wan_prewarm_common_signatures,
                    wan_engine_adapter=config.controlplane.wan_engine_adapter,
                    wan_repo_dir=config.controlplane.wan_repo_dir,
                    wan_conda_env=config.controlplane.wan_conda_env,
                    wan_ckpt_dir=config.controlplane.wan_ckpt_dir,
                    wan_i2v_diffusers_dir=config.controlplane.wan_i2v_diffusers_dir,
                    conda_sh_path=config.controlplane.conda_sh_path,
                )
            )
        wan_backend = registry.get("wan-video")

        if registry.get("cosmos-predict") is None:
            cosmos_root = config.controlplane.cosmos_output_root or str(Path(tempfile.gettempdir()) / "wm_infra_cosmos")
            registry.register(
                CosmosPredictBackend(
                    cosmos_root,
                    base_url=config.controlplane.cosmos_base_url,
                    api_key=config.controlplane.cosmos_api_key,
                    model_name=config.controlplane.cosmos_model_name,
                    shell_runner=config.controlplane.cosmos_shell_runner,
                    timeout_s=config.controlplane.cosmos_timeout_s,
                )
            )
        cosmos_backend = registry.get("cosmos-predict")

        if isinstance(wan_backend, WanVideoBackend):
            wan_queue_batch_size = wan_backend.queue_batch_size_limit(config.controlplane.wan_max_batch_size)
            runtime.wan_job_queue = WanJobQueue(
                execute_fn=wan_backend.execute_job,
                execute_many_fn=wan_backend.execute_job_batch,
                batch_key_fn=wan_backend.queue_batch_key,
                batch_select_fn=wan_backend.queue_batch_score,
                store=runtime.sample_store,
                queue_name="wan",
                max_queue_size=config.controlplane.wan_max_queue_size,
                max_concurrent=config.controlplane.wan_max_concurrent_jobs,
                max_batch_size=wan_queue_batch_size,
                batch_wait_ms=config.controlplane.wan_batch_wait_ms,
            )
            runtime.wan_job_queue.start()

        if isinstance(cosmos_backend, CosmosPredictBackend):
            runtime.cosmos_job_queue = CosmosJobQueue(
                execute_fn=cosmos_backend.execute_job,
                store=runtime.sample_store,
                queue_name="cosmos",
                max_queue_size=config.controlplane.cosmos_max_queue_size,
                max_concurrent=config.controlplane.cosmos_max_concurrent_jobs,
            )
            runtime.cosmos_job_queue.start()

        device_str = config.device.value if hasattr(config.device, "value") else str(config.device)
        logger.info(
            "Engine ready: device=%s, dynamics=%d params",
            device_str,
            sum(p.numel() for p in runtime.engine.engine.dynamics_model.parameters()),
        )
        yield
        logger.info("Shutting down engine")
        if runtime.wan_job_queue is not None:
            await runtime.wan_job_queue.stop()
            runtime.wan_job_queue = None
        if runtime.cosmos_job_queue is not None:
            await runtime.cosmos_job_queue.stop()
            runtime.cosmos_job_queue = None
        if runtime.engine is not None:
            await runtime.engine.stop()
            runtime.engine = None

    return lifespan
