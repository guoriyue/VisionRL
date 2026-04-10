"""Gateway bootstrap helpers for assembling runtime dependencies."""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from wm_infra.config import EngineConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore
from wm_infra.gateway.state import GatewayRuntime

logger = logging.getLogger("wm_infra")


def build_default_store(config: EngineConfig) -> SampleManifestStore:
    """Create the sample manifest store used by the gateway."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return SampleManifestStore(root)


def build_temporal_store(config: EngineConfig) -> TemporalStore:
    """Create the temporal entity store used by gateway control-plane routes."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return TemporalStore(Path(root) / "temporal")


def create_gateway_runtime(
    config: EngineConfig,
    *,
    sample_store: SampleManifestStore | None = None,
    temporal_store: TemporalStore | None = None,
) -> GatewayRuntime:
    """Assemble static Gateway dependencies."""
    return GatewayRuntime(
        config=config,
        sample_store=sample_store or build_default_store(config),
        temporal_store=temporal_store or build_temporal_store(config),
    )


def build_gateway_lifespan(runtime: GatewayRuntime):
    """Create the FastAPI lifespan."""

    @asynccontextmanager
    async def lifespan(_app):
        device_str = (
            runtime.config.device.value
            if hasattr(runtime.config.device, "value")
            else str(runtime.config.device)
        )
        logger.info("Gateway ready: device=%s", device_str)
        yield
        logger.info("Shutting down gateway")

    return lifespan
