"""Shared Gateway runtime state and request-scoped accessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fastapi import Request

from wm_infra.backends import BackendRegistry, SampleJobQueue
from wm_infra.config import EngineConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore
from wm_infra.engine.engine import AsyncWorldModelEngine
from wm_infra.workloads.reinforcement_learning.runtime import ReinforcementLearningEnvManager

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclass(slots=True)
class GatewayRuntime:
    """Owns Gateway-scoped runtime dependencies and lifecycle state."""

    config: EngineConfig
    sample_store: SampleManifestStore
    temporal_store: TemporalStore
    backend_registry: BackendRegistry
    temporal_env_manager: ReinforcementLearningEnvManager
    engine: AsyncWorldModelEngine | None = None
    wan_job_queue: SampleJobQueue | None = None
    cosmos_job_queue: SampleJobQueue | None = None


def bind_gateway_runtime(app: "FastAPI", runtime: GatewayRuntime) -> None:
    """Attach the assembled Gateway runtime to the FastAPI app."""
    app.state.gateway_runtime = runtime


def get_gateway_runtime(request: Request) -> GatewayRuntime:
    """Return the per-app Gateway runtime for the current request."""
    return cast(GatewayRuntime, request.app.state.gateway_runtime)
