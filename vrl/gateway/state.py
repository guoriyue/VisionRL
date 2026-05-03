"""Gateway runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fastapi import Request

from vrl.gateway.config import EngineConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

    from vrl.ipc.client import EngineIPCClient


@dataclass(slots=True)
class GatewayRuntime:
    """Gateway-scoped runtime dependencies."""

    config: EngineConfig
    engine_client: EngineIPCClient | None = None


def bind_gateway_runtime(app: FastAPI, runtime: GatewayRuntime) -> None:
    """Attach runtime to the FastAPI app."""
    app.state.gateway_runtime = runtime


def get_gateway_runtime(request: Request) -> GatewayRuntime:
    """Get runtime for the current request."""
    return cast("GatewayRuntime", request.app.state.gateway_runtime)
