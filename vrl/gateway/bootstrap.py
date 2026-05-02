"""Gateway bootstrap."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from vrl.config import EngineConfig
from vrl.gateway.state import GatewayRuntime

logger = logging.getLogger("vrl")


def create_gateway_runtime(config: EngineConfig) -> GatewayRuntime:
    """Assemble Gateway dependencies."""
    client = None
    if config.ipc.enabled:
        from vrl.ipc.client import EngineIPCClient

        client = EngineIPCClient(ipc_path=config.ipc.socket_path)
    return GatewayRuntime(config=config, engine_client=client)


def build_gateway_lifespan(runtime: GatewayRuntime):
    """FastAPI lifespan."""

    @asynccontextmanager
    async def lifespan(_app):
        device_str = (
            runtime.config.device.value
            if hasattr(runtime.config.device, "value")
            else str(runtime.config.device)
        )
        if runtime.engine_client is not None:
            await runtime.engine_client.start()
            logger.info("Gateway ready: device=%s, IPC connected", device_str)
        else:
            logger.info("Gateway ready: device=%s", device_str)
        yield
        if runtime.engine_client is not None:
            await runtime.engine_client.stop()
        logger.info("Shutting down gateway")

    return lifespan
