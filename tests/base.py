"""Shared test support for integration-heavy suites."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

import httpx
import pytest_asyncio
from asgi_lifespan import LifespanManager

from wm_infra.api.server import create_app
from wm_infra.config import (
    ControlPlaneConfig,
    DynamicsConfig,
    EngineConfig,
    ServerConfig,
    StateCacheConfig,
    TokenizerConfig,
)
from wm_infra.controlplane import SampleManifestStore, TemporalStore


ConfigMutator = Callable[[EngineConfig], None]


class BaseTestCase:
    """Provide shared builders without forcing fixture-heavy inheritance everywhere."""

    @staticmethod
    def build_engine_config(
        *,
        tmp_path: Path | None = None,
        api_key: str | None = None,
        controlplane: ControlPlaneConfig | None = None,
    ) -> EngineConfig:
        resolved_controlplane = (
            controlplane.model_copy(deep=True)
            if controlplane is not None
            else ControlPlaneConfig(wan_engine_adapter="stub")
        )
        if tmp_path is not None:
            cosmos_root = tmp_path / "cosmos"
            wan_root = tmp_path / "wan"
            if not resolved_controlplane.cosmos_output_root:
                resolved_controlplane.cosmos_output_root = str(cosmos_root)
            if not resolved_controlplane.wan_output_root:
                resolved_controlplane.wan_output_root = str(wan_root)

        return EngineConfig(
            device="cpu",
            dtype="float32",
            dynamics=DynamicsConfig(
                hidden_dim=64,
                num_heads=4,
                num_layers=2,
                action_dim=8,
                latent_token_dim=6,
                max_rollout_steps=16,
            ),
            tokenizer=TokenizerConfig(
                spatial_downsample=2,
                temporal_downsample=1,
                latent_channels=16,
                fsq_levels=[4, 4, 4, 3, 3, 3],
            ),
            state_cache=StateCacheConfig(
                max_batch_size=8,
                max_rollout_steps=16,
                latent_dim=6,
                num_latent_tokens=16,
                pool_size_gb=0.1,
            ),
            server=ServerConfig(api_key=api_key),
            controlplane=resolved_controlplane,
        )

    @staticmethod
    def load_module(relative_path: str, module_name: str):
        module_path = Path(__file__).resolve().parents[1] / relative_path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def wait_for_terminal_sample(
        self,
        client: httpx.AsyncClient,
        sample_id: str,
        *,
        timeout_s: float = 2.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            response = await client.get(f"/v1/samples/{sample_id}")
            assert response.status_code == 200
            last = response.json()
            if last["status"] in {"accepted", "failed", "rejected", "succeeded"}:
                return last
            await asyncio.sleep(0.05)
        raise AssertionError(f"sample {sample_id} did not reach terminal state; last={last}")

    @staticmethod
    def make_sample_store(root: Path) -> SampleManifestStore:
        return SampleManifestStore(root)

    @staticmethod
    def make_temporal_store(root: Path) -> TemporalStore:
        return TemporalStore(root / "temporal")


class BaseApiTestCase(BaseTestCase):
    """Pytest-friendly base that exposes a reusable ``client`` fixture."""

    def build_app(
        self,
        tmp_path: Path,
        *,
        config: EngineConfig | None = None,
        backend_registry=None,
        include_temporal_store: bool = True,
        sample_store_root: Path | None = None,
        sample_store: SampleManifestStore | None = None,
        temporal_store: TemporalStore | None = None,
    ):
        resolved_config = config or self.build_engine_config(tmp_path=tmp_path)
        manifest_root = sample_store_root or Path(resolved_config.controlplane.manifest_store_root or tmp_path)
        resolved_sample_store = sample_store or self.make_sample_store(manifest_root)
        resolved_temporal_store = temporal_store or (
            self.make_temporal_store(tmp_path) if include_temporal_store else None
        )
        return create_app(
            resolved_config,
            sample_store=resolved_sample_store,
            temporal_store=resolved_temporal_store,
            backend_registry=backend_registry,
        )

    @asynccontextmanager
    async def client_for_app(self, app) -> AsyncIterator[httpx.AsyncClient]:
        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as client:
                yield client

    @asynccontextmanager
    async def client_for_config(
        self,
        tmp_path: Path,
        *,
        config: EngineConfig | None = None,
        backend_registry=None,
        include_temporal_store: bool = True,
        sample_store_root: Path | None = None,
        sample_store: SampleManifestStore | None = None,
        temporal_store: TemporalStore | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        app = self.build_app(
            tmp_path,
            config=config,
            backend_registry=backend_registry,
            include_temporal_store=include_temporal_store,
            sample_store_root=sample_store_root,
            sample_store=sample_store,
            temporal_store=temporal_store,
        )
        async with self.client_for_app(app) as client:
            yield client

    @pytest_asyncio.fixture
    async def client(self, tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
        self.tmp_path = tmp_path
        async with self.client_for_config(tmp_path) as client:
            yield client


class BaseAppTestCase(BaseApiTestCase):
    """Centralize app and client construction for integration-style tests."""

    def build_app(
        self,
        tmp_path: Path,
        *,
        api_key: str | None = None,
        include_temporal_store: bool = False,
        config_mutator: ConfigMutator | None = None,
        backend_registry=None,
    ):
        config = self.build_engine_config(tmp_path=tmp_path, api_key=api_key)
        if config_mutator is not None:
            config_mutator(config)

        manifest_root = Path(config.controlplane.manifest_store_root or tmp_path)
        temporal_store = TemporalStore(tmp_path / "temporal") if include_temporal_store else None
        return create_app(
            config,
            sample_store=SampleManifestStore(manifest_root),
            temporal_store=temporal_store,
            backend_registry=backend_registry,
        )

    @asynccontextmanager
    async def make_client(
        self,
        tmp_path: Path,
        *,
        api_key: str | None = None,
        include_temporal_store: bool = False,
        config_mutator: ConfigMutator | None = None,
        backend_registry=None,
    ):
        app = self.build_app(
            tmp_path,
            api_key=api_key,
            include_temporal_store=include_temporal_store,
            config_mutator=config_mutator,
            backend_registry=backend_registry,
        )
        async with self.client_for_app(app) as client:
            yield client


class BaseClientTestCase(BaseAppTestCase):
    """Attach a default client to classes that primarily exercise the HTTP surface."""

    include_temporal_store = False

    @pytest_asyncio.fixture(autouse=True)
    async def _setup_client(self, tmp_path: Path):
        self.tmp_path = tmp_path
        async with self.make_client(
            tmp_path,
            include_temporal_store=self.include_temporal_store,
        ) as client:
            self.client = client
            yield
