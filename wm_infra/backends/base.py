"""Backend interface for producing control-plane samples."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wm_infra.controlplane.schemas import ProduceSampleRequest, SampleRecord


class ProduceSampleBackend(ABC):
    """Interface for a runtime/backend that can materialize a sample request."""

    backend_name: str

    @abstractmethod
    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        """Execute the request and return a populated sample record."""
