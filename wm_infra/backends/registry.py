"""Backend registry for control-plane sample production."""

from __future__ import annotations

from wm_infra.backends.base import ProduceSampleBackend


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, ProduceSampleBackend] = {}

    def register(self, backend: ProduceSampleBackend) -> None:
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str) -> ProduceSampleBackend | None:
        return self._backends.get(backend_name)

    def names(self) -> list[str]:
        return sorted(self._backends.keys())
