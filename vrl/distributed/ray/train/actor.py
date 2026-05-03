"""Ray train actor wrapper for distributed rollout ownership."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from typing import Any

from vrl.distributed.ray.dependencies import current_gpu_ids, current_node_ip
from vrl.distributed.ray.module_loading import import_from_path


def _resolve_factory(train_config: Any) -> Callable[..., Any]:
    factory = _config_get(train_config, "trainer_factory")
    if factory is None:
        raise NotImplementedError(
            "RayTrainActor requires `trainer_factory` in train_config. "
            "The driver must not construct CUDA trainer objects for Ray mode.",
        )
    if isinstance(factory, str):
        factory = import_from_path(factory)
    if not callable(factory):
        raise TypeError("trainer_factory must be callable or an import path")
    return factory


def _config_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _invoke_factory(factory: Callable[..., Any], train_config: Any) -> Any:
    signature = inspect.signature(factory)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(train_config=train_config)
    if "train_config" in params:
        return factory(train_config=train_config)
    positional = [
        p for p in params.values()
        if (
            p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    ]
    if positional:
        return factory(train_config)
    return factory()


def _resolve_result(result: Any) -> Any:
    if not inspect.isawaitable(result):
        return result
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    if not loop.is_running():
        return loop.run_until_complete(result)

    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class RayTrainActor:
    """Actor that owns trainer, collector, reward, and train policy."""

    def __init__(self, train_config: Any | None = None) -> None:
        self.train_config = train_config
        self.trainer: Any | None = None

    def setup(self, train_config: Any | None = None) -> None:
        if train_config is not None:
            self.train_config = train_config
        if self.train_config is None:
            raise ValueError("RayTrainActor.setup requires train_config")
        factory = _resolve_factory(self.train_config)
        self.trainer = _invoke_factory(factory, self.train_config)

    def train_step(self, prompt_batch: list[Any]) -> Any:
        if self.trainer is None:
            self.setup()
        assert self.trainer is not None
        step = self.trainer.step
        return _resolve_result(step(prompt_batch))

    def metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "local"
            gpu_ids = []
        return {
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
            "trainer_loaded": self.trainer is not None,
        }


__all__ = ["RayTrainActor"]
