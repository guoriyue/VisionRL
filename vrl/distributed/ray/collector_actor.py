"""Ray actor wrapper for prompt-level rollout collection."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vrl.distributed.ray.actor import RayActorBase
from vrl.distributed.ray.utils import import_from_path
from vrl.rollouts.types import ExperienceBatch


@dataclass(slots=True)
class CollectorJob:
    """Prompt-level collect job dispatched to a Ray collector actor."""

    prompt_index: int
    prompt: str
    group_size: int
    collect_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


def _config_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_collector_factory(cfg: Any, family: str) -> Callable[..., Any]:
    factory = _config_get(cfg, "collector_factory")
    if factory is None:
        raise NotImplementedError(
            "RayCollectorActor needs a collector factory for P0. Provide "
            "`collector_factory` as a callable or import path on cfg. "
            f"No default collector builder is registered for family={family!r}."
        )
    if isinstance(factory, str):
        factory = import_from_path(factory)
    if not callable(factory):
        raise TypeError("collector_factory must be callable or an import path")
    return factory


def _invoke_factory(factory: Callable[..., Any], cfg: Any, family: str) -> Any:
    signature = inspect.signature(factory)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(cfg=cfg, family=family)

    kwargs: dict[str, Any] = {}
    if "cfg" in params:
        kwargs["cfg"] = cfg
    if "family" in params:
        kwargs["family"] = family
    if kwargs:
        return factory(**kwargs)

    positional = [
        p for p in params.values()
        if (
            p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    ]
    if len(positional) >= 2:
        return factory(cfg, family)
    if len(positional) == 1:
        return factory(cfg)
    return factory()


def _set_group_ids(batch: ExperienceBatch, prompt_index: int) -> None:
    group_ids = batch.group_ids
    try:
        group_ids[:] = prompt_index
        return
    except TypeError:
        pass

    if hasattr(group_ids, "fill_"):
        group_ids.fill_(prompt_index)
    elif hasattr(group_ids, "fill"):
        group_ids.fill(prompt_index)
    else:
        batch.group_ids = [prompt_index for _ in group_ids]


def _resolve_collect_result(result: Any) -> ExperienceBatch:
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


class RayCollectorActor(RayActorBase):
    """Ray actor that owns one collector instance and executes collect jobs."""

    def __init__(self, cfg: Any, family: str) -> None:
        self.cfg = cfg
        self.family = family
        factory = _resolve_collector_factory(cfg, family)
        self.collector = _invoke_factory(factory, cfg, family)

    def collect_jobs(self, jobs: list[CollectorJob]) -> list[ExperienceBatch]:
        batches: list[ExperienceBatch] = []
        for job in jobs:
            kwargs = dict(job.collect_kwargs)
            if job.seed is not None:
                kwargs["seed"] = job.seed
            result = self.collector.collect(
                [job.prompt],
                group_size=job.group_size,
                **kwargs,
            )
            batch = _resolve_collect_result(result)
            _set_group_ids(batch, job.prompt_index)
            batch.context = dict(batch.context)
            batch.context.setdefault("ray_prompt_index", job.prompt_index)
            batches.append(batch)
        return batches
