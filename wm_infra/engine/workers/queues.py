"""Typed async queues for engine communication."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from wm_infra.engine._types import EntityRequest, StepResult

T = TypeVar("T")


class AsyncQueue(Generic[T]):
    """Simple typed wrapper around ``asyncio.Queue``."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: T) -> None:
        await self._queue.put(item)

    def put_nowait(self, item: T) -> None:
        self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()

    def get_nowait(self) -> T:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def drain(self) -> list[T]:
        """Non-blocking drain: pop all currently available items."""
        items: list[T] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items


class RequestQueue(AsyncQueue[EntityRequest]):
    """Queue for incoming entity requests."""
    pass


class ResultQueue(AsyncQueue[StepResult]):
    """Queue for outgoing step results."""
    pass
