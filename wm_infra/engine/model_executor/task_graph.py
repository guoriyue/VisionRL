"""Task graph — DAG-based execution with optional CUDA stream parallelism.

Nodes represent units of work (stage forward calls, memory ops, etc.).
Edges represent data dependencies. Execution uses Kahn's topological sort.
On GPU, different stream IDs enable overlap; on CPU, execution is sequential
in topological order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class TaskNode:
    """One node in the task graph."""

    name: str
    fn: Callable[..., Any]
    stream_id: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskEdge:
    """Directed edge from *src* to *dst* (src must complete before dst)."""

    src: str
    dst: str


class TaskGraph:
    """DAG of tasks with topological execution.

    Parameters
    ----------
    use_cuda_streams : bool
        If True and CUDA is available, execute nodes on different streams
        for overlap. Otherwise, execute sequentially.
    """

    def __init__(self, use_cuda_streams: bool = False) -> None:
        self.use_cuda_streams = use_cuda_streams
        self._nodes: dict[str, TaskNode] = {}
        self._edges: list[TaskEdge] = []
        self._adj: dict[str, list[str]] = {}
        self._in_degree: dict[str, int] = {}

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def add_node(
        self,
        name: str,
        fn: Callable[..., Any],
        stream_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Add a task node to the graph."""
        if name in self._nodes:
            raise ValueError(f"Node {name!r} already exists")
        self._nodes[name] = TaskNode(name=name, fn=fn, stream_id=stream_id, kwargs=kwargs)
        self._adj.setdefault(name, [])
        self._in_degree.setdefault(name, 0)

    def add_edge(self, src: str, dst: str) -> None:
        """Add a dependency edge: *src* must finish before *dst* starts."""
        if src not in self._nodes:
            raise KeyError(f"Source node {src!r} not found")
        if dst not in self._nodes:
            raise KeyError(f"Destination node {dst!r} not found")
        self._edges.append(TaskEdge(src=src, dst=dst))
        self._adj[src].append(dst)
        self._in_degree[dst] = self._in_degree.get(dst, 0) + 1

    def topological_order(self) -> list[str]:
        """Return node names in topological order (Kahn's algorithm).

        Raises ``RuntimeError`` if the graph contains a cycle.
        """
        in_deg = dict(self._in_degree)
        queue: deque[str] = deque()
        for name in self._nodes:
            if in_deg.get(name, 0) == 0:
                queue.append(name)

        order: list[str] = []
        while queue:
            name = queue.popleft()
            order.append(name)
            for neighbor in self._adj.get(name, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            raise RuntimeError(
                f"Cycle detected: processed {len(order)} of {len(self._nodes)} nodes"
            )
        return order

    def execute(self) -> dict[str, Any]:
        """Execute all nodes in topological order.

        Returns a dict mapping node name -> result. On CPU (or when
        ``use_cuda_streams=False``), execution is strictly sequential.
        When CUDA streams are enabled, nodes on different streams may
        overlap (with event-based synchronization on edges).
        """
        order = self.topological_order()
        results: dict[str, Any] = {}

        if self.use_cuda_streams and _cuda_available():
            results = self._execute_with_streams(order)
        else:
            results = self._execute_sequential(order)

        return results

    # ------------------------------------------------------------------
    # Sequential (CPU) execution
    # ------------------------------------------------------------------

    def _execute_sequential(self, order: list[str]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name in order:
            node = self._nodes[name]
            results[name] = node.fn(**node.kwargs)
        return results

    # ------------------------------------------------------------------
    # CUDA stream execution
    # ------------------------------------------------------------------

    def _execute_with_streams(self, order: list[str]) -> dict[str, Any]:
        import torch

        # Build stream pool and event tracking
        streams: dict[int, torch.cuda.Stream] = {}
        events: dict[str, torch.cuda.Event] = {}
        results: dict[str, Any] = {}

        # Build reverse edge map: for each node, which nodes must complete first?
        deps: dict[str, list[str]] = {name: [] for name in self._nodes}
        for edge in self._edges:
            deps[edge.dst].append(edge.src)

        for name in order:
            node = self._nodes[name]
            sid = node.stream_id
            if sid not in streams:
                streams[sid] = torch.cuda.Stream()
            stream = streams[sid]

            # Synchronize on all dependency events
            for dep_name in deps[name]:
                if dep_name in events:
                    stream.wait_event(events[dep_name])

            with torch.cuda.stream(stream):
                results[name] = node.fn(**node.kwargs)

            # Record completion event
            event = torch.cuda.Event()
            event.record(stream)
            events[name] = event

        # Synchronize all streams
        for stream in streams.values():
            stream.synchronize()

        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check that the graph is a valid DAG (no cycles)."""
        try:
            self.topological_order()
            return True
        except RuntimeError:
            return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
