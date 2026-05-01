"""Base utilities shared by Ray actors."""

from __future__ import annotations


class RayActorBase:
    """Debug helpers for Ray actors.

    This class intentionally contains no collector or trainer logic.
    """

    def get_node_ip(self) -> str:
        ray = __import__("ray")
        return str(ray.util.get_node_ip_address())

    def get_gpu_ids(self) -> list[int]:
        ray = __import__("ray")
        ids = ray.get_gpu_ids()
        out: list[int] = []
        for gpu_id in ids:
            try:
                out.append(int(gpu_id))
            except (TypeError, ValueError):
                # Some Ray accelerators may report non-integer IDs. Keep a
                # stable integer-free debug path by dropping unparseable IDs.
                continue
        return out
