"""Network-aware helpers for Ray placement decisions."""

from __future__ import annotations

import socket


def sort_node_gpu_key(item: tuple[int, str, int]) -> tuple[list[int], int, int]:
    """Stable sort key for placement bundles.

    Adapted from slime's Ray placement sorting: sort by node IP, then GPU ID,
    then bundle index. The bundle index tie-breaker keeps CPU-only local-mode
    tests deterministic.
    """

    index, node_identifier, gpu_id = item
    try:
        node_ip_parts = [int(part) for part in node_identifier.split(".")]
    except ValueError:
        try:
            resolved = socket.gethostbyname(node_identifier)
            node_ip_parts = [int(part) for part in resolved.split(".")]
        except (socket.gaierror, TypeError):
            node_ip_parts = [ord(char) for char in node_identifier]
    return node_ip_parts, gpu_id, index
