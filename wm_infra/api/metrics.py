"""Prometheus metrics for world model serving.

Exposes counters, histograms, and gauges for monitoring request
throughput, latency, batching efficiency, and resource usage.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ─── Request-level metrics ───

REQUEST_TOTAL = Counter(
    "wm_request_total",
    "Total rollout requests",
    ["status"],
)

REQUEST_DURATION = Histogram(
    "wm_request_duration_seconds",
    "End-to-end rollout request duration",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

API_AUTH_FAILURES = Counter(
    "wm_api_auth_failures_total",
    "API authentication failures",
    ["endpoint"],
)

