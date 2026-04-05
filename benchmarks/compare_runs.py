#!/usr/bin/env python3
"""Compare structured benchmark result files honestly.

This script refuses to present apples-to-oranges numbers as if they were
comparable. If workload keys differ, it prints the mismatches and exits non-zero.
"""

from __future__ import annotations

import argparse
import sys

from wm_infra.benchmarking import comparable_run_pair, load_json


METRICS = [
    ("submit_mean_ms", ("summary", "latency", "submit", "mean_ms")),
    ("submit_p95_ms", ("summary", "latency", "submit", "p95_ms")),
    ("terminal_mean_ms", ("summary", "latency", "terminal", "mean_ms")),
    ("terminal_p95_ms", ("summary", "latency", "terminal", "p95_ms")),
    ("success_rate", ("summary", "success_rate")),
]


def _dig(payload, path):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two benchmark result JSON files")
    parser.add_argument("left")
    parser.add_argument("right")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = load_json(args.left)
    right = load_json(args.right)

    ok, mismatches = comparable_run_pair(left, right)
    if not ok:
        print("Runs are not comparable. Mismatched workload axes:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 2

    left_name = left.get("system", {}).get("name", args.left)
    right_name = right.get("system", {}).get("name", args.right)
    print(f"Comparable workload confirmed: {left.get('workload', {})}")
    print(f"{'metric':<20} {'left':>12} {'right':>12} {'delta(right-left)':>18}")
    for name, path in METRICS:
        lval = _dig(left, path)
        rval = _dig(right, path)
        delta = None if lval is None or rval is None else (float(rval) - float(lval))
        print(f"{name:<20} {str(lval):>12} {str(rval):>12} {str(round(delta, 3) if delta is not None else None):>18}")
    print(f"left={left_name}  right={right_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
