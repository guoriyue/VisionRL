"""Analyze GPU util timeline + phase_times, split by epoch."""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

PERF_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/perf_v1")

# ---- 1. Parse gpu_util.csv ----
rows: list[tuple[int, int, int]] = []
with (PERF_DIR / "gpu_util.csv").open() as f:
    next(f)
    for line in f:
        p = line.strip().split(",")
        if len(p) < 4:
            continue
        rows.append((int(p[0]), int(p[1]), int(p[3])))

# Use memory as training-liveness signal (>500 MiB = model loaded)
start_idx = next(i for i, r in enumerate(rows) if r[2] > 500)
end_idx = len(rows) - 1
while end_idx > start_idx and rows[end_idx][2] <= 500:
    end_idx -= 1
train_rows = rows[start_idx : end_idx + 1]
t0_ms = train_rows[0][0]

# ---- 2. Parse launch.log: map wall-clock timestamps to epoch boundaries ----
log = (PERF_DIR / "launch.log").read_text()

def parse_ts(line: str) -> float | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)", line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()

wall_start = None
wall_epoch_end: list[tuple[int, float]] = []
for line in log.splitlines():
    ts = parse_ts(line)
    if ts is None:
        continue
    if wall_start is None and "Starting OCR GRPO training" in line:
        wall_start = ts
    m = re.search(r"Epoch (\d+) \|", line)
    if m:
        wall_epoch_end.append((int(m.group(1)), ts))

if wall_start is None:
    print("Could not find training start in log", file=sys.stderr)
    sys.exit(1)

# Estimate monitor t0 wall time: train_rows[0] assumed ~= wall_start (mem first rises)
# Map wall-clock epoch ends to monitor-relative ms
epoch_ends_ms = [
    (e, int((ts - wall_start) * 1000))
    for e, ts in wall_epoch_end
]

def rows_in_window(t_start_ms: int, t_end_ms: int):
    return [r for r in train_rows if t_start_ms <= (r[0] - t0_ms) <= t_end_ms]

# ---- 3. Per-epoch GPU util summary ----
print("=" * 74)
print("GPU UTILIZATION PER EPOCH")
print("=" * 74)
print(f"{'epoch':6s}  {'window (s)':18s}  {'mean':>6s}  {'median':>7s}  "
      f"{'%idle<10':>8s}  {'%busy>90':>8s}  {'mem_peak':>9s}")
prev_end = 0
for epoch_n, end_ms in epoch_ends_ms:
    win = rows_in_window(prev_end, end_ms)
    if not win:
        continue
    utils = [r[1] for r in win]
    mems = [r[2] for r in win]
    idle_pct = 100 * sum(1 for u in utils if u < 10) / len(utils)
    busy_pct = 100 * sum(1 for u in utils if u > 90) / len(utils)
    note = " (warmup)" if epoch_n == 0 else ""
    print(f"{epoch_n:<6d}  {prev_end/1000:5.1f}s..{end_ms/1000:5.1f}s{note:9s}  "
          f"{mean(utils):5.1f}%  {median(utils):6.0f}%  "
          f"{idle_pct:7.1f}%  {busy_pct:7.1f}%  {max(mems):7d} MiB")
    prev_end = end_ms

# ---- 4. Per-epoch idle stretches (stable state focus) ----
print()
print("IDLE STRETCHES IN STABLE EPOCHS (>= 300ms, util < 10):")
for epoch_n, end_ms in epoch_ends_ms:
    if epoch_n == 0:
        prev_end = end_ms
        continue
    win = rows_in_window(prev_end, end_ms)
    stretches: list[tuple[int, int]] = []
    i = 0
    while i < len(win):
        if win[i][1] < 10:
            j = i
            while j + 1 < len(win) and win[j + 1][1] < 10:
                j += 1
            dur = win[j][0] - win[i][0]
            if dur >= 300:
                stretches.append((win[i][0] - t0_ms, dur))
            i = j + 1
        else:
            i += 1
    print(f"\n  epoch {epoch_n} — {len(stretches)} idle stretches, "
          f"total {sum(d for _,d in stretches)/1000:.1f}s:")
    for start_ms, dur in sorted(stretches, key=lambda x: -x[1])[:8]:
        print(f"    t={start_ms/1000:6.1f}s ({dur}ms)")
    prev_end = end_ms

# ---- 5. Phase breakdown (already have this from log) ----
print()
print("=" * 74)
print("PHASE BREAKDOWN (stable state = epoch 2)")
print("=" * 74)
for m in re.finditer(
    r"phase_times\[step=(\d+)\] total=([\d.]+)s \| (.+)$", log, re.MULTILINE
):
    step = int(m.group(1))
    if step != 2:
        continue
    total = float(m.group(2))
    phases: dict[str, tuple[float, float]] = {}
    for pm in re.finditer(r"(\S+)=([\d.]+)s \(([\d.]+)%\)", m.group(3)):
        phases[pm.group(1)] = (float(pm.group(2)), float(pm.group(3)))
    print(f"total {total:.1f}s\n")
    coarse = [k for k in phases if not k.startswith("collect.")]
    for k in coarse:
        sec, pct = phases[k]
        print(f"  {k:20s}  {sec:6.2f}s  {pct:5.1f}%")
    sub = [k for k in phases if k.startswith("collect.")]
    if sub:
        print(f"  └── collect sub-breakdown:")
        for k in sub:
            sec, pct = phases[k]
            print(f"      {k:20s}  {sec:6.2f}s  {pct:5.1f}%")
    # Derived metrics
    print(f"\n  Derived:")
    train_sec = phases["evaluate"][0] + phases["backward"][0]
    collect_sec = phases["collect"][0]
    print(f"    training (eval+bwd):  {train_sec:.1f}s  ({100*train_sec/total:.1f}%)")
    print(f"    collection:           {collect_sec:.1f}s  ({100*collect_sec/total:.1f}%)")
    print(f"    eval/bwd ratio:       {phases['evaluate'][0]/phases['backward'][0]:.2f}")
    # Per-timestep (20 train steps × 20 denoise steps)
    N_DENOISE = 20
    N_TRAIN = 20  # timestep_fraction=0.99 ~ all
    print(f"    per denoise step:     {phases['collect.denoise_loop'][0]/N_DENOISE*1000:.0f} ms")
    print(f"    per train timestep:   {train_sec/N_TRAIN*1000:.0f} ms")
