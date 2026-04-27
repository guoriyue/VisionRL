"""Variance decomposition for noise_floor samples.csv.

Splits total reward variance into:
  * sigma_within  — within-group std (fixed prompt, fixed rollout seed, G parallel samples)
  * sigma_between — between-group same-prompt std (fixed prompt, different rollout seeds)
  * sigma_prompt  — across prompts, using per-prompt means (difficulty distribution)

Compares sigma_within vs the epoch-level reward_std logged by training, and
sigma_between vs the epoch-to-epoch reward_mean jitter, to verify whether
observed "training movement" is within sampling noise.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze(samples_csv: Path, train_csv: Path | None = None) -> None:
    # samples_csv rows: prompt_idx, prompt_orig_idx, target_text, group_idx,
    # rollout_seed, sample_idx, reward, r_ocr
    by_prompt_group: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_prompt: dict[int, list[float]] = defaultdict(list)
    targets: dict[int, str] = {}
    all_rewards: list[float] = []

    with samples_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = int(row["prompt_idx"])
            g = int(row["group_idx"])
            r = float(row["reward"])
            by_prompt_group[(p, g)].append(r)
            by_prompt[p].append(r)
            targets[p] = row["target_text"]
            all_rewards.append(r)

    print(f"Total samples: {len(all_rewards)}")
    print(f"Overall mean:  {_mean(all_rewards):.4f}")
    print(f"Overall std:   {_std(all_rewards):.4f}")
    print()

    # Within-group std (per prompt+group, then averaged across groups)
    within_stds = [_std(vals) for vals in by_prompt_group.values()]
    sigma_within = math.sqrt(sum(s * s for s in within_stds) / len(within_stds)) \
        if within_stds else 0.0

    # Between-group same-prompt: per prompt, compute std of group means
    between_group_stds: list[float] = []
    group_means_per_prompt: dict[int, list[float]] = defaultdict(list)
    for (p, g), vals in by_prompt_group.items():
        group_means_per_prompt[p].append(_mean(vals))
    for p, gm in group_means_per_prompt.items():
        between_group_stds.append(_std(gm))
    sigma_between = math.sqrt(
        sum(s * s for s in between_group_stds) / len(between_group_stds)
    ) if between_group_stds else 0.0

    # Between-prompt: std of per-prompt means
    prompt_means = [_mean(by_prompt[p]) for p in sorted(by_prompt)]
    sigma_prompt = _std(prompt_means)

    print("=== Variance decomposition ===")
    print(f"sigma_within  (within-group, 4 parallel samples):  {sigma_within:.4f}")
    print(f"sigma_between (same-prompt, different seeds):       {sigma_between:.4f}")
    print(f"sigma_prompt  (across 4 prompts' means):            {sigma_prompt:.4f}")
    print()

    print("=== Per-prompt breakdown ===")
    print(f"{'p':>3} {'target':<25} {'mean':>7} {'std':>7} {'groups':>7}")
    for p in sorted(by_prompt):
        vals = by_prompt[p]
        tgt = targets.get(p, "")[:24]
        gms = group_means_per_prompt[p]
        print(
            f"{p:>3} {tgt:<25} {_mean(vals):>7.4f} {_std(vals):>7.4f} "
            f"{len(gms):>7}"
        )
    print()

    print("=== Per-group means (rows=prompt, cols=group) ===")
    for p in sorted(group_means_per_prompt):
        gms = group_means_per_prompt[p]
        tgt = targets.get(p, "")[:20]
        print(f"p{p} {tgt:<22} " + " ".join(f"{v:.3f}" for v in gms))
    print()

    # Compare to training CSV if provided
    if train_csv and train_csv.exists():
        with train_csv.open() as f:
            reader = csv.DictReader(f)
            train_rows = list(reader)
        if train_rows:
            train_reward_means = [float(r["reward_mean"]) for r in train_rows]
            train_reward_stds = [float(r["reward_std"]) for r in train_rows]
            n_epochs = len(train_rows)
            rm_mean = _mean(train_reward_means)
            rm_std = _std(train_reward_means)  # epoch-to-epoch reward_mean jitter
            rs_mean = _mean(train_reward_stds)  # avg within-group std in training
            print(f"=== Training CSV reference ({n_epochs} epochs) ===")
            print(f"reward_mean:  mean={rm_mean:.4f}, std_across_epochs={rm_std:.4f}")
            print(f"reward_std:   mean_per_epoch={rs_mean:.4f}")
            print()
            print("=== Verdict ===")
            # Compare epoch-to-epoch reward_mean jitter vs baseline sigma_between
            ratio = rm_std / sigma_between if sigma_between > 1e-6 else float("inf")
            print(f"epoch-to-epoch reward_mean std     = {rm_std:.4f}")
            print(f"baseline sigma_between (sampling)  = {sigma_between:.4f}")
            print(f"ratio                              = {ratio:.2f}x")
            if ratio < 1.5:
                print("→ Observed training jitter is within sampling noise. No signal.")
            elif ratio < 3:
                print("→ Observed training jitter is borderline (1.5x-3x sampling noise).")
            else:
                print("→ Observed training jitter > 3x sampling noise — likely real signal.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_csv", type=Path)
    parser.add_argument(
        "--train-csv", type=Path, default=None,
        help="Optional training metrics.csv for apples-to-apples comparison.",
    )
    args = parser.parse_args()
    analyze(args.samples_csv, args.train_csv)


if __name__ == "__main__":
    main()
