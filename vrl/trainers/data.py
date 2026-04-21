"""Data loading utilities for RL training.

Ported from flow_grpo training scripts.  The key piece is the
DistributedKRepeatSampler which ensures each prompt appears exactly K
times across all GPUs — required for GRPO group-relative advantages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class PromptExample:
    """A single training example loaded from a JSONL prompt file."""

    prompt: str
    target_text: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    request_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_prompt_manifest(path: str | Path) -> list[PromptExample]:
    """Load prompt examples from a manifest file. Supports two formats:

    * ``.jsonl``: one JSON per line with explicit fields — native
      :class:`PromptExample` manifest.
    * ``.txt``:   one prompt per line with target in double quotes,
      matching flow_grpo's ``dataset/ocr/train.txt`` convention. The
      target is extracted via ``prompt.split('"')[1]``.
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        return list(JsonlPromptDataset(p).examples)
    if p.suffix == ".txt":
        examples: list[PromptExample] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('"')
                target = parts[1] if len(parts) >= 3 else ""
                examples.append(PromptExample(prompt=line, target_text=target))
        return examples
    raise ValueError(f"Unsupported manifest suffix: {p.suffix}")


class JsonlPromptDataset(Dataset):
    """Dataset that loads :class:`PromptExample` objects from a JSONL file.

    Each line must be a JSON object whose keys match the
    :class:`PromptExample` fields.  Only ``prompt`` is required; all
    other fields fall back to their dataclass defaults when absent.
    """

    def __init__(self, path: str | Path) -> None:
        self.examples: list[PromptExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append(PromptExample(**obj))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"prompt": ex.prompt, "metadata": ex.metadata, "example": ex}

    @staticmethod
    def collate_fn(examples: list[dict[str, Any]]) -> tuple[list[str], list[dict]]:
        return (
            [e["prompt"] for e in examples],
            [e["metadata"] for e in examples],
        )


class TextPromptDataset(Dataset):
    """Simple dataset that loads prompts from a text file (one per line)."""

    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples: list[dict[str, Any]]) -> tuple[list[str], list[dict]]:
        return (
            [e["prompt"] for e in examples],
            [e["metadata"] for e in examples],
        )


class DistributedKRepeatSampler(Sampler):
    """Sampler that repeats each prompt K times across all GPUs.

    For GRPO to work, we need K samples per prompt in each batch so we
    can compute per-prompt advantages.  This sampler:
    1. Selects M = (num_replicas * batch_size) / K unique prompts
    2. Repeats each K times
    3. Shuffles deterministically (synced across ranks via seed)
    4. Splits to each rank

    Yields lists of indices (one batch per iteration), infinitely.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k ({k}) must divide num_replicas*batch_size ({self.total_samples})"
        )
        self.m = self.total_samples // self.k  # unique prompts per iteration
        self.epoch = 0

    def __iter__(self):  # type: ignore[override]
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated = [idx for idx in indices for _ in range(self.k)]

            shuffled_order = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[i] for i in shuffled_order]

            per_rank = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                per_rank.append(shuffled[start : start + self.batch_size])

            yield per_rank[self.rank]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
