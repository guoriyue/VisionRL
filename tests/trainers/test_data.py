"""Tests for vrl.trainers.data (DistributedKRepeatSampler)."""

from __future__ import annotations

import pytest


class TestDistributedKRepeatSampler:
    def test_k_repeat_distribution(self) -> None:
        from torch.utils.data import TensorDataset
        import torch
        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        sampler = DistributedKRepeatSampler(
            dataset=dataset, batch_size=6, k=3, num_replicas=2, rank=0, seed=42
        )
        it = iter(sampler)
        batch = next(it)
        assert len(batch) == 6

    def test_rank_sync(self) -> None:
        """Both ranks should see the same unique prompts."""
        from torch.utils.data import TensorDataset
        import torch
        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        s0 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=0, seed=0)
        s1 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=1, seed=0)
        b0 = next(iter(s0))
        b1 = next(iter(s1))
        # Together they should have 8 items from 4 unique indices, each repeated 2x
        all_indices = b0 + b1
        assert len(all_indices) == 8
        unique = set(all_indices)
        assert len(unique) == 4
