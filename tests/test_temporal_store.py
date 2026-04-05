"""Focused tests for durable temporal-store persistence behavior."""

from __future__ import annotations

import pytest

from wm_infra.controlplane.temporal import EpisodeCreate, TemporalStore


def test_temporal_store_put_is_atomic(tmp_path, monkeypatch):
    store = TemporalStore(tmp_path)
    episode = store.create_episode(EpisodeCreate(title="original"))
    episode_path = tmp_path / "episodes" / f"{episode.episode_id}.json"
    original_payload = episode_path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("replace failed")

    monkeypatch.setattr("wm_infra.controlplane.temporal.os.replace", boom)

    updated = episode.model_copy(update={"title": "updated"})
    with pytest.raises(RuntimeError, match="replace failed"):
        store.episodes.put(updated)

    assert episode_path.read_text(encoding="utf-8") == original_payload
    loaded = store.episodes.get(episode.episode_id)
    assert loaded is not None
    assert loaded.title == "original"


def test_temporal_store_skips_corrupted_records(tmp_path):
    store = TemporalStore(tmp_path)
    episode = store.create_episode(EpisodeCreate(title="valid"))

    broken_path = tmp_path / "episodes" / "broken.json"
    broken_path.write_text("{not valid json", encoding="utf-8")

    assert store.episodes.get("broken") is None
    listed_ids = [item.episode_id for item in store.episodes.list()]
    assert listed_ids == [episode.episode_id]
