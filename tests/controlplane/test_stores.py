"""Tests for sample manifest store and temporal store durability."""

from __future__ import annotations

import pytest

from wm_infra.controlplane import ExperimentRef, SampleManifestStore, SampleRecord, SampleSpec, SampleStatus, TaskType
from wm_infra.controlplane.temporal import EpisodeCreate, TemporalStore


# ---------------------------------------------------------------------------
# Sample manifest store
# ---------------------------------------------------------------------------


def test_sample_manifest_store_put_writes_clean_final_file(tmp_path):
    store = SampleManifestStore(tmp_path)
    record = SampleRecord(
        sample_id="sample_atomic",
        task_type=TaskType.TEMPORAL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.SUCCEEDED,
        experiment=ExperimentRef(experiment_id="exp_atomic"),
        sample_spec=SampleSpec(prompt="atomic write"),
    )

    store.put(record)

    final_path = tmp_path / "samples" / "exp_atomic" / "sample_atomic.json"
    assert final_path.exists()
    assert store.get("sample_atomic") is not None
    assert [path.name for path in final_path.parent.iterdir()] == ["sample_atomic.json"]


def test_sample_manifest_store_skips_corrupt_canonical_manifest_files(tmp_path):
    store = SampleManifestStore(tmp_path)
    record = SampleRecord(
        sample_id="sample_good",
        task_type=TaskType.TEMPORAL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.SUCCEEDED,
        experiment=ExperimentRef(experiment_id="exp_clean"),
        sample_spec=SampleSpec(prompt="keep the clean record"),
    )

    store.put(record)

    corrupt_path = tmp_path / "samples" / "exp_corrupt" / "sample_corrupt.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not valid json")

    loaded = store.get("sample_corrupt")
    assert loaded is None

    records = store.list()
    assert len(records) == 1
    assert records[0].sample_id == "sample_good"


def test_sample_manifest_store_returns_none_for_corrupt_only_sample(tmp_path):
    store = SampleManifestStore(tmp_path)
    corrupt_path = tmp_path / "samples" / "sample_missing.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{broken")

    assert store.get("sample_missing") is None
    assert store.list() == []


# ---------------------------------------------------------------------------
# Temporal store
# ---------------------------------------------------------------------------


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
