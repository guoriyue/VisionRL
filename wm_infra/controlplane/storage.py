"""Persistence helpers for control-plane sample manifests."""

from __future__ import annotations

import json
from pathlib import Path

from wm_infra.controlplane.schemas import SampleRecord


class SampleManifestStore:
    """Simple file-backed sample manifest store.

    Persists each sample record as one JSON document under ``root/samples``.
    When an experiment id is present, records are organized as
    ``root/samples/<experiment_id>/<sample_id>.json``.
    Unscoped samples are stored under ``root/samples/_default/<sample_id>.json``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def _experiment_bucket(self, record: SampleRecord) -> str:
        if record.experiment and record.experiment.experiment_id:
            return record.experiment.experiment_id
        return "_default"

    def _record_path(self, record: SampleRecord) -> Path:
        return self.samples_dir / self._experiment_bucket(record) / f"{record.sample_id}.json"

    def _find_path(self, sample_id: str) -> Path | None:
        legacy_path = self.samples_dir / f"{sample_id}.json"
        if legacy_path.exists():
            return legacy_path

        for path in sorted(self.samples_dir.glob(f"**/{sample_id}.json")):
            return path
        return None

    def put(self, record: SampleRecord) -> SampleRecord:
        path = self._record_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
        return record

    def get(self, sample_id: str) -> SampleRecord | None:
        path = self._find_path(sample_id)
        if path is None:
            return None
        return SampleRecord.model_validate_json(path.read_text())

    def list(self) -> list[SampleRecord]:
        records = []
        for path in sorted(self.samples_dir.glob("**/*.json")):
            records.append(SampleRecord.model_validate_json(path.read_text()))
        return records
