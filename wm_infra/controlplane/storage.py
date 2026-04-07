"""Persistence helpers for control-plane sample manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

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

    def _bucket_dirs(self) -> list[Path]:
        return [path for path in sorted(self.samples_dir.iterdir()) if path.is_dir()]

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_record(self, path: Path) -> SampleRecord | None:
        try:
            return SampleRecord.model_validate_json(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, ValidationError):
            return None

    def put(self, record: SampleRecord) -> SampleRecord:
        path = self._record_path(record)
        self._atomic_write_text(path, json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
        return record

    def get(self, sample_id: str) -> SampleRecord | None:
        for bucket_dir in self._bucket_dirs():
            record = self._load_record(bucket_dir / f"{sample_id}.json")
            if record is not None:
                return record
        return None

    def list(self) -> list[SampleRecord]:
        records = []
        for bucket_dir in self._bucket_dirs():
            for path in sorted(bucket_dir.glob("*.json")):
                record = self._load_record(path)
                if record is not None:
                    records.append(record)
        return records
