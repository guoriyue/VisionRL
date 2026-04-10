"""Gateway routes for sample production and discovery."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from wm_infra.controlplane import SampleManifestStore
from wm_infra.gateway.state import get_gateway_runtime


def register_sample_routes(app: FastAPI) -> None:
    """Register sample-production and serving routes."""
    router = APIRouter()

    @router.get("/v1/health")
    async def health():
        return {"status": "ready"}

    @router.get("/v1/models")
    async def list_models():
        from wm_infra.models.registry import list_models as _list_models

        return {"models": _list_models()}

    @router.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.get("/v1/samples")
    async def list_samples(
        request: Request,
        status: str | None = None,
        backend: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
    ):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        records = store.list()
        if status is not None:
            records = [record for record in records if record.status.value == status]
        if backend is not None:
            records = [record for record in records if record.backend == backend]
        if experiment_id is not None:
            records = [
                record
                for record in records
                if record.experiment is not None
                and record.experiment.experiment_id == experiment_id
            ]
        limit = max(1, min(limit, 200))
        records = records[:limit]
        return {
            "samples": [record.model_dump(mode="json") for record in records],
            "count": len(records),
        }

    @router.get("/v1/samples/{sample_id}")
    async def get_sample(sample_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return record.model_dump(mode="json")

    @router.get("/v1/samples/{sample_id}/artifacts")
    async def list_artifacts(sample_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return {
            "artifacts": [artifact.model_dump(mode="json") for artifact in record.artifacts],
            "count": len(record.artifacts),
        }

    @router.get("/v1/samples/{sample_id}/artifacts/{artifact_id}")
    async def get_artifact(sample_id: str, artifact_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        decoded_id = unquote(artifact_id)
        for artifact in record.artifacts:
            if artifact.artifact_id == decoded_id:
                return artifact.model_dump(mode="json")
        raise HTTPException(status_code=404, detail=f"Artifact not found: {decoded_id}")

    @router.get("/v1/samples/{sample_id}/artifacts/{artifact_id}/content")
    async def get_artifact_content(sample_id: str, artifact_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        decoded_id = unquote(artifact_id)
        artifact = next((a for a in record.artifacts if a.artifact_id == decoded_id), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {decoded_id}")
        if not artifact.uri.startswith("file://"):
            raise HTTPException(status_code=400, detail="Only file:// artifacts can be served")

        file_path = Path(artifact.uri[7:])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Artifact file not found on disk")
        return FileResponse(
            path=str(file_path),
            media_type=artifact.mime_type or "application/octet-stream",
            filename=file_path.name,
        )

    app.include_router(router)
