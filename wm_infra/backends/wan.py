"""Wan-family video backend with async job queue and official runner support.

This backend supports three execution modes:
- **stub**: No runner configured — materializes request/log paths only.
- **shell**: A shell command template is executed (WM_WAN_SHELL_RUNNER).
- **official**: Builds and executes the real Wan2.2 generate.py command using
  the local repo path and conda env from WAN22_BASELINE.md.

All modes support both synchronous execution (via produce_sample) and
asynchronous submission (via submit_async → poll status).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.controlplane.resource_estimator import estimate_wan_request
from wm_infra.controlplane.schemas import (
    ArtifactKind,
    ArtifactRecord,
    ProduceSampleRequest,
    SampleRecord,
    SampleStatus,
    TaskType,
    WanTaskConfig,
)


class WanVideoBackend(ProduceSampleBackend):
    """Wan video generation backend with stub, shell, and official runner modes."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        backend_name: str = "wan-video",
        shell_runner: str | None = None,
        shell_runner_timeout_s: int | None = None,
        wan_admission_max_units: float | None = None,
        wan_admission_max_vram_gb: float | None = 32.0,
        # Official runner config
        wan_repo_dir: str | None = None,
        wan_conda_env: str | None = None,
        conda_sh_path: str | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.shell_runner = shell_runner
        self.shell_runner_timeout_s = shell_runner_timeout_s
        self.wan_admission_max_units = wan_admission_max_units
        self.wan_admission_max_vram_gb = wan_admission_max_vram_gb

        # Official runner: local Wan2.2 repo invocation
        self.wan_repo_dir = wan_repo_dir or os.environ.get("WM_WAN_REPO_DIR")
        self.wan_conda_env = wan_conda_env or os.environ.get("WM_WAN_CONDA_ENV", "kosen")
        self.conda_sh_path = conda_sh_path or os.environ.get(
            "WM_CONDA_SH_PATH", os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
        )

        # Job queue is attached externally by the server lifespan
        self._job_queue = None

    @property
    def runner_mode(self) -> str:
        if self.wan_repo_dir:
            return "official"
        if self.shell_runner:
            return "shell"
        return "stub"

    def _sample_dir(self, sample_id: str) -> Path:
        path = self.output_root / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_wan_config(self, request: ProduceSampleRequest) -> WanTaskConfig:
        return request.wan_config or WanTaskConfig(
            width=request.sample_spec.width or 832,
            height=request.sample_spec.height or 480,
        )

    def _validate_request(self, request: ProduceSampleRequest) -> None:
        if request.task_type not in {TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO}:
            raise ValueError(f"Backend {self.backend_name} only supports Wan-style video tasks")

        references = request.sample_spec.references
        if request.task_type == TaskType.TEXT_TO_VIDEO and not (request.sample_spec.prompt or "").strip():
            raise ValueError("Wan text_to_video requests require a non-empty sample_spec.prompt")
        if request.task_type in {TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO} and not references:
            raise ValueError(f"Wan {request.task_type.value} requests require at least one sample_spec.references item")
        if self.runner_mode == "official" and request.task_type == TaskType.VIDEO_TO_VIDEO:
            raise ValueError("Official Wan runner wiring currently supports text_to_video and image_to_video only")

    def _admission_result(self, request: ProduceSampleRequest, wan_config: WanTaskConfig) -> tuple[bool, dict[str, Any], Any]:
        estimate = estimate_wan_request(wan_config)
        reasons: list[str] = []
        if self.wan_admission_max_units is not None and estimate.estimated_units > self.wan_admission_max_units:
            reasons.append(
                f"estimated_units {estimate.estimated_units:.2f} exceeds limit {self.wan_admission_max_units:.2f}"
            )
        if (
            self.wan_admission_max_vram_gb is not None
            and estimate.estimated_vram_gb is not None
            and estimate.estimated_vram_gb > self.wan_admission_max_vram_gb
        ):
            reasons.append(
                f"estimated_vram_gb {estimate.estimated_vram_gb:.2f} exceeds limit {self.wan_admission_max_vram_gb:.2f}"
            )
        admitted = not reasons
        return admitted, {
            "admitted": admitted,
            "reasons": reasons,
            "max_units": self.wan_admission_max_units,
            "max_vram_gb": self.wan_admission_max_vram_gb,
        }, estimate

    def _build_request_payload(
        self,
        sample_id: str,
        request: ProduceSampleRequest,
        wan_config: WanTaskConfig,
        estimate: Any,
        plan_path: Path,
        log_path: Path,
        video_path: Path,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "task_type": request.task_type.value,
            "backend": request.backend,
            "model": request.model,
            "model_revision": request.model_revision,
            "sample_spec": request.sample_spec.model_dump(mode="json"),
            "wan_config": wan_config.model_dump(mode="json"),
            "resource_estimate": estimate.model_dump(mode="json"),
            "artifacts": {
                "request_path": str(plan_path),
                "log_path": str(log_path),
                "output_path": str(video_path),
            },
            "request_context": {
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "experiment": request.experiment.model_dump(mode="json") if request.experiment else None,
            },
        }

    def _format_shell_command(
        self,
        request: ProduceSampleRequest,
        sample_id: str,
        wan_config: WanTaskConfig,
        plan_path: Path,
        log_path: Path,
        video_path: Path,
    ) -> str:
        assert self.shell_runner is not None
        return self.shell_runner.format(
            sample_id=sample_id,
            task_type=request.task_type.value,
            model=request.model,
            model_revision=request.model_revision or "",
            prompt=shlex.quote(request.sample_spec.prompt or ""),
            negative_prompt=shlex.quote(request.sample_spec.negative_prompt or ""),
            width=wan_config.width,
            height=wan_config.height,
            frame_count=wan_config.frame_count,
            num_steps=wan_config.num_steps,
            guidance_scale=wan_config.guidance_scale,
            shift=wan_config.shift,
            memory_profile=wan_config.memory_profile.value,
            model_size=wan_config.model_size,
            ckpt_dir=shlex.quote(wan_config.ckpt_dir or ""),
            seed="" if request.sample_spec.seed is None else request.sample_spec.seed,
            fps="" if request.sample_spec.fps is None else request.sample_spec.fps,
            duration_seconds="" if request.sample_spec.duration_seconds is None else request.sample_spec.duration_seconds,
            reference_path=shlex.quote(request.sample_spec.references[0]) if request.sample_spec.references else "",
            references_json=shlex.quote(json.dumps(request.sample_spec.references)),
            controls_json=shlex.quote(json.dumps(request.sample_spec.controls, sort_keys=True)),
            metadata_json=shlex.quote(json.dumps(request.sample_spec.metadata, sort_keys=True)),
            labels_json=shlex.quote(json.dumps(request.labels, sort_keys=True)),
            output_path=shlex.quote(str(video_path)),
            request_path=shlex.quote(str(plan_path)),
            log_path=shlex.quote(str(log_path)),
        )

    def _build_official_command(
        self,
        request: ProduceSampleRequest,
        sample_id: str,
        wan_config: WanTaskConfig,
        video_path: Path,
    ) -> str:
        """Build the official Wan2.2 generate.py command per WAN22_BASELINE.md."""
        assert self.wan_repo_dir is not None

        task_flag_map = {
            TaskType.TEXT_TO_VIDEO: f"t2v-{wan_config.model_size}",
            TaskType.IMAGE_TO_VIDEO: f"i2v-{wan_config.model_size}",
        }
        task_flag = task_flag_map.get(request.task_type, f"t2v-{wan_config.model_size}")

        parts = [
            f"source {shlex.quote(self.conda_sh_path)}",
            f"conda activate {shlex.quote(self.wan_conda_env)}",
            f"cd {shlex.quote(self.wan_repo_dir)}",
            "python generate.py",
            f"  --task {task_flag}",
            f"  --size {wan_config.width}*{wan_config.height}",
            f"  --frame_num {wan_config.frame_count}",
        ]
        if wan_config.ckpt_dir:
            parts.append(f"  --ckpt_dir {shlex.quote(wan_config.ckpt_dir)}")
        if wan_config.offload_model:
            parts.append("  --offload_model True")
        if wan_config.convert_model_dtype:
            parts.append("  --convert_model_dtype")
        if wan_config.t5_cpu:
            parts.append("  --t5_cpu")
        if request.task_type == TaskType.IMAGE_TO_VIDEO and request.sample_spec.references:
            parts.append(f"  --image {shlex.quote(request.sample_spec.references[0])}")
        parts.extend([
            f"  --sample_steps {wan_config.num_steps}",
            f"  --sample_shift {wan_config.shift}",
            f"  --sample_guide_scale {wan_config.guidance_scale}",
            f"  --prompt {shlex.quote(request.sample_spec.prompt or '')}",
            f"  --save_file {shlex.quote(str(video_path))}",
        ])
        if request.sample_spec.seed is not None:
            parts.append(f"  --seed {request.sample_spec.seed}")

        return " && ".join(parts[:3]) + " && " + " \\\n".join(parts[3:])

    def _artifact_details(self, path: Path) -> tuple[int | None, str | None]:
        if not path.exists() or not path.is_file():
            return None, None
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()

    def _artifact_record(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        path: Path,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        bytes_size, sha256 = self._artifact_details(path)
        payload = {"exists": path.exists()}
        if metadata:
            payload.update(metadata)
        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            uri=f"file://{path}",
            mime_type=mime_type,
            bytes=bytes_size,
            sha256=sha256,
            metadata=payload,
        )

    def _failure_payload(
        self,
        *,
        sample_id: str,
        status: SampleStatus,
        command: str | None,
        log_path: Path,
        video_path: Path,
        stdout: bytes | None,
        returncode: int | None,
        timed_out: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "status": status.value,
            "returncode": returncode,
            "timed_out": timed_out,
            "command": command,
            "log_path": str(log_path),
            "output_path": str(video_path),
            "output_exists": video_path.exists(),
            "tail": (stdout or b"")[-4000:].decode("utf-8", errors="replace"),
            "error": error,
        }

    async def execute_job(self, request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        """Execute a Wan job synchronously (called by queue worker or produce_sample)."""
        self._validate_request(request)

        wan_config = self._resolve_wan_config(request)
        sample_dir = self._sample_dir(sample_id)
        plan_path = sample_dir / "request.json"
        log_path = sample_dir / "runner.log"
        video_path = sample_dir / "sample.mp4"
        runtime_path = sample_dir / "runtime.json"
        failure_path = sample_dir / "failure.json"

        estimate = estimate_wan_request(wan_config)
        request_payload = self._build_request_payload(sample_id, request, wan_config, estimate, plan_path, log_path, video_path)
        plan_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True))

        started_at = time.time()
        mode = self.runner_mode
        runtime: dict[str, Any] = {
            "runner": mode,
            "request_path": str(plan_path),
            "output_path": str(video_path),
            "log_path": str(log_path),
            "runtime_path": str(runtime_path),
            "failure_path": str(failure_path),
            "status_history": [
                {"status": SampleStatus.QUEUED.value, "timestamp": started_at},
                {"status": SampleStatus.RUNNING.value, "timestamp": started_at},
            ],
            "started_at": started_at,
        }
        status = SampleStatus.ACCEPTED if mode == "stub" else SampleStatus.SUCCEEDED
        metadata: dict[str, Any] = {
            "evaluation_policy": request.evaluation_policy,
            "priority": request.priority,
            "labels": request.labels,
            "stubbed": mode == "stub",
            "runner_mode": mode,
        }

        stdout: bytes | None = None
        returncode: int | None = None
        timed_out = False
        command: str | None = None
        if mode in ("shell", "official"):
            command = (
                self._build_official_command(request, sample_id, wan_config, video_path)
                if mode == "official"
                else self._format_shell_command(request, sample_id, wan_config, plan_path, log_path, video_path)
            )
            runtime["command"] = command
            runtime["timeout_s"] = self.shell_runner_timeout_s
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env={
                        **os.environ,
                        "WM_SAMPLE_ID": sample_id,
                        "WM_REQUEST_PATH": str(plan_path),
                        "WM_OUTPUT_PATH": str(video_path),
                        "WM_LOG_PATH": str(log_path),
                    },
                )
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.shell_runner_timeout_s)
                except asyncio.TimeoutError:
                    timed_out = True
                    proc.kill()
                    stdout, _ = await proc.communicate()
                returncode = proc.returncode
                log_path.write_bytes(stdout or b"")
                runtime.update({"returncode": returncode, "timed_out": timed_out})
                if timed_out or returncode != 0:
                    status = SampleStatus.FAILED
                    metadata["runner_error"] = (
                        f"runner timed out after {self.shell_runner_timeout_s}s"
                        if timed_out
                        else f"runner exited with code {returncode}"
                    )
                    failure_path.write_text(
                        json.dumps(
                            self._failure_payload(
                                sample_id=sample_id,
                                status=status,
                                command=command,
                                log_path=log_path,
                                video_path=video_path,
                                stdout=stdout,
                                returncode=returncode,
                                timed_out=timed_out,
                                error=metadata["runner_error"],
                            ),
                            indent=2,
                            sort_keys=True,
                        )
                    )
                elif not video_path.exists():
                    status = SampleStatus.FAILED
                    metadata["runner_error"] = "runner completed without producing output video"
                    failure_path.write_text(
                        json.dumps(
                            self._failure_payload(
                                sample_id=sample_id,
                                status=status,
                                command=command,
                                log_path=log_path,
                                video_path=video_path,
                                stdout=stdout,
                                returncode=returncode,
                                timed_out=False,
                                error=metadata["runner_error"],
                            ),
                            indent=2,
                            sort_keys=True,
                        )
                    )
            except Exception as exc:
                status = SampleStatus.FAILED
                metadata["runner_error"] = str(exc)
                runtime["spawn_error"] = str(exc)
                log_path.write_text(f"Failed to launch Wan runner: {exc}\n")
                failure_path.write_text(
                    json.dumps(
                        self._failure_payload(
                            sample_id=sample_id,
                            status=status,
                            command=command,
                            log_path=log_path,
                            video_path=video_path,
                            stdout=None,
                            returncode=returncode,
                            timed_out=False,
                            error=str(exc),
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
        else:
            log_path.write_text("Wan backend scaffold executed in stub mode. No runner configured.\n")

        completed_at = time.time()
        runtime["completed_at"] = completed_at
        runtime["elapsed_ms"] = round((completed_at - started_at) * 1000, 2)
        runtime["status_history"].append({"status": status.value, "timestamp": completed_at})
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True))

        artifacts = [
            self._artifact_record(
                artifact_id=f"{sample_id}:log",
                kind=ArtifactKind.LOG,
                path=log_path,
                mime_type="text/plain",
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:metadata",
                kind=ArtifactKind.METADATA,
                path=plan_path,
                mime_type="application/json",
                metadata={"role": "request"},
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:runtime",
                kind=ArtifactKind.METADATA,
                path=runtime_path,
                mime_type="application/json",
                metadata={"role": "runtime"},
            ),
        ]
        if video_path.exists():
            artifacts.insert(
                0,
                self._artifact_record(
                    artifact_id=f"{sample_id}:video",
                    kind=ArtifactKind.VIDEO,
                    path=video_path,
                    mime_type="video/mp4",
                    metadata={"stubbed": mode == "stub"},
                ),
            )
        if failure_path.exists():
            artifacts.append(
                self._artifact_record(
                    artifact_id=f"{sample_id}:failure",
                    kind=ArtifactKind.METADATA,
                    path=failure_path,
                    mime_type="application/json",
                    metadata={"role": "failure"},
                )
            )

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            model_revision=request.model_revision,
            status=status,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            wan_config=wan_config,
            resource_estimate=estimate,
            artifacts=artifacts,
            runtime=runtime,
            metadata=metadata,
        )

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        self._validate_request(request)
        sample_id = str(uuid.uuid4())
        return await self.execute_job(request, sample_id)

    def submit_async(self, request: ProduceSampleRequest) -> SampleRecord:
        self._validate_request(request)
        if self._job_queue is None:
            raise RuntimeError("No job queue attached — cannot submit async")

        sample_id = str(uuid.uuid4())
        wan_config = self._resolve_wan_config(request)
        admitted, admission, estimate = self._admission_result(request, wan_config)
        queued_at = time.time()
        queue_position = None
        if admitted:
            self._job_queue.submit(sample_id, request)
            queue_position = self._job_queue.position(sample_id)

        runtime = {
            "runner": self.runner_mode,
            "async": True,
            "queued_at": queued_at,
            "admission": admission,
            "status_history": [
                {"status": (SampleStatus.QUEUED if admitted else SampleStatus.REJECTED).value, "timestamp": queued_at},
            ],
        }
        if queue_position is not None:
            runtime["queue_position"] = queue_position
            runtime["queue_snapshot"] = self._job_queue.snapshot()

        metadata = {
            "evaluation_policy": request.evaluation_policy,
            "priority": request.priority,
            "labels": request.labels,
            "stubbed": self.runner_mode == "stub",
            "async": True,
            "runner_mode": self.runner_mode,
        }
        if not admitted:
            metadata["runner_error"] = "; ".join(admission["reasons"])

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            model_revision=request.model_revision,
            status=SampleStatus.QUEUED if admitted else SampleStatus.REJECTED,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            wan_config=wan_config,
            resource_estimate=estimate,
            runtime=runtime,
            metadata=metadata,
        )
