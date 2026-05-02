"""Generation worker that executes family pipeline executors."""

from __future__ import annotations

import logging
from typing import Any

from vrl.engine.generation.registry import FamilyPipelineRegistry
from vrl.engine.generation.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)

logger = logging.getLogger(__name__)


class GenerationIdFactory:
    """Build deterministic sample specs from a generation request."""

    def build_sample_specs(
        self,
        request: GenerationRequest,
    ) -> list[GenerationSampleSpec]:
        base_seed = request.sampling.get("seed")
        seed_int = int(base_seed) if base_seed is not None else None
        specs: list[GenerationSampleSpec] = []
        for prompt_index, prompt in enumerate(request.prompts):
            prompt_id = f"{request.request_id}:prompt:{prompt_index}"
            group_id = prompt_id
            for sample_index in range(request.samples_per_prompt):
                flat_index = len(specs)
                sample_id = f"{prompt_id}:sample:{sample_index}"
                metadata = dict(request.metadata)
                metadata.update(
                    {
                        "request_id": request.request_id,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "flat_sample_index": flat_index,
                        "policy_version": request.policy_version,
                    }
                )
                specs.append(
                    GenerationSampleSpec(
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        prompt=prompt,
                        prompt_id=prompt_id,
                        group_id=group_id,
                        sample_id=sample_id,
                        trajectory_id=sample_id,
                        seed=None if seed_int is None else seed_int + flat_index,
                        metadata=metadata,
                    )
                )
        return specs


class GenerationWorker:
    """Execute generation requests on one local worker."""

    def __init__(
        self,
        registry: FamilyPipelineRegistry,
        *,
        id_factory: GenerationIdFactory | None = None,
        device: Any | None = None,
    ) -> None:
        self.registry = registry
        self.id_factory = id_factory or GenerationIdFactory()
        self.device = device

    def execute(
        self,
        requests: list[GenerationRequest],
    ) -> dict[str, OutputBatch]:
        outputs: dict[str, OutputBatch] = {}
        for grouped_requests in self._group_batchable_requests(requests):
            if len(grouped_requests) == 1:
                request = grouped_requests[0]
                outputs[request.request_id] = self._execute_one(request)
            else:
                outputs.update(self._execute_group(grouped_requests))
        return outputs

    def _execute_one(self, request: GenerationRequest) -> OutputBatch:
        sample_specs = self.id_factory.build_sample_specs(request)
        try:
            executor = self.registry.resolve(request.family, request.task)
            output = executor.forward(request, sample_specs)
            if output.request_id != request.request_id:
                raise ValueError(
                    f"Executor returned request_id={output.request_id!r} for "
                    f"request_id={request.request_id!r}"
                )
            return output
        except Exception as exc:
            logger.exception("Generation request %s failed", request.request_id)
            return OutputBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=list(request.prompts),
                sample_specs=sample_specs,
                output=None,
                metrics=GenerationMetrics(
                    num_prompts=len(request.prompts),
                    num_samples=len(sample_specs),
                ),
                error=str(exc),
            )

    def _execute_group(
        self,
        requests: list[GenerationRequest],
    ) -> dict[str, OutputBatch]:
        sample_specs_by_request = {
            request.request_id: self.id_factory.build_sample_specs(request)
            for request in requests
        }
        try:
            executor = self.registry.resolve(requests[0].family, requests[0].task)
            forward_batch = getattr(executor, "forward_batch", None)
            if forward_batch is None:
                return {
                    request.request_id: self._execute_one(request)
                    for request in requests
                }
            outputs = forward_batch(requests, sample_specs_by_request)
            return {
                request.request_id: outputs.get(request.request_id)
                or _error_output(
                    request,
                    sample_specs_by_request[request.request_id],
                    "Batched executor did not return an output for this request",
                )
                for request in requests
            }
        except Exception as exc:
            logger.exception(
                "Generation request batch failed: %s",
                [request.request_id for request in requests],
            )
            return {
                request.request_id: _error_output(
                    request,
                    sample_specs_by_request[request.request_id],
                    str(exc),
                )
                for request in requests
            }

    def _group_batchable_requests(
        self,
        requests: list[GenerationRequest],
    ) -> list[list[GenerationRequest]]:
        groups: dict[Any, list[GenerationRequest]] = {}
        ordered_keys: list[Any] = []
        for request in requests:
            key = _strict_batch_key(request)
            if key not in groups:
                groups[key] = []
                ordered_keys.append(key)
            groups[key].append(request)
        return [groups[key] for key in ordered_keys]


def _strict_batch_key(request: GenerationRequest) -> tuple[Any, ...]:
    if not _safe_to_batch(request):
        return (request.request_id,)
    return (
        request.family,
        request.task,
        request.samples_per_prompt,
        request.policy_version,
        tuple(sorted(request.return_artifacts)),
        _freeze(request.sampling),
    )


def _safe_to_batch(request: GenerationRequest) -> bool:
    sampling = request.sampling
    if "seed" in sampling:
        return False
    return int(sampling.get("sde_window_size", 0)) <= 0


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(inner)) for key, inner in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(inner) for inner in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(inner) for inner in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _error_output(
    request: GenerationRequest,
    sample_specs: list[Any],
    error: str,
) -> OutputBatch:
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=None,
        metrics=GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
        ),
        error=error,
    )
