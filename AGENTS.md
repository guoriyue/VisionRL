# AGENTS.md

## Purpose

`wm-infra` is a temporal model serving and control-plane repository for video generation and world-model rollout production.

Keep the product framing concrete:
- primary backend paths are `wan-video` and `genie-rollout`
- the low-level `rollout-engine` still exists, but it is runtime substrate and bring-up infrastructure
- this is not a generic "serve any model" repo

When you update docs, APIs, or code comments, preserve that distinction.

## Working Style

- Make targeted changes that match the existing architecture instead of introducing a parallel abstraction.
- Prefer schema-first changes when behavior affects persisted samples, temporal entities, or backend request payloads.
- Keep backend-specific behavior explicit. Do not collapse Wan and Genie into a vague generic config surface unless the code already supports that cleanly.
- Avoid broad refactors unless they are required to complete the task.

## Repository Map

- `wm_infra/api/`: FastAPI server, HTTP protocol models, metrics, and app wiring
- `wm_infra/backends/`: concrete backend adapters, job queues, and backend registry
- `wm_infra/controlplane/`: sample schemas, manifests, temporal lineage, storage, and resource estimation
- `wm_infra/core/`: runtime engine, scheduler, and state handling for rollout execution
- `wm_infra/models/`: model interfaces and registry
- `wm_infra/tokenizer/`: video tokenizer code
- `wm_infra/ops/`, `wm_infra/kernels/`, `wm_infra/layers/`: lower-level model and kernel primitives
- `tests/`: unit and integration coverage for server, engine, control plane, benchmarking, and Genie behavior
- `docs/`: architecture, profiling, and strategy notes
- `benchmarks/`: benchmark harnesses and related utilities

## Main Entry Points

- App factory: `wm_infra.api.server:create_app`
- CLI entry point: `wm-serve`
- Package metadata and dependencies: `pyproject.toml`

Important HTTP surfaces:
- `POST /v1/samples` is the main product-facing sample-production API
- `GET /v1/samples`, `GET /v1/samples/{sample_id}`, and artifact endpoints are part of the persisted control-plane surface
- `POST /v1/rollout` and `GET /v1/rollout/{job_id}` are lower-level runtime endpoints

## Environment And Config

The codebase uses Python 3.10+ with editable install support.

Typical setup:

```bash
pip install -e .[dev]
pytest
wm-serve
```

Relevant environment variables include:
- `WM_MANIFEST_STORE_ROOT`
- `WM_WAN_OUTPUT_ROOT`
- `WM_WAN_SHELL_RUNNER`
- `WM_WAN_REPO_DIR`
- `WM_WAN_CONDA_ENV`
- `WM_WAN_MAX_QUEUE_SIZE`
- `WM_WAN_MAX_CONCURRENT_JOBS`
- `WM_GENIE_OUTPUT_ROOT`
- `WM_GENIE_MODEL_NAME`
- `WM_GENIE_DEVICE`
- `WM_GENIE_NUM_PROMPT_FRAMES`
- `WM_GENIE_MASKGIT_STEPS`
- `WM_GENIE_TEMPERATURE`
- `WM_GENIE_MAX_QUEUE_SIZE`
- `WM_GENIE_MAX_CONCURRENT_JOBS`

Defaults used by the README and app wiring:
- sample manifests under `${TMPDIR:-/tmp}/wm_infra`
- Wan outputs under `${TMPDIR:-/tmp}/wm_infra_wan`
- Genie outputs under `${TMPDIR:-/tmp}/wm_infra_genie`

## Change Guidelines

When changing APIs or schemas:
- update the relevant Pydantic models in `wm_infra/api/protocol.py` or `wm_infra/controlplane/`
- keep request fields first-class when they affect scheduling, runtime fit, lineage, or artifact semantics
- maintain compatibility where the server still backfills legacy metadata, unless the task explicitly removes it

When changing backend behavior:
- make backend-specific knobs live with the backend that owns them
- keep queueing and execution concerns separate from sample manifest persistence
- preserve async job flow for queue-backed backends

When changing docs:
- describe the repo as temporal sample-production infra
- avoid generic inference-platform language unless you are describing future work explicitly

## Validation

Prefer the smallest validation loop that proves the change:
- run focused tests for touched modules first, then broaden if needed
- use `pytest tests/test_server.py` for HTTP/server changes
- use `pytest tests/test_genie.py` for Genie backend changes
- use `pytest tests/test_controlplane.py` for manifest or schema changes
- use `pytest tests/test_engine.py` for runtime engine changes
- run full `pytest` after substantial cross-cutting edits

If you cannot run tests, state that clearly and explain why.

## Pitfalls To Avoid

- Do not reframe the repo as a generic LLM serving stack.
- Do not hide important Wan or Genie execution controls inside opaque metadata blobs.
- Do not treat the rollout engine as the sole product surface when touching top-level docs or API narratives.
- Do not mix unrelated cleanup into backend or schema changes without a clear payoff.

## Next-Step Plan

The likely next step for this repository is to grow from a temporal sample-production stack into stronger infrastructure for world models and video models.

The current ambition is not only to integrate existing engines.
The longer-term research direction may include directly implementing new runtime systems for world-model and video-model workloads, including original performance primitives that are meaningfully comparable in impact to ideas such as paged KV management or prefix/state reuse systems.

The intended direction is:
- keep `wm-infra` as the runtime + control plane + sample/lineage system
- study `vLLM`, `SGLang`, and similar systems carefully, but do not assume the repo must stop at integration
- identify workload-specific bottlenecks in temporal, video, and world-model serving, then design original systems that improve those bottlenecks
- keep explicit support for temporal workloads, long-running jobs, artifacts, state, and evaluation/export hooks

### Strategic Positioning

If the repo expands toward "best-in-class infra" for world models or video models, it should own:
- temporal request schemas and product-facing APIs
- job orchestration, queueing, admission control, and resource estimation
- sample manifests, artifacts, lineage, evaluation, and export
- stateful entities such as episodes, branches, rollouts, checkpoints, and handles
- observability, policy, safety, and multi-backend operations

It should usually not own:
- generic LLM token decoding engines already solved better by `vLLM`, `SGLang`, or `TensorRT-LLM`
- low-level optimizations whose main value is generic transformer serving rather than temporal product behavior

Important nuance:
- do not assume `wm-infra` should remain only a high-level control plane
- a valid target for this repo is to become a lower-level runtime substrate for learned temporal simulators built from video models and world models
- if the workload uses a video model or world model as the simulator itself, the repo may need to own execution primitives below the API/control-plane layer

For learned-simulator workloads, `wm-infra` may appropriately own:
- latent/state residency and memory management
- action-conditioned transition stepping and batching
- branch/fork/checkpoint primitives for temporal state
- disaggregated encode / transition / decode execution
- prompt/state reuse, cache locality, and state transfer reduction
- env setup and rollout execution when the "environment step" is implemented by a learned model rather than a traditional physics engine

For avoidance of doubt:
- the intended lower-level direction is not "build a generic game engine"
- the intended lower-level direction is "build runtime systems for learned temporal simulation"
- when comparing to `EnvPool` or `Madrona`, focus on the analogous systems problems for learned simulators: transition throughput, temporal state layout, cache reuse, branch efficiency, decode cost, and rollout continuity

### Recommended Architecture Direction

Use a layered architecture:
- data plane: video ingest, decoding, frame/audio extraction, preprocessing
- model-execution plane: Wan, Genie, rollout-engine, and future backend adapters
- inference-engine plane: external engines such as `vLLM` or `SGLang` when a backend depends on LLM/VLM decoding
- control plane: manifests, temporal lineage, state, storage, review, and export
- ops plane: metrics, tracing, policy, auth, quota, and deployment/runtime health

For future world-model work:
- treat state as a first-class object, not as request-local metadata
- keep rollout/session/episode identity explicit and durable
- separate runtime execution from persistent world-state bookkeeping
- treat learned transition execution as the core "env stepping" primitive when a world model is acting as the simulator

For learned-simulator work specifically:
- if the dominant bottleneck is inside transition execution, latent-state movement, or decode/encode staging, it is in scope for `wm-infra` to own that lower-level runtime work directly
- agents should not prematurely defer these problems to external serving engines or traditional simulator frameworks when the workload is fundamentally a learned temporal simulator
- the repo may grow execution abstractions that are closer to a temporal simulation engine than a request/response inference server, as long as they stay grounded in concrete `Wan` or `Genie` workload needs

For future video-model work:
- keep video-specific execution knobs first-class
- model the cost drivers explicitly: frames, resolution, steps, prompt/context size, offload mode, queue pressure
- treat video decode, preprocessing, and encoder stages as distinct infrastructure concerns from text decoding

### vLLM / SGLang Adoption Guidance

Use `vLLM` when:
- the main need is stable, high-throughput OpenAI-compatible serving
- the bottleneck is mostly LLM/VLM decode throughput and KV-cache efficiency
- the team wants lower integration overhead and mature metrics

Use `SGLang` when:
- the workload has repeated prefixes, multi-call programs, structured output, agent loops, or cache-aware routing needs
- prefill/decode disaggregation and gateway-style routing materially help the workload
- the system behaves more like a structured runtime than a single-shot generation API

Prefer a hybrid path when useful:
- `wm-infra` remains the product and control plane
- `vLLM` can serve as a general decoding backend
- `SGLang` can be introduced as a runtime or gateway layer for structured flows, routing, and prefix-heavy workloads

Important clarification:
- learning from `vLLM` and `SGLang` is encouraged
- reusing them as baselines or temporary backends is acceptable
- but the repo may also pursue direct implementation of new systems if they are specifically motivated by world-model or video-model workloads and are not just generic re-creations of existing LLM serving engines

### Original Systems Research Direction

If this repo pursues original systems work, the standard should be high:
- do not invent new mechanisms without showing they target a dominant workload bottleneck
- do not optimize for synthetic microbenchmarks only; tie the design to real temporal, video, or world-model workloads
- aim for mechanisms with clear system value, such as better memory efficiency, reduced state transfer, higher cache hit rates, lower queueing delay, or more stable tail latency

Possible directions include:
- temporal-aware memory management for KV, encoder outputs, latent states, or rollout states
- state-aware scheduling that keeps episode or rollout execution near the relevant cached state
- hierarchical caching across text prefixes, visual embeddings, latent trajectories, and control-plane artifacts
- disaggregated execution across decode, prefill, encoder, and state-transition stages
- new admission-control or routing systems that optimize for temporal workloads rather than generic text-only serving
- learned-environment execution runtimes where env setup, step, branch, and checkpoint are implemented over model state rather than explicit physics state
- world-model/video-model serving engines that are optimized for simulator-style rollout throughput instead of single-shot generation APIs

The right goal is not "build another vLLM".
The right goal is to discover and implement performance primitives that matter specifically for world-model and video-model infrastructure.

Profile first.
Before committing to a new systems primitive, agents should assume the bottleneck is not yet proven.
The expected workflow is:
- define the workload clearly
- measure the workload on the current stack
- identify the dominant cost centers
- only then propose or implement a new systems mechanism

For this repository, that means original systems work should usually follow profiling of the current `Wan` and `Genie` paths before broad architectural claims are made.

### Current Research Focus

The primary concrete workload paths in this repo are:
- `wan-video` as the current video-model path
- `genie-rollout` as the current world-model / temporal-rollout path

Agents should treat these two paths as the highest-priority sources of truth for systems research in this repository.

Practical implication:
- if a performance idea is proposed for video workloads, validate it first against the `Wan` path
- if a performance idea is proposed for world-model workloads, validate it first against the `Genie` path
- avoid drifting into abstract "general multimodal infra" design without first checking whether the issue is visible in `Wan` or `Genie`

### Performance Measurement Requirements

Any major systems claim in this repo should be backed by measurement.
At minimum, agents should try to instrument, preserve, or improve visibility into:
- `TTFT`
- `TPOT`
- end-to-end latency
- GPU utilization
- KV/cache usage
- encoder latency
- decode latency
- queue wait
- state transfer latency
- artifact IO latency

When proposing a new optimization, clarify which of the above metrics it is expected to improve and which tradeoffs it may worsen.

For video-model workloads, assume the first bottleneck may be visual input expansion, encoder cost, or prefill cost rather than decode alone.
For world-model workloads, assume the first bottleneck may involve state management, rollout continuity, queueing, or state transfer rather than generic text-only serving costs.

### Continuous Optimization Loop

Agents working on performance should follow an explicit iteration loop rather than making broad speculative changes.

Default loop:
1. choose the concrete workload path first: `wan-video` or `genie-rollout`
2. define the target behavior and the metric or metrics that matter
3. measure the current baseline before changing code
4. identify the dominant bottleneck rather than guessing
5. make one focused change aimed at that bottleneck
6. rerun the relevant benchmark, test, or profiling pass
7. compare the result against the baseline
8. record what improved, what regressed, and what remains unclear

Iteration rules:
- prefer one bottleneck per iteration
- prefer one clearly attributable systems change per iteration
- do not stack many speculative optimizations into one patch if attribution will become unclear
- if a change does not improve the target metric meaningfully, do not justify it with vague architectural arguments
- if profiling shows a different bottleneck than expected, update direction quickly

Expected optimization mindset:
- benchmark first, then optimize
- optimize the dominant cost center, not the most fashionable subsystem
- preserve correctness, reproducibility, and observability while optimizing
- turn successful optimizations into reusable mechanisms, not one-off hacks

When practical, preserve or add artifacts that help future iterations:
- reproducible benchmark inputs
- profiling notes
- before/after metric snapshots
- explicit assumptions about workload shape

The goal is to make repeated agent work cumulative.
Future agents should be able to continue from measured evidence rather than restarting from intuition.

### Execution Priorities

Short term:
- keep the northbound API centered on `POST /v1/samples`
- make internal/runtime/control-plane APIs clearly distinguished from product-facing APIs
- improve observability for queue depth, job latency, backend latency, artifact persistence, admission behavior, and the performance metrics listed above
- make backend adapters cleaner so external inference engines can be integrated without distorting the top-level API
- profile `wan-video` and `genie-rollout` end to end before committing to major new systems primitives

Medium term:
- add explicit engine adapter boundaries for future `vLLM`/`SGLang`-backed backends
- strengthen stateful temporal entities for world-model workloads
- formalize evaluation, review, and export interfaces
- add policy around resource budgeting, caching assumptions, and multi-tenant isolation

Long term:
- evolve into a temporal infrastructure layer that can support world-model serving, video generation, rollout systems, and future evaluation/export loops
- remain honest about scope: original systems work is welcome, but it should be driven by real workload bottlenecks and defended with rigorous benchmarks

### Non-Goals For Now

- Do not turn the repo into a generic chatbot-serving platform.
- Do not replace specialized inference engines unless there is a clear temporal-product reason.
- Do not let backend integration erase the repo's control-plane identity.
