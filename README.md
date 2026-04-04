# wm-infra

Serving and control-plane infrastructure for video generation and world-model workloads.

`wm-infra` is **not** trying to be a general-purpose replacement for vLLM.
The target is narrower and more valuable:

> **Build the best serving stack for video-data production** — then grow it into a full data loop with lineage, evaluation, and training exports.

That means the repo is optimized for:
- long-running video generation jobs
- rollout- and diffusion-style workloads
- sample-level metadata and artifact tracking
- quality control and failure analysis
- experiment tracking and training export

## Product thesis

Most infra stops at "run the model".
That is not enough for video teams.

Real users need to:
- generate large volumes of video samples
- compare prompts / configs / backends
- filter bad outputs automatically
- review edge cases with humans
- trace every sample back to its origin
- export accepted data into training / eval pipelines

So the product is:

> **a serving-first Video DataOps platform**

The serving layer is the wedge.
The defensibility comes from:
- video-native scheduling
- sample lineage
- failure taxonomy
- evaluation loops
- training/export interfaces

## Strategic positioning

### What we are
- A serving runtime for world-model and video-generation workloads
- A control plane for producing, tracking, evaluating, and exporting samples
- A system for turning generation into reproducible data production

### What we are not
- Not a generic omni-model inference framework
- Not a 3DGS / NeRF reconstruction engine
- Not "another vLLM" competing on broad model coverage alone

## Where we can beat vLLM

Not by being more general.
By being more **video-native**.

### Better than generic inference infra at:
- long-job scheduling for video generation
- sample-level lineage and reproducibility
- artifact-aware execution
- generation → QC → review → export pipelines
- failure-mode classification and debugging
- training-data production workflows

## Architecture layers

### 1. Runtime layer
Responsible for executing model workloads.
Examples:
- latent rollout models
- diffusion / DiT video models
- image-to-video pipelines
- post-processing stages

Current code in this repo mostly lives here.

### 2. Control plane
Responsible for coordinating production workflows.
Needs to own:
- experiments
- jobs
- sample manifests
- artifact metadata
- backend selection
- cost / latency accounting

### 3. Evaluation layer
Responsible for deciding if outputs are usable.
Needs to own:
- automatic QC
- failure tags
- pairwise ranking
- human review queues
- acceptance policies

### 4. Export / training layer
Responsible for closing the loop.
Needs to own:
- training manifest export
- benchmark set export
- scorer / reranker datasets
- accepted-sample datasets

## Roadmap

### Phase 0 — Current repo
- World-model rollout engine
- scheduler / state cache concepts
- FastAPI server skeleton
- benchmark harness

### Phase 1 — Serving-first MVP
- unify job schema around `produce_sample`
- add backend abstraction for video pipelines
- persist sample manifests and artifacts
- expose experiment IDs and sample IDs
- basic QC hooks and failure tags

### Phase 2 — Data loop
- review queues
- pairwise comparison
- auto-scoring interfaces
- acceptance / rejection policies
- training export manifests

### Phase 3 — Model improvement loop
- scorer / reranker training
- routing / config optimization
- LoRA / adapter export pipelines
- active hard-case mining

## Repo layout

```text
wm_infra/
  api/            HTTP surface
  core/           runtime engine, scheduler, state handling
  controlplane/   sample schemas, manifests, experiment abstractions
  models/         model interfaces and registry
  tokenizer/      video tokenization
  ops/            backend ops / kernels
  kernels/        Triton kernels
  layers/         neural network building blocks
benchmarks/       microbenchmarks and runtime tests
docs/             product, architecture, and roadmap docs
tests/            unit tests
```

## Immediate priorities

1. Keep the runtime clean and measurable
2. Add a real control-plane schema instead of ad hoc request objects
3. Treat samples, artifacts, and lineage as first-class entities
4. Prepare the repo for evaluation and training export without bloating the core runtime

## Development

```bash
pip install -e .[dev]
pytest
wm-serve
```

## Startup framing

If this becomes a company, the pitch is not:

> we serve video models

It is:

> we turn video generation into a reproducible, scalable, quality-controlled data factory.

That distinction matters.
