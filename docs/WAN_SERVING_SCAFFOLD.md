# Wan serving scaffold

`wm-infra` includes a concrete `wan-video` backend for temporal sample production.
This doc exists to describe the implemented Wan 2.2 path inside a repo that is otherwise broader temporal infra.

## What it does

- accepts `text_to_video`, `image_to_video`, and `video_to_video` sample requests
- uses first-class `wan_config` fields instead of hiding key execution knobs in metadata
- computes a scheduler-friendly resource estimate aligned with the memory-aware rollout work
- batches queue-compatible Wan 2.2 requests by shared diffusion shape and near-shape execution hints
- tracks a warmed execution-profile pool keyed by resolution, frame count, step count, and CFG buckets
- returns quality-cost fallback hints (`auto_step_reduction`, `resolution_fallback`, `progressive_preview`) when admission rejects an oversized request
- can run the official Wan2.2 Python modules in-process through an engine adapter and explicit `text_encode -> diffusion -> vae_decode -> safety -> postprocess -> persist` stage scheduler
- persists a durable sample record through the existing manifest store
- can run in:
  - **real in-process mode**: `WM_WAN_ENGINE_ADAPTER=official` loads the local Wan2.2 Python runtime directly and persists a real MP4 artifact
  - **real diffusers I2V mode**: `WM_WAN_ENGINE_ADAPTER=diffusers-i2v` loads the local Wan2.2 I2V diffusers snapshot directly and persists a real MP4 artifact
  - **real hybrid mode**: `WM_WAN_ENGINE_ADAPTER=hybrid` uses the official in-process adapter for T2V and the diffusers adapter for I2V
  - **test stub mode**: `WM_WAN_ENGINE_ADAPTER=stub` keeps the fast fake adapter used by unit tests
  - **shell mode**: `WM_WAN_SHELL_RUNNER` template is executed and its stdout/stderr is captured
  - **official external mode**: set `WM_WAN_ENGINE_ADAPTER=disabled` and `WM_WAN_REPO_DIR=...` to force the legacy `generate.py` subprocess path

## API example

```json
POST /v1/samples
{
  "task_type": "text_to_video",
  "backend": "wan-video",
  "model": "wan2.2-t2v-A14B",
  "sample_spec": {
    "prompt": "a corgi surfing through a data center"
  },
  "wan_config": {
    "num_steps": 4,
    "frame_count": 9,
    "width": 832,
    "height": 480,
    "guidance_scale": 4.0,
    "shift": 12.0,
    "memory_profile": "low_vram"
  }
}
```

## Environment variables

- `WM_WAN_OUTPUT_ROOT=/path/to/wan-output`
- `WM_WAN_SHELL_RUNNER='python run_wan.py --prompt {prompt} --size {width}x{height} --frames {frame_count} --steps {num_steps} --out {output_path}'`
- `WM_WAN_ENGINE_ADAPTER=official`
- `WM_WAN_REPO_DIR=/home/mingfeiguo/Desktop/Wan2.2`
- `WM_WAN_CKPT_DIR=/home/mingfeiguo/Desktop/Wan2.2/Wan2.2-T2V-A14B`
- `WM_WAN_I2V_DIFFUSERS_DIR=/home/mingfeiguo/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/<snapshot>`

## Notes

This is still an early serving path, not a claim of complete production maturity.
The queue batching and warm-profile logic still live above the model kernel layer, but the in-process path is now real on both verified lanes:
- T2V can run through the official Wan2.2 Python modules inside the service process.
- I2V can run through the local Wan2.2 diffusers snapshot inside the same stage-scheduler contract.
One practical detail matters: `sample_spec.references` is a URI-shaped field, so the backend normalizes `file://...` references into local paths before handing them to either the official runner or the in-process adapters.
The point is to keep Wan 2.2 sample production a first-class backend and API path while the runtime moves away from `generate.py` subprocess glue toward a true video-model execution plane.
