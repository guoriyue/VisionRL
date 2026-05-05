# visual-rl

RL-style post-training infrastructure for visual generative models.

This matrix tracks repository integration status, not model quality or benchmark
claims.

## Model / Algorithm Matrix

Legend:

- `[x]` active: experiment YAML, training entrypoint, rollout adapter, and
  structural tests exist.
- `[~]` wired: code path exists, but the model binding still needs real
  checkpoint smoke validation.
- `[ ]` not wired.
- `-` not a target pairing for the current model family.

| Model | Modality | GRPO | TokenGRPO | Diffusion DPO | Current progress |
| --- | --- | --- | --- | --- | --- |
| SD3.5 | text-to-image diffusion | `[x]` `sd3_5_ocr_grpo` | - | `[ ]` | Active GRPO path. |
| Wan 2.1 1.3B | text-to-video diffusion | `[x]` `wan_2_1_1_3b_grpo`, `wan_2_1_1_3b_ocr_grpo`, `wan_2_1_1_3b_multi_reward_grpo` | - | `[x]` `wan_2_1_1_3b_dpo` | Active GRPO and offline DPO paths. |
| Wan 2.1 14B | text-to-video diffusion | `[x]` `wan_2_1_14b_grpo` | - | `[ ]` | Shares the Wan GRPO path; 14B scale still needs real-run validation. |
| Cosmos Predict2 2B | video-to-world diffusion | `[x]` `cosmos_predict2_2b_grpo` | - | `[ ]` | Active Predict2 GRPO wiring. |
| Janus-Pro 1B | autoregressive image | - | `[x]` `janus_pro_1b_grpo`, `janus_pro_1b_ocr_grpo` | - | Active TokenGRPO path. |
| NextStep-1 1.1 | continuous-token autoregressive image | - | `[~]` `nextstep_1_ocr_grpo` | - | Wired, but still marked pre-smoke for real checkpoint binding. |

## Algorithm Kinds

| Algorithm kind | Used by | Config base |
| --- | --- | --- |
| `grpo` | SD3.5, Wan 2.1, Cosmos Predict2 | `configs/base/algorithm/grpo.yaml` |
| `token_grpo` | Janus-Pro, NextStep-1 | `configs/base/algorithm/token_grpo.yaml` |
| `diffusion_dpo` | Wan 2.1 offline DPO | `configs/base/algorithm/dpo.yaml` |

Run any active experiment with:

```bash
python -m vrl.scripts.train --config experiment/<config_name>
```

Example:

```bash
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_grpo
```
