"""Wan 2.1 14B GRPO — thin entry-point.

Default config: ``configs/experiment/wan_2_1_14b_grpo.yaml``.

NOTE: the 14B variant historically used the *official* WanT2V dual-expert
checkpoint (multi-GPU FSDP). The current YAML config drives the
``Wan_2_1Collector`` (diffusers single-GPU) path, matching the 1.3B setup —
suitable for prototyping. Multi-GPU FSDP migration is pending.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.wan_2_1.train import train_wan_2_1_grpo

    cfg = parse_and_load(
        default_config="experiment/wan_2_1_14b_grpo",
        description="Wan 2.1 14B GRPO Training",
    )
    asyncio.run(train_wan_2_1_grpo(cfg))


if __name__ == "__main__":
    main()
