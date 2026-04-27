"""Wan 2.1 1.3B + single-reward GRPO — thin entry-point.

Default config: ``configs/experiment/wan_2_1_1_3b_grpo.yaml``.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.wan_2_1.train import train_wan_2_1_grpo

    cfg = parse_and_load(
        default_config="experiment/wan_2_1_1_3b_grpo",
        description="Wan 2.1 1.3B GRPO Training",
    )
    asyncio.run(train_wan_2_1_grpo(cfg))


if __name__ == "__main__":
    main()
