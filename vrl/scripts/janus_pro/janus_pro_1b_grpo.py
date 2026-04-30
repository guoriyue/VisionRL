"""Janus-Pro-1B GRPO — thin entry-point.

Default config: ``configs/experiment/janus_pro_1b_grpo.yaml``.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.janus_pro.train import train_janus_pro_grpo

    cfg = parse_and_load(
        default_config="experiment/janus_pro_1b_grpo",
        description="Janus-Pro-1B GRPO Training",
    )
    asyncio.run(train_janus_pro_grpo(cfg))


if __name__ == "__main__":
    main()
