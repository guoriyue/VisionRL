"""Cosmos Predict2 2B Video2World GRPO — thin entry-point.

Default config: ``configs/experiment/cosmos_predict2_2b_grpo.yaml``.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.cosmos.train import train_cosmos_predict2_grpo

    cfg = parse_and_load(
        default_config="experiment/cosmos_predict2_2b_grpo",
        description="Cosmos Predict2 2B Video2World GRPO",
    )
    asyncio.run(train_cosmos_predict2_grpo(cfg))


if __name__ == "__main__":
    main()
