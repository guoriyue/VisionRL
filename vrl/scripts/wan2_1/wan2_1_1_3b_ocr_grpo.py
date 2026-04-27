"""Wan 2.1 1.3B + OCR GRPO — thin entry-point.

Loads ``configs/experiment/wan_2_1_1_3b_ocr_grpo.yaml`` by default.

Usage:
    python -m vrl.scripts.wan2_1.wan2_1_1_3b_ocr_grpo
    python -m vrl.scripts.wan2_1.wan2_1_1_3b_ocr_grpo trainer.seed=42
    python -m vrl.scripts.wan2_1.wan2_1_1_3b_ocr_grpo \\
        --config experiment/my_custom_wan_ocr
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.wan2_1.train import train_wan_grpo

    cfg = parse_and_load(
        default_config="experiment/wan_2_1_1_3b_ocr_grpo",
        description="Wan 2.1 1.3B OCR GRPO Training",
    )
    asyncio.run(train_wan_grpo(cfg))


if __name__ == "__main__":
    main()
