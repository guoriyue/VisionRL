"""SD 3.5 Medium + OCR GRPO — thin entry-point.

Default config: ``configs/experiment/sd3_5_ocr_grpo.yaml``.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.sd3_5.train import train_sd3_5_grpo

    cfg = parse_and_load(
        default_config="experiment/sd3_5_ocr_grpo",
        description="SD 3.5 OCR GRPO Training",
    )
    asyncio.run(train_sd3_5_grpo(cfg))


if __name__ == "__main__":
    main()
