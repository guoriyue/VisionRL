"""NextStep-1 + OCR GRPO -- thin entry-point.

Default config: ``configs/experiment/nextstep_1_ocr_grpo.yaml``.
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.nextstep_1.train import train_nextstep_1_ocr_grpo

    cfg = parse_and_load(
        default_config="experiment/nextstep_1_ocr_grpo",
        description="NextStep-1 OCR GRPO Training",
    )
    asyncio.run(train_nextstep_1_ocr_grpo(cfg))


if __name__ == "__main__":
    main()
