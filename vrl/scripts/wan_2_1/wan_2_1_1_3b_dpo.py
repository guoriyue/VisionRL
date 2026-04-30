"""Wan 2.1 1.3B + Diffusion-DPO — thin entry-point.

Default config: ``configs/experiment/wan_2_1_1_3b_dpo.yaml``.
"""

from __future__ import annotations


def main() -> None:
    from vrl.config.cli import parse_and_load
    from vrl.scripts.wan_2_1.train_dpo import train_wan_2_1_dpo

    cfg = parse_and_load(
        default_config="experiment/wan_2_1_1_3b_dpo",
        description="Wan 2.1 1.3B Diffusion-DPO Training",
    )
    train_wan_2_1_dpo(cfg)


if __name__ == "__main__":
    main()
