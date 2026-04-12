"""Tests for vrl.rollouts.collectors (WanCollector config, SDE window, request template)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# WanCollectorConfig — sde_window (Gap 4)
# ---------------------------------------------------------------------------

class TestWanCollectorSdeWindow:
    def test_sde_window_disabled(self) -> None:
        """sde_window_size=0 → _get_sde_window returns None."""
        from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig

        cfg = WanCollectorConfig(sde_window_size=0)
        collector = WanCollector(wan_model=None, reward_fn=None, config=cfg)
        assert collector._get_sde_window() is None

    def test_sde_window_range(self) -> None:
        """sde_window_size>0 → returns a window within the range."""
        from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig

        cfg = WanCollectorConfig(sde_window_size=5, sde_window_range=(0, 20))
        collector = WanCollector(wan_model=None, reward_fn=None, config=cfg)
        for _ in range(20):  # Random — test multiple times
            window = collector._get_sde_window()
            assert window is not None
            start, end = window
            assert start >= 0
            assert end == start + 5
            assert start <= 15  # max start = 20 - 5

    def test_sde_window_fixed_size(self) -> None:
        """Window always has exactly sde_window_size steps."""
        from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig

        cfg = WanCollectorConfig(sde_window_size=3, sde_window_range=(2, 10))
        collector = WanCollector(wan_model=None, reward_fn=None, config=cfg)
        for _ in range(10):
            window = collector._get_sde_window()
            assert window[1] - window[0] == 3


# ---------------------------------------------------------------------------
# WanCollectorConfig — config defaults (Gaps 3, 4, 5)
# ---------------------------------------------------------------------------

class TestWanCollectorConfig:
    def test_defaults(self) -> None:
        """Verify new config fields have correct defaults."""
        from vrl.rollouts.collectors.wan import WanCollectorConfig

        cfg = WanCollectorConfig()
        assert cfg.kl_reward == 0.0
        assert cfg.sde_window_size == 0
        assert cfg.sde_window_range == (0, 10)
        assert cfg.same_latent is False

    def test_custom_values(self) -> None:
        """Can set custom values for all gap fields."""
        from vrl.rollouts.collectors.wan import WanCollectorConfig

        cfg = WanCollectorConfig(
            kl_reward=0.1,
            sde_window_size=10,
            sde_window_range=(5, 30),
            same_latent=True,
        )
        assert cfg.kl_reward == 0.1
        assert cfg.sde_window_size == 10
        assert cfg.sde_window_range == (5, 30)
        assert cfg.same_latent is True


# ---------------------------------------------------------------------------
# Regression: Bug 1 — collector with request_template (no explicit request)
# ---------------------------------------------------------------------------

class TestWanCollectorRequestTemplate:
    def test_no_request_no_template_raises(self) -> None:
        """Without request kwarg OR template, collect() should raise ValueError."""
        import asyncio
        from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig

        collector = WanCollector(
            wan_model=None, reward_fn=None,
            config=WanCollectorConfig(),
            request_template=None,
        )
        with pytest.raises(ValueError, match="request_template"):
            asyncio.run(collector.collect(["hello"]))

    def test_template_builds_request(self) -> None:
        """With request_template, collect() builds request from template + prompt."""
        from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig
        from vrl.models.base import VideoGenerationRequest

        template = VideoGenerationRequest(
            prompt="placeholder",
            model_name="wan-A14B",
            width=832,
            height=480,
        )

        built_requests = []

        class _EarlyExit(Exception):
            pass

        class MockWanModel:
            async def encode_text(self, request, state):
                built_requests.append(request)
                raise _EarlyExit("captured request, stop early")

        collector = WanCollector(
            wan_model=MockWanModel(),
            reward_fn=None,
            config=WanCollectorConfig(),
            request_template=template,
        )

        import asyncio
        try:
            asyncio.run(collector.collect(["a sunset over the ocean"]))
        except _EarlyExit:
            pass

        assert len(built_requests) == 1
        assert built_requests[0].prompt == "a sunset over the ocean"
        assert built_requests[0].width == 832  # inherited from template
