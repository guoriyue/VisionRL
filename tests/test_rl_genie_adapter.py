from __future__ import annotations

import torch

from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.consumers.rl.genie_adapter import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter


def _stub_adapter() -> GenieWorldModelAdapter:
    runner = GenieRunner(device="cpu")
    runner._mode = "stub"
    runner.load = lambda: "stub"  # type: ignore[method-assign]
    return GenieWorldModelAdapter(
        runner=runner,
        spec=GenieRLSpec(history_frames=4, spatial_h=16, spatial_w=16),
        device="cpu",
    )


def test_genie_adapter_predict_next_preserves_shape_and_changes_state() -> None:
    adapter = _stub_adapter()
    initial = adapter.sample_initial_state(seed=7)
    action = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    next_state = adapter.predict_next(initial, action)

    assert next_state.shape == initial.shape
    assert not torch.equal(next_state, initial)


def test_genie_adapter_distinguishes_different_actions() -> None:
    adapter = _stub_adapter()
    initial = adapter.sample_initial_state(seed=11)
    shift_left = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    token_plus = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    left_state = adapter.predict_next(initial, shift_left)
    plus_state = adapter.predict_next(initial, token_plus)

    assert not torch.equal(left_state, plus_state)


def test_genie_reward_reports_success_for_matching_goal() -> None:
    adapter = _stub_adapter()
    reward_fn = GenieTokenReward(adapter.spec, success_threshold=0.02, reward_scale=4.0)
    goal = adapter.sample_goal_state(seed=5)

    reward, terminated, info = reward_fn.evaluate(goal, goal.clone())

    assert reward.shape == (1,)
    assert bool(terminated.item()) is True
    assert float(info["token_l1"].item()) == 0.0
