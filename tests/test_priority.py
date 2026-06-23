"""Tests for replay_buffer.priority — 4-signal fusion."""

import math

import pytest

from replay_buffer.priority import Priority, RewardPriority


def test_alpha_must_sum_to_one():
    with pytest.raises(ValueError):
        Priority(alpha=(0.5, 0.5, 0.5, 0.5))
    Priority(alpha=(1.0, 0.0, 0.0, 0.0))  # OK


def test_forgetting_risk_positive_drift():
    p = Priority(alpha=(1.0, 0.0, 0.0, 0.0))
    # Original logprobs higher than current -> drift positive -> forgetting.
    traj = {"original_logprobs": [-1.0, -1.0, -1.0], "current_logprobs": [-1.5, -1.5, -1.5]}
    score = p.compute(traj, {"pattern_counts": {}})
    assert math.isclose(score, 0.5, abs_tol=1e-6)


def test_forgetting_risk_negative_drift_clipped_to_zero():
    p = Priority(alpha=(1.0, 0.0, 0.0, 0.0))
    # Model improved on this trajectory.
    traj = {"original_logprobs": [-2.0, -2.0], "current_logprobs": [-1.0, -1.0]}
    assert p.compute(traj, {"pattern_counts": {}}) == 0.0


def test_rarity_inverse_log_frequency():
    p = Priority(alpha=(0.0, 1.0, 0.0, 0.0))
    rare = p.compute({"pattern_id": "p1"}, {"pattern_counts": {"p1": 0}})
    common = p.compute({"pattern_id": "p1"}, {"pattern_counts": {"p1": 100}})
    assert rare > common > 0


def test_rarity_normalized_within_unit_interval():
    # Bug B3: rarity must stay in [0, 1]; count=0 yields exactly 1.0.
    p = Priority(alpha=(0.0, 1.0, 0.0, 0.0))
    for count in (0, 1, 10, 1000):
        r = p.compute({"pattern_id": "p"}, {"pattern_counts": {"p": count}})
        assert 0.0 <= r <= 1.0
    assert math.isclose(p.compute({"pattern_id": "p"}, {"pattern_counts": {"p": 0}}), 1.0, abs_tol=1e-9)


def test_zero_alpha_short_circuits_signal():
    """If a weight is 0, the corresponding signal is not even read.

    Enforces project rule (memory: short_circuit_zero_coefficient).
    """
    p = Priority(alpha=(1.0, 0.0, 0.0, 0.0))
    # Pass a trajectory with NO embedding / pattern_id; rarity & diversity
    # would normally short-circuit to 0 anyway, but verify the loss term
    # weight zeroing also skips reading those fields cleanly.
    score = p.compute(
        {"original_logprobs": [-1.0], "current_logprobs": [-1.2]},
        {},
    )
    assert score > 0


def test_difficulty_one_minus_success_rate():
    p = Priority(alpha=(0.0, 0.0, 0.0, 1.0))
    assert math.isclose(p.compute({"success_rate": 0.3}, {}), 0.7, abs_tol=1e-6)
    assert p.compute({"success_rate": 1.0}, {}) == 0.0


def test_reward_priority_returns_reward_directly():
    rp = RewardPriority()
    assert rp.compute({"reward": 0.8}, {}) == 0.8
    assert rp.compute({}, {}) == 0.0
