"""Tests for trainer.verl_runner.compute_std_metrics (dispersion metrics).

These require torch (only present in the training image); skipped otherwise.
The function operates on a minimal DataProto-like duck type: an object with
``.batch`` (a mapping supporting ``.get``) and ``.non_tensor_batch`` (a dict).
"""

import math

import pytest

from trainer.verl_runner import compute_std_metrics

torch = pytest.importorskip("torch")


class _FakeBatch:
    def __init__(self, batch, non_tensor_batch=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}


def test_compute_std_metrics_empty_on_missing_fields():
    assert compute_std_metrics(_FakeBatch({})) == {}
    assert compute_std_metrics(object()) == {}


def test_compute_std_metrics_reward_and_advantage_std():
    # 3 sequences, per-sequence reward = row sum = [1, 2, 3]
    tlr = torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    resp_mask = torch.tensor([[1, 1], [1, 1], [1, 0]])
    adv = torch.tensor([[0.0, 2.0], [1.0, 1.0], [4.0, 9.0]])  # 9.0 is masked out
    b = _FakeBatch({"token_level_rewards": tlr, "response_mask": resp_mask, "advantages": adv})
    m = compute_std_metrics(b)
    # reward mean/std of [1,2,3]
    assert math.isclose(m["cl/reward_mean"], 2.0, rel_tol=1e-6)
    assert math.isclose(m["cl/reward_std"], math.sqrt(2.0 / 3.0), rel_tol=1e-6)
    # valid advantages = [0,2,1,1,4] (the 9.0 masked) -> std population
    valid = [0.0, 2.0, 1.0, 1.0, 4.0]
    mean = sum(valid) / len(valid)
    exp_std = math.sqrt(sum((x - mean) ** 2 for x in valid) / len(valid))
    assert math.isclose(m["cl/advantage_std"], exp_std, rel_tol=1e-6)


def test_compute_std_metrics_group_reward_std():
    # 4 sequences, 2 GRPO groups. Group g0 rewards [1,3] (std=1.0),
    # group g1 rewards [5,5] (std=0.0). Mean group std = 0.5, max = 1.0.
    tlr = torch.tensor([[1.0], [3.0], [5.0], [5.0]])
    resp_mask = torch.tensor([[1], [1], [1], [1]])
    adv = torch.zeros_like(tlr)
    b = _FakeBatch(
        {"token_level_rewards": tlr, "response_mask": resp_mask, "advantages": adv},
        non_tensor_batch={"uid": ["g0", "g0", "g1", "g1"]},
    )
    m = compute_std_metrics(b)
    assert math.isclose(m["cl/group_reward_std"], 0.5, rel_tol=1e-6)
    assert math.isclose(m["cl/group_reward_std_max"], 1.0, rel_tol=1e-6)
    assert m["cl/num_groups"] == 2.0
