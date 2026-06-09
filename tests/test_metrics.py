"""Tests for eval.metrics."""

import math

from eval.metrics import (
    cl_score,
    new_task_performance,
    old_task_forgetting,
    output_entropy,
    trajectory_diversity,
)


def test_new_task_performance_with_reward():
    rollouts = [{"reward": 0.5}, {"reward": 0.9}, {"reward": 0.2}]
    assert math.isclose(new_task_performance(rollouts), (0.5 + 0.9 + 0.2) / 3)


def test_new_task_performance_with_passed_fallback():
    rollouts = [{"passed": True}, {"passed": False}, {"passed": True}]
    assert math.isclose(new_task_performance(rollouts), 2 / 3)


def test_old_task_forgetting_clips_improvements():
    cur = {"t1": 0.5, "t2": 0.9, "t3": 0.3}
    prev = {"t1": 0.8, "t2": 0.7, "t3": 0.4}
    # Drops: t1 = 0.3, t2 = 0 (improved, clipped), t3 = 0.1
    assert math.isclose(old_task_forgetting(cur, prev), (0.3 + 0.0 + 0.1) / 3)


def test_old_task_forgetting_no_overlap():
    assert old_task_forgetting({"a": 1}, {"b": 1}) == 0.0


def test_cl_score_default_alpha_one():
    assert math.isclose(cl_score(new_perf=0.7, forgetting=0.2), 0.5, abs_tol=1e-9)


def test_output_entropy_uniform_higher_than_peaked():
    uniform = [[0.0, 0.0, 0.0, 0.0]]
    peaked = [[10.0, 0.0, 0.0, 0.0]]
    assert output_entropy(uniform) > output_entropy(peaked)
    # Uniform 4-class: H = ln(4)
    assert math.isclose(output_entropy(uniform), math.log(4), abs_tol=1e-4)


def test_output_entropy_mask_skips_tokens():
    logits = [[1.0, 1.0], [10.0, 0.0]]
    # Without mask: average of both rows
    full = output_entropy(logits)
    # With mask skipping the peaked row: only uniform row contributes
    masked = output_entropy(logits, mask=[1, 0])
    assert masked > full


def test_trajectory_diversity_distinct_n():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    out = trajectory_diversity([[a, b]])
    # No shared tokens or 4-grams.
    assert out["distinct_1"] == 1.0
    assert out["distinct_4"] == 1.0
    assert out["self_bleu_4"] == 0.0


def test_trajectory_diversity_identical_pair():
    a = [1, 2, 3, 4, 5]
    out = trajectory_diversity([[a, a]])
    # Identical trajectories: distinct_1 drops to 0.5 (each token doubled).
    assert math.isclose(out["distinct_1"], 0.5, abs_tol=1e-6)
    # Self-BLEU 4-gram: full overlap -> 1.0
    assert math.isclose(out["self_bleu_4"], 1.0, abs_tol=1e-6)
