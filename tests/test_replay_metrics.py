"""Tests for the paper-evidence hooks (buffer stats logging + forgetting backfill).

The pure helpers are validated here off-GPU; the current-logprob *forward*
(``compute_replay_current_logprobs``) needs verl + the worker group and is
covered on the cluster.
"""

from __future__ import annotations

import json

import pytest

from replay_buffer.bucket import BucketReplayBuffer
from trainer.replay_metrics import (
    BufferStatsLogger,
    backfill_forgetting,
    flatten_buffer_stats,
    per_row_masked_mean,
)


def _stats():
    return {
        "total_size": 3,
        "total_capacity": 100,
        "step": 5,
        "eviction_type": "priority",
        "within_bucket_sampling": "priority",
        "per_bucket": {
            "workflow": {"size": 2, "soft_target": 10, "fill_ratio": 0.2, "evictions": 1},
            "ops": {"size": 1, "soft_target": 10, "fill_ratio": 0.1, "evictions": 0},
        },
        "reservoir_rejected": {"workflow": 4},
        "priority_active_signals": {"forgetting_risk": 0.5, "diversity": 0.0},
    }


def test_flatten_buffer_stats_scalars():
    flat = flatten_buffer_stats(_stats())
    assert flat["buffer/total_size"] == 3.0
    assert flat["buffer/size/workflow"] == 2.0
    assert flat["buffer/fill_ratio/ops"] == pytest.approx(0.1)
    assert flat["buffer/evictions/workflow"] == 1.0
    assert flat["buffer/fill_ratio_mean"] == pytest.approx(0.15)
    assert flat["buffer/reservoir_rejected/workflow"] == 4.0
    # diversity disabled is verifiable from the logged signal weight.
    assert flat["buffer/signal_weight/diversity"] == 0.0
    assert flat["buffer/signal_weight/forgetting_risk"] == 0.5
    assert all(isinstance(v, float) for v in flat.values())


def test_flatten_no_minmax_substrings():
    # reduce_metrics routes "max"/"min" keys to np.max/np.min; ours must mean.
    flat = flatten_buffer_stats(_stats())
    assert not any("max" in k or "min" in k for k in flat)


def test_buffer_stats_logger_appends_jsonl(tmp_path):
    logger = BufferStatsLogger(tmp_path / "sub" / "stats.jsonl")
    logger.log(1, _stats())
    logger.log(2, _stats())
    lines = (tmp_path / "sub" / "stats.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["step"] == 1
    assert rec["total_size"] == 3


def test_per_row_masked_mean():
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.0, 3.0, 0.0], [2.0, 2.0, 2.0]])
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    means = per_row_masked_mean(values, mask)
    assert means[0] == pytest.approx(2.0)  # mean(1,3)
    assert means[1] == pytest.approx(0.0)  # all-zero mask -> 0


def test_backfill_forgetting_activates_signal():
    buf = BucketReplayBuffer(total_capacity=1000, q_min=100, seed=0)
    # Two identical trajectories; only A's current logprob drifts down.
    for tid in ("traj-A", "traj-B"):
        buf.add_trajectory(
            tid,
            "workflow",
            metadata={
                "trajectory_id": tid,
                "pattern_id": "p0",
                "original_logprobs": [-0.1, -0.1, -0.1, -0.1],
            },
        )

    updated = backfill_forgetting(buf, ["traj-A", "traj-B"], current_means=[-2.0, -0.1])
    assert updated == 2

    meta_a = buf.store.get_metadata("traj-A")
    meta_b = buf.store.get_metadata("traj-B")
    assert len(meta_a["current_logprobs"]) == 4  # repeated to orig length
    # A: drift = -0.1 - (-2.0) = 1.9 -> clipped to 1.0; B: no drift -> 0.
    assert buf.priority_fn.forgetting_risk(meta_a) == pytest.approx(1.0)
    assert buf.priority_fn.forgetting_risk(meta_b) == pytest.approx(0.0)
    # The forgotten trajectory now outranks the stable one (same bucket/pattern).
    assert meta_a["priority"] > meta_b["priority"]


def test_backfill_skips_missing_trajectory():
    buf = BucketReplayBuffer(total_capacity=1000, q_min=100, seed=0)
    assert backfill_forgetting(buf, ["ghost"], current_means=[-1.0]) == 0
