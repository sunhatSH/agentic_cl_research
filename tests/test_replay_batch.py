"""Tests for replay batch warmup ramp and unlabeled-bucket skipping."""

from __future__ import annotations

from replay_buffer.bucket import BucketReplayBuffer
from trainer.replay_batch import effective_replay_batch_size
from trainer.trajectory_adapter import extract_trajectories_from_batch


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def _buffer_with(n: int) -> BucketReplayBuffer:
    buf = BucketReplayBuffer(total_capacity=14000, q_min=500, seed=0)
    for i in range(n):
        buf.add_trajectory("t", "Workflow", metadata={"pattern_id": f"p{i}"})
    return buf


def test_warmup_disabled_returns_full_batch():
    buf = _buffer_with(5)
    assert effective_replay_batch_size(buf, batch_size=32, warmup_size=0) == 32


def test_warmup_empty_buffer_returns_zero():
    buf = _buffer_with(0)
    assert effective_replay_batch_size(buf, batch_size=32, warmup_size=1000) == 0


def test_warmup_ramps_linearly():
    buf = _buffer_with(500)
    # 500 / 1000 = 0.5 -> 16
    assert effective_replay_batch_size(buf, batch_size=32, warmup_size=1000) == 16


def test_warmup_caps_at_full_batch():
    buf = _buffer_with(2000)
    # ratio capped at 1.0 -> full batch
    assert effective_replay_batch_size(buf, batch_size=32, warmup_size=1000) == 32


def test_unlabeled_bucket_skipped_by_default():
    # No bucket / category -> skipped, not dumped into a default bucket (B12).
    batch = FakeBatch({"messages": [[{"role": "assistant", "content": "a"}]]})
    assert extract_trajectories_from_batch(batch) == []


def test_explicit_default_bucket_catch_all():
    batch = FakeBatch({"messages": [[{"role": "assistant", "content": "a"}]]})
    out = extract_trajectories_from_batch(batch, default_bucket="Workflow")
    assert len(out) == 1
    assert out[0][1] == "Workflow"
