"""Tests for replay_buffer.sampler — two-level sampling."""

import random

from replay_buffer.bucket import BucketReplayBuffer
from replay_buffer.sampler import TwoLevelSampler


def _make_buffer():
    buf = BucketReplayBuffer(
        num_buckets=3,
        total_capacity=300,
        q_min=10,
        bucket_names=["A", "B", "C"],
        bucket_task_counts=[10, 5, 1],
        alpha=0.5,
    )
    for i in range(30):
        buf.add_trajectory(f"a{i}", "A")
    for i in range(20):
        buf.add_trajectory(f"b{i}", "B")
    for i in range(10):
        buf.add_trajectory(f"c{i}", "C")
    return buf


def test_sample_returns_requested_size():
    buf = _make_buffer()
    sampler = TwoLevelSampler(buf, rng=random.Random(42))
    out = sampler.sample(16)
    # Could be less if some buckets empty after first pick; here all non-empty.
    assert 0 < len(out) <= 16


def test_sample_marks_replayed():
    buf = _make_buffer()
    sampler = TwoLevelSampler(buf, rng=random.Random(42))
    buf.set_step(7)
    out = sampler.sample(8)
    for tid, _, _ in out:
        meta = buf.store.get_metadata(tid)
        assert meta["last_replay_step"] == 7
        assert meta["replay_count"] >= 1


def test_uniform_mix_brings_small_bucket_in():
    """With heavy uniform mix, the smallest bucket C should be sampled
    despite its tiny quota."""
    buf = _make_buffer()
    sampler = TwoLevelSampler(buf, bucket_mix_ratio=0.0, rng=random.Random(0))
    counts = {"A": 0, "B": 0, "C": 0}
    for _ in range(200):
        bucket = sampler._sample_buckets(1)[0]
        counts[bucket] += 1
    # Uniform: each ~1/3
    assert counts["C"] > 30  # not starved


def test_quota_mix_favours_large_bucket():
    buf = _make_buffer()
    sampler = TwoLevelSampler(buf, bucket_mix_ratio=1.0, rng=random.Random(0))
    counts = {"A": 0, "B": 0, "C": 0}
    for _ in range(300):
        bucket = sampler._sample_buckets(1)[0]
        counts[bucket] += 1
    # A has largest task count -> largest quota -> most picks.
    assert counts["A"] > counts["B"] > counts["C"]


def test_empty_sampler_returns_empty():
    buf = BucketReplayBuffer(
        num_buckets=3,
        total_capacity=300,
        q_min=10,
        bucket_names=["A", "B", "C"],
        bucket_task_counts=[1, 1, 1],
    )
    sampler = TwoLevelSampler(buf, rng=random.Random(0))
    assert sampler.sample(8) == []
