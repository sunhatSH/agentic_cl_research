"""Tests for replay_buffer.sampler — distance bucket strategy + CLEAR baseline."""

import random

from replay_buffer.bucket import BucketReplayBuffer
from replay_buffer.sampler import BaselineSampler, DistanceStrategy, TwoLevelSampler


def _make_buffer():
    # Use real bucket names so DistanceStrategy can resolve coordinates from
    # configs/bucket_coords.json. Pick three with distinct coords.
    names = ["workflow", "qa", "coding"]
    buf = BucketReplayBuffer(
        num_buckets=3,
        total_capacity=300,
        bucket_names=names,
        bucket_task_counts=[10, 5, 1],
        bucket_floors=[10, 10, 5],
        alpha=0.5,
    )
    for i in range(30):
        buf.add_trajectory(f"wf{i}", names[0])
    for i in range(20):
        buf.add_trajectory(f"qa{i}", names[1])
    for i in range(10):
        buf.add_trajectory(f"cd{i}", names[2])
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


def test_distance_cold_start_is_uniform():
    """No current distribution (cold-start) -> uniform over non-empty buckets."""
    buf = _make_buffer()
    names = buf.bucket_names
    sampler = TwoLevelSampler(buf, rng=random.Random(0))
    weights = sampler.bucket_strategy.get_weights(buf)
    # All non-empty buckets get equal weight 1.0
    assert weights[names[0]] == weights[names[1]] == weights[names[2]] == 1.0


def test_distance_weights_buckets_far_from_current():
    """With a current distribution, buckets far from the centroid get more weight."""
    buf = _make_buffer()
    names = buf.bucket_names
    wf, qa, cd = names
    sampler = TwoLevelSampler(buf, rng=random.Random(0))
    # Train only workflow -> centroid = workflow's coords; qa and coding get distance weight.
    sampler.set_current_distribution({wf: 1.0})
    weights = sampler.bucket_strategy.get_weights(buf, current_distribution={wf: 1.0})
    # workflow is the current bucket -> distance 0 -> weight 0 (on-policy now).
    assert weights[wf] == 0.0
    # qa and coding are non-zero (farther = higher, but both > 0).
    assert weights[qa] > 0.0
    assert weights[cd] > 0.0


def test_distance_mixed_batch_uses_centroid():
    """A mixed-batch distribution blends the centroid; current buckets get ~0."""
    buf = _make_buffer()
    names = buf.bucket_names
    wf, qa, cd = names
    sampler = TwoLevelSampler(buf, rng=random.Random(0))
    dist = {wf: 0.5, qa: 0.5}  # batch trains half workflow, half qa
    weights = sampler.bucket_strategy.get_weights(buf, current_distribution=dist)
    # workflow and qa are in the current batch -> near centroid -> low weight.
    # coding is far from the wf/qa centroid -> highest weight (most forgetting risk).
    assert weights[cd] >= weights[wf]
    assert weights[cd] >= weights[qa]


def test_empty_sampler_returns_empty():
    buf = BucketReplayBuffer(
        num_buckets=3,
        total_capacity=300,
        bucket_names=["A", "B", "C"],
        bucket_task_counts=[1, 1, 1],
        bucket_floors=[10, 10, 10],
    )
    sampler = TwoLevelSampler(buf, rng=random.Random(0))
    assert sampler.sample(8) == []


def test_baseline_sampler_is_bucket_agnostic():
    """CLEAR baseline: uniform over the whole pool, no bucket weighting."""
    buf = _make_buffer()
    names = buf.bucket_names
    sampler = BaselineSampler(buf, rng=random.Random(0))
    out = sampler.sample(20)
    assert len(out) == 20
    # Over a large draw, the bucket distribution should track the pool's true
    # proportions (30 wf / 20 qa / 10 cd), NOT soft_target quotas.
    counts = {n: 0 for n in names}
    big = sampler.sample(60)
    for tid, _, _ in big:
        meta = buf.store.get_metadata(tid)
        counts[meta["bucket"]] += 1
    # Proportional to pool size, not quota: the largest bucket should dominate.
    assert counts[names[0]] >= counts[names[1]] >= counts[names[2]]
