"""Tests for replay_buffer.bucket quota allocation and basic flow."""

import pytest

from replay_buffer.bucket import BucketReplayBuffer, allocate_quota


def test_allocate_quota_sums_to_capacity():
    targets = allocate_quota(
        total_capacity=25000, q_min=2000,
        bucket_task_counts=[54, 52, 38, 18, 12, 11, 10], alpha=0.5,
    )
    assert sum(targets) == 25000
    assert all(t >= 2000 for t in targets)


def test_allocate_quota_respects_q_min():
    # Equal weights -> equal soft targets (above q_min).
    targets = allocate_quota(total_capacity=10000, q_min=1000,
                             bucket_task_counts=[1, 1, 1, 1, 1], alpha=0.5)
    assert sum(targets) == 10000
    assert max(targets) - min(targets) <= 1  # rounding


def test_allocate_quota_alpha_dampens_large_buckets():
    # alpha=1.0 (proportional) vs alpha=0.5 (sqrt) -- alpha=0.5 should
    # give the smallest bucket more share.
    counts = [100, 1]
    prop = allocate_quota(20000, 1000, counts, alpha=1.0)
    sqrt = allocate_quota(20000, 1000, counts, alpha=0.5)
    assert prop[0] > sqrt[0]
    assert prop[1] < sqrt[1]


def test_allocate_quota_raises_when_q_min_too_large():
    with pytest.raises(ValueError):
        allocate_quota(total_capacity=1000, q_min=500,
                       bucket_task_counts=[1, 1, 1], alpha=0.5)


def test_buffer_add_and_stats():
    buf = BucketReplayBuffer(
        num_buckets=7, total_capacity=14000, q_min=500,
        alpha=0.5,
    )
    buf.set_step(1)
    buf.add_trajectory("traj1", "Workflow", metadata={"pattern_id": "p1"})
    buf.add_trajectory("traj2", "SysOps", metadata={"pattern_id": "p1"})
    stats = buf.stats()
    assert stats["total_size"] == 2
    assert stats["per_bucket"]["Workflow"]["size"] == 1
    assert stats["per_bucket"]["Dialogue"]["size"] == 0


def test_buffer_rejects_unknown_bucket():
    buf = BucketReplayBuffer(total_capacity=14000, q_min=500)
    with pytest.raises(ValueError):
        buf.add_trajectory("x", "NotABucket")


def test_eviction_kicks_in_above_soft_target():
    # Tiny buffer to force eviction quickly.
    buf = BucketReplayBuffer(
        num_buckets=2, total_capacity=20, q_min=2,
        bucket_names=["A", "B"], bucket_task_counts=[1, 1], alpha=0.5,
    )
    # Soft target for A = 10. Fill 15 -> should evict 5.
    for i in range(15):
        buf.add_trajectory(f"t{i}", "A", metadata={"pattern_id": f"p{i % 3}"})
    size_a = buf.store.bucket_size("A")
    assert size_a <= buf.soft_target["A"] + 1
    # Bucket B untouched (no cross-bucket displacement).
    assert buf.store.bucket_size("B") == 0
