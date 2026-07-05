"""Tests for replay_buffer.bucket quota allocation and basic flow."""

import pytest

from replay_buffer.bucket import BucketReplayBuffer, allocate_quota


def test_allocate_quota_sums_to_capacity():
    targets = allocate_quota(
        total_capacity=25000,
        q_min=2000,
        bucket_task_counts=[56, 44, 36, 20, 11, 11, 9, 2, 6],
        alpha=0.5,
    )
    assert sum(targets) == 25000
    assert all(t >= 2000 for t in targets)


def test_allocate_quota_respects_q_min():
    # Equal weights -> equal soft targets (above q_min).
    targets = allocate_quota(total_capacity=10000, q_min=1000, bucket_task_counts=[1, 1, 1, 1, 1], alpha=0.5)
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
        allocate_quota(total_capacity=1000, q_min=500, bucket_task_counts=[1, 1, 1], alpha=0.5)


def test_buffer_add_and_stats():
    buf = BucketReplayBuffer(
        num_buckets=9,
        total_capacity=14000,
        q_min=500,
        alpha=0.5,
    )
    buf.set_step(1)
    buf.add_trajectory("traj1", "workflow", metadata={"pattern_id": "p1"})
    buf.add_trajectory("traj2", "ops", metadata={"pattern_id": "p1"})
    stats = buf.stats()
    assert stats["total_size"] == 2
    assert stats["per_bucket"]["workflow"]["size"] == 1
    assert stats["per_bucket"]["qa"]["size"] == 0


def test_buffer_rejects_unknown_bucket():
    buf = BucketReplayBuffer(total_capacity=14000, q_min=500)
    with pytest.raises(ValueError):
        buf.add_trajectory("x", "NotABucket")


def test_eviction_kicks_in_above_soft_target():
    # Tiny buffer to force eviction quickly.
    buf = BucketReplayBuffer(
        num_buckets=2,
        total_capacity=20,
        q_min=2,
        bucket_names=["A", "B"],
        bucket_task_counts=[1, 1],
        alpha=0.5,
    )
    # Soft target for A = 10. Fill 15 -> should evict 5.
    for i in range(15):
        buf.add_trajectory(f"t{i}", "A", metadata={"pattern_id": f"p{i % 3}"})
    size_a = buf.store.bucket_size("A")
    # Insert-before-evict keeps steady-state size at soft_target (bug A5).
    assert size_a <= buf.soft_target["A"]
    # Bucket B untouched (no cross-bucket displacement).
    assert buf.store.bucket_size("B") == 0


def test_single_bucket_collapse_for_r0():
    # num_buckets=1 with inherited 9-name list -> collapse to one "All" bucket.
    buf = BucketReplayBuffer(
        num_buckets=1,
        total_capacity=10000,
        q_min=10000,
        bucket_names=["workflow", "ops", "qa", "finance", "office", "communication", "safety", "coding", "research"],
        bucket_task_counts=[56, 44, 36, 20, 11, 11, 9, 2, 6],
    )
    assert buf.bucket_names == ["All"]
    assert buf.bucket_task_counts == [195]


def test_reservoir_eviction_caps_size():
    buf = BucketReplayBuffer(
        num_buckets=1,
        total_capacity=10,
        q_min=10,
        bucket_names=["All"],
        bucket_task_counts=[195],
        eviction_type="reservoir",
        within_bucket_sampling="uniform",
        seed=0,
    )
    for i in range(1000):
        buf.add_trajectory({"messages": []}, "All", metadata={"pattern_id": str(i)})
    # Reservoir never exceeds capacity even though q_min == capacity.
    assert buf.store.bucket_size("All") == 10


def test_persistent_sampler_keeps_state():
    buf = BucketReplayBuffer(total_capacity=14000, q_min=500, seed=0)
    s1 = buf._sampler
    buf.add_trajectory("t", "workflow", metadata={"pattern_id": "p"})
    buf.sample(1)
    # Same sampler object across sample() calls (bug A4).
    assert buf._sampler is s1


def test_uniform_priority_constant():
    from replay_buffer.priority import UniformPriority

    p = UniformPriority()
    assert p.compute({"reward": 9.9}, {}) == 1.0
    assert p.compute({}, {}) == 1.0


def test_buffer_dump_load_roundtrip(tmp_path):
    buf = BucketReplayBuffer(total_capacity=14000, q_min=500, seed=0)
    buf.set_step(7)
    for i in range(5):
        buf.add_trajectory({"messages": []}, "workflow", metadata={"pattern_id": f"p{i}"})
        buf.add_trajectory({"messages": []}, "ops", metadata={"pattern_id": "shared"})
    snap = tmp_path / "buf.sqlite"
    buf.dump(snap)

    restored = BucketReplayBuffer(total_capacity=14000, q_min=500, seed=0)
    restored.load(snap)
    assert restored.store.bucket_size("workflow") == 5
    assert restored.store.bucket_size("ops") == 5
    assert restored._step == 7
    assert restored._seen_counts["workflow"] == 5
    # Restored buffer is functional (can sample).
    restored.set_step(8)
    assert len(restored.sample(3)) == 3
