"""Tests for replay_buffer.bucket quota allocation and basic flow."""

import pytest

from replay_buffer.bucket import BucketReplayBuffer, allocate_quota

D9 = [163, 145, 131, 97, 72, 72, 65, 30, 53]


def test_allocate_quota_sums_to_capacity():
    targets = allocate_quota(total_capacity=10000, bucket_task_counts=[1, 1, 1, 1, 1], alpha=0.5)
    assert sum(targets) == 10000
    assert max(targets) - min(targets) <= 1


def test_allocate_quota_alpha_dampens_large_buckets():
    counts = [100, 1]
    prop = allocate_quota(20000, counts, alpha=1.0)
    sqrt = allocate_quota(20000, counts, alpha=0.5)
    assert prop[0] > sqrt[0]
    assert prop[1] < sqrt[1]


def test_allocate_quota_all_zero_counts_uniform_no_crash():
    # all-zero counts -> sum(n^alpha)=0 used to ZeroDivisionError; now uniform.
    t = allocate_quota(100, [0, 0, 0], alpha=0.5)
    assert sum(t) == 100
    assert max(t) - min(t) <= 1


def test_allocate_quota_empty_counts_returns_empty():
    assert allocate_quota(100, []) == []


def test_buffer_add_and_stats():
    buf = BucketReplayBuffer(num_buckets=9, total_capacity=14000, bucket_floors=D9, alpha=0.5)
    buf.set_step(1)
    buf.add_trajectory("traj1", "workflow", metadata={"pattern_id": "p1"})
    buf.add_trajectory("traj2", "ops", metadata={"pattern_id": "p1"})
    stats = buf.stats()
    assert stats["total_size"] == 2
    assert stats["per_bucket"]["workflow"]["size"] == 1
    assert stats["per_bucket"]["qa"]["size"] == 0


def test_buffer_rejects_unknown_bucket():
    buf = BucketReplayBuffer(total_capacity=14000, bucket_floors=D9)
    with pytest.raises(ValueError):
        buf.add_trajectory("x", "NotABucket")


def test_eviction_kicks_in_above_soft_target():
    buf = BucketReplayBuffer(
        num_buckets=2, total_capacity=20,
        bucket_names=["A", "B"], bucket_task_counts=[1, 1],
        bucket_floors=[1, 1], alpha=0.5,
    )
    for i in range(15):
        buf.add_trajectory(f"t{i}", "A", metadata={"pattern_id": f"p{i % 3}"})
    assert buf.store.bucket_size("A") <= buf.soft_target["A"]
    assert buf.store.bucket_size("B") == 0


def test_single_bucket_collapse_for_r0():
    buf = BucketReplayBuffer(
        num_buckets=1, total_capacity=10000,
        bucket_names=["workflow","ops","qa","finance","office","communication","safety","coding","research"],
        bucket_task_counts=[56,44,36,20,11,11,9,2,6],
        bucket_floors=[330],
    )
    assert buf.bucket_names == ["All"]
    assert buf.bucket_task_counts == [195]


def test_reservoir_eviction_caps_size():
    buf = BucketReplayBuffer(
        num_buckets=1, total_capacity=10,
        bucket_names=["All"], bucket_task_counts=[195],
        bucket_floors=[2], eviction_type="reservoir", within_bucket_sampling="uniform", seed=0,
    )
    for i in range(1000):
        buf.add_trajectory({"messages": []}, "All", metadata={"pattern_id": str(i)})
    assert buf.store.bucket_size("All") == 10


def test_persistent_sampler_keeps_state():
    buf = BucketReplayBuffer(total_capacity=14000, bucket_floors=D9, seed=0)
    s1 = buf._sampler
    buf.add_trajectory("t", "workflow", metadata={"pattern_id": "p"})
    buf.sample(1)
    assert buf._sampler is s1


def test_uniform_priority_constant():
    from replay_buffer.priority import UniformPriority
    p = UniformPriority()
    assert p.compute({"reward": 9.9}, {}) == 1.0
    assert p.compute({}, {}) == 1.0


def test_buffer_dump_load_roundtrip(tmp_path):
    buf = BucketReplayBuffer(total_capacity=14000, bucket_floors=D9, seed=0)
    buf.set_step(7)
    for i in range(5):
        buf.add_trajectory({"messages": []}, "workflow", metadata={"pattern_id": f"p{i}"})
        buf.add_trajectory({"messages": []}, "ops", metadata={"pattern_id": "shared"})
    snap = tmp_path / "buf.sqlite"
    buf.dump(snap)
    restored = BucketReplayBuffer(total_capacity=14000, bucket_floors=D9, seed=0)
    restored.load(snap)
    assert restored.store.bucket_size("workflow") == 5
    assert restored._step == 7
    restored.set_step(8)
    assert len(restored.sample(3)) == 3


def test_per_bucket_floors():
    buf = BucketReplayBuffer(
        num_buckets=9, total_capacity=25000,
        bucket_names=["workflow","ops","qa","finance","office","communication","safety","coding","research"],
        bucket_task_counts=[56,44,36,20,11,11,9,2,6],
        bucket_floors=[151,136,124,97,76,76,70,42,61],
    )
    assert buf.bucket_floors["coding"] == 42
    assert buf.bucket_floors["workflow"] == 151
    assert buf.eviction._floor("coding") == 42
    assert buf.eviction._floor("workflow") == 151


def test_bucket_floors_length_mismatch_raises():
    try:
        BucketReplayBuffer(num_buckets=9, bucket_floors=[1, 2, 3])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_bucket_floors_required():
    try:
        BucketReplayBuffer(num_buckets=9)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
