"""Tests for replay_buffer.store."""

from replay_buffer.store import TrajectoryStore


def test_put_and_get():
    s = TrajectoryStore()
    s.put("t1", [1, 2, 3], {"bucket": "Workflow", "priority": 0.5, "insert_step": 0})
    traj, meta = s.get("t1")
    assert traj == [1, 2, 3]
    assert meta["priority"] == 0.5
    assert "t1" in s
    assert len(s) == 1


def test_bucket_indexing():
    s = TrajectoryStore()
    for i in range(5):
        s.put(f"t{i}", i, {"bucket": "Workflow", "priority": float(i)})
    for i in range(3):
        s.put(f"u{i}", i, {"bucket": "SysOps", "priority": float(i)})
    assert s.bucket_size("Workflow") == 5
    assert s.bucket_size("SysOps") == 3
    assert s.bucket_size("Dialogue") == 0
    assert set(s.list_by_bucket("Workflow")) == {f"t{i}" for i in range(5)}


def test_top_and_bottom_k_priority():
    s = TrajectoryStore()
    for i in range(10):
        s.put(f"t{i}", i, {"bucket": "B", "priority": float(i)})
    top3 = s.top_k_priority("B", 3)
    assert set(top3) == {"t9", "t8", "t7"}
    bot2 = s.bottom_k_priority("B", 2)
    assert set(bot2) == {"t0", "t1"}


def test_delete_removes_from_indexes():
    s = TrajectoryStore()
    s.put("t1", "x", {"bucket": "A", "priority": 0.1, "pattern_id": "p1"})
    s.put("t2", "y", {"bucket": "A", "priority": 0.2, "pattern_id": "p1"})
    s.delete("t1")
    assert "t1" not in s
    assert s.bucket_size("A") == 1
    assert s.list_by_pattern("p1") == ["t2"]


def test_update_metadata_moves_indexes():
    s = TrajectoryStore()
    s.put("t1", "x", {"bucket": "A", "priority": 0.1, "pattern_id": "p1"})
    s.update_metadata("t1", bucket="B", pattern_id="p2", priority=0.9)
    assert s.bucket_size("A") == 0
    assert s.bucket_size("B") == 1
    assert s.list_by_pattern("p1") == []
    assert s.list_by_pattern("p2") == ["t1"]
    assert s.get_metadata("t1")["priority"] == 0.9


def test_long_unreplayed():
    s = TrajectoryStore()
    s.put("recent", "x", {"bucket": "A", "priority": 0.1, "last_replay_step": 95})
    s.put("stale", "y", {"bucket": "A", "priority": 0.1, "last_replay_step": 10})
    s.put("never", "z", {"bucket": "A", "priority": 0.1})
    stale = set(s.long_unreplayed(threshold_steps=20, current_step=100))
    assert "stale" in stale
    assert "never" in stale
    assert "recent" not in stale


def test_pattern_counts_restricted_to_bucket():
    s = TrajectoryStore()
    s.put("a1", "x", {"bucket": "A", "priority": 0.1, "pattern_id": "p1"})
    s.put("a2", "x", {"bucket": "A", "priority": 0.1, "pattern_id": "p1"})
    s.put("b1", "y", {"bucket": "B", "priority": 0.1, "pattern_id": "p1"})
    assert s.pattern_counts(bucket="A") == {"p1": 2}
    assert s.pattern_counts() == {"p1": 3}


def test_invalid_metadata_rejected():
    s = TrajectoryStore()
    try:
        s.put("t1", "x", {"priority": 0.1})  # missing bucket
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
