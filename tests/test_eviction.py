"""Tests for replay_buffer.eviction."""

from replay_buffer.eviction import Eviction
from replay_buffer.store import TrajectoryStore


def test_should_evict_only_above_target_and_above_floor():
    store = TrajectoryStore()
    for i in range(20):
        store.put(f"t{i}", i, {"bucket": "A", "priority": float(i)})
    ev = Eviction(soft_target={"A": 10}, floors={"A": 5})
    # 20 > target 10 AND 20 > floor 5 -> evict
    assert ev.should_evict(store, "A") is True

    ev2 = Eviction(soft_target={"A": 30}, floors={"A": 0})
    # 20 < target 30 -> don't evict
    assert ev2.should_evict(store, "A") is False


def test_select_victim_returns_lowest_priority():
    store = TrajectoryStore()
    store.put("hi", 1, {"bucket": "A", "priority": 0.9})
    store.put("lo", 2, {"bucket": "A", "priority": 0.1})
    store.put("mid", 3, {"bucket": "A", "priority": 0.5})
    ev = Eviction(soft_target={"A": 1}, floors={"A": 0})
    assert ev.select_victim(store, "A") == "lo"


def test_select_victim_returns_none_at_floor():
    store = TrajectoryStore()
    for i in range(3):
        store.put(f"t{i}", i, {"bucket": "A", "priority": float(i)})
    ev = Eviction(soft_target={"A": 1}, floors={"A": 3})
    # Bucket at floor -> cannot evict.
    assert ev.select_victim(store, "A") is None


def test_select_victim_fifo_returns_oldest_inserted():
    store = TrajectoryStore()
    # Insert in a fixed order; priority is intentionally NOT monotonic with age
    # so FIFO cannot accidentally coincide with priority order.
    store.put("first", 1, {"bucket": "A", "priority": 0.9})
    store.put("second", 2, {"bucket": "A", "priority": 0.1})
    store.put("third", 3, {"bucket": "A", "priority": 0.5})
    ev = Eviction(soft_target={"A": 1}, floors={"A": 0}, eviction_type="fifo")
    # FIFO evicts the earliest-inserted regardless of priority.
    assert ev.select_victim(store, "A") == "first"
    store.delete("first")
    assert ev.select_victim(store, "A") == "second"


def test_fifo_reinsert_keeps_original_order():
    store = TrajectoryStore()
    store.put("a", 1, {"bucket": "A", "priority": 0.5})
    store.put("b", 2, {"bucket": "A", "priority": 0.5})
    store.put("c", 3, {"bucket": "A", "priority": 0.5})
    # Re-put "a" (in-place update) must NOT move it to the back of the queue.
    store.put("a", 99, {"bucket": "A", "priority": 0.5})
    ev = Eviction(soft_target={"A": 1}, floors={"A": 0}, eviction_type="fifo")
    assert ev.select_victim(store, "A") == "a"


def test_unknown_eviction_type_raises():
    import pytest

    with pytest.raises(ValueError):
        Eviction(soft_target={"A": 1}, eviction_type="bogus")


def test_pioneer_boost_decays_with_bucket_size():
    store = TrajectoryStore()
    ev = Eviction(soft_target={"A": 100}, pioneer_boost=0.5, pioneer_threshold=10)

    # Empty bucket -> full boost
    boosted = ev.boost_initial_priority(store, "A", base_priority=0.1)
    assert boosted > 0.1
    assert abs(boosted - 0.6) < 1e-6

    # Add 9 -> nearly full, almost no boost
    for i in range(9):
        store.put(f"t{i}", i, {"bucket": "A", "priority": 0.5})
    boosted2 = ev.boost_initial_priority(store, "A", base_priority=0.1)
    assert boosted2 < boosted
    assert abs(boosted2 - 0.15) < 1e-6

    # Add 10th -> at threshold, no boost
    store.put("t9", 9, {"bucket": "A", "priority": 0.5})
    assert ev.boost_initial_priority(store, "A", base_priority=0.1) == 0.1
