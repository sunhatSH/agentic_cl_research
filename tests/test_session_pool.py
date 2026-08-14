"""Tests for rollout.session_pool + rollout.scheduler (Gap C, mock backend)."""

import random

from rollout.scheduler import RolloutScheduler, SessionSpec
from rollout.session_pool import (
    SessionSandboxPool,
    Trajectory,
    select_winner_with_fallback,
)


class MockSandbox:
    """Records kills; carries no real state (state lives in the pool)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.killed = False

    def kill(self):
        self.killed = True


def _factory(backend, **kwargs):
    return MockSandbox(**kwargs)


def _make_pool(**kw):
    return SessionSandboxPool(
        slots=8,
        backend="mock",
        sandbox_factory=_factory,
        initial_state_factory=lambda: {"files": {"seed.txt": "x"}, "step": 0},
        **kw,
    )


def _agent_fn(reward_by_slot, seen_history=None):
    """Build an agent_fn whose reward depends on slot idx (deterministic).

    If ``seen_history`` is a list, records the history passed to slot 0 each call
    (to assert winner-context propagation).
    """

    def fn(client, query, state, slot_idx, history=None):
        if seen_history is not None and slot_idx == 0:
            seen_history.append(list(history or []))
        new_state = dict(state)
        new_state["step"] = state.get("step", 0) + 1
        new_state["files"] = {**state.get("files", {}), f"slot{slot_idx}.out": query}
        return Trajectory(
            slot_idx=slot_idx,
            trajectory_id=f"t-{slot_idx}",
            messages=[{"role": "assistant", "content": f"slot{slot_idx}:{query}"}],
            reward=reward_by_slot(slot_idx, query),
            next_state=new_state,
        )

    return fn


def test_spawn_identical_start():
    pool = _make_pool()
    pool.spawn()
    states = [s.state for s in pool._slots]
    assert len(states) == 8
    assert all(st == states[0] for st in states)  # bit-identical start
    pool.destroy_all()


def test_winner_sync_aligns_all_slots():
    pool = _make_pool()
    pool.spawn()
    # slot 3 is the clear winner
    trajs = pool.run_query("q1", _agent_fn(lambda i, q: 1.0 if i == 3 else 0.0))
    winner = pool.pick_winner(trajs)
    assert winner == 3
    pool.sync_to_winner(winner, trajs)
    # all slots now carry slot3's state (slot3.out present, others' not)
    for s in pool._slots:
        assert "slot3.out" in s.state["files"]
        assert "slot0.out" not in s.state["files"]
    # conversation history = winner trajectory only
    assert pool.session_history == [{"role": "assistant", "content": "slot3:q1"}]
    pool.destroy_all()


def test_destroy_all_kills_and_no_master_writeback():
    pool = _make_pool()
    pool.spawn()
    clients = [s.client for s in pool._slots]
    pool.run_query("q", _agent_fn(lambda i, q: float(i)))
    pool.destroy_all()
    assert all(c.killed for c in clients)
    assert pool._slots == []  # nothing persisted as a new master


def test_winner_fallback_rules():
    rng = random.Random(0)
    # scorer error -> None (keep previous boundary)
    assert select_winner_with_fallback([1.0, None, 0.0], ["a", "b", "c"], rng) is None
    # all equal (incl all-zero) -> some valid index (random, seeded)
    w = select_winner_with_fallback([0.0, 0.0, 0.0], ["a", "b", "c"], rng)
    assert w in (0, 1, 2)
    # normal -> argmax
    assert select_winner_with_fallback([0.1, 0.9, 0.2], ["a", "b", "c"], rng) == 1


def test_scorer_error_skips_sync_keeps_state():
    pool = _make_pool()
    pool.spawn()

    def err_agent(client, query, state, slot_idx, history=None):
        return Trajectory(slot_idx=slot_idx, trajectory_id=f"t{slot_idx}", reward=None)

    trajs = pool.run_query("q1", err_agent)
    assert pool.pick_winner(trajs) is None
    # history stays empty (no winner to adopt)
    assert pool.session_history == []
    pool.destroy_all()


def test_winner_history_propagates_to_next_query():
    """§3 ①: query_{k+1}'s prompt prefix = winner(q_k) messages, not own past."""
    pool = _make_pool()
    seen = []
    agent = _agent_fn(lambda i, q: 1.0 if i == 5 else 0.0, seen_history=seen)
    pool.run_session(["q1", "q2"], agent)
    # q1 saw empty history; q2 saw winner(q1) = slot5's message
    assert seen[0] == []
    assert seen[1] == [{"role": "assistant", "content": "slot5:q1"}]


def test_run_session_collects_all_trajectories():
    pool = _make_pool()
    trajs = pool.run_session(["q1", "q2", "q3"], _agent_fn(lambda i, q: float(i)))
    # 8 slots × 3 queries = 24 trajectories
    assert len(trajs) == 24
    # winner each query is slot 7 (highest reward); history has 3 winner msgs
    assert len(pool.session_history) == 3
    assert all(m["content"].startswith("slot7:") for m in pool.session_history)


def test_run_checkers_on_winner_state():
    pool = _make_pool()
    pool.spawn()
    trajs = pool.run_query("q1", _agent_fn(lambda i, q: 1.0 if i == 2 else 0.0))
    winner = pool.pick_winner(trajs)
    checkers = [
        {"type": "file_exists", "path": "slot2.out"},
        {"type": "file_exists", "path": "nope.out"},
    ]
    res = pool.run_checkers(checkers, winner)
    assert res == {"0": True, "1": False}
    pool.destroy_all()


def test_scheduler_16x8_topology():
    sched = RolloutScheduler(
        _agent_fn(lambda i, q: float(i)),
        sessions_per_step=4,
        slots=8,
        backend="mock",
        pool_factory=lambda spec, seed: _make_pool(persona=spec.persona, seed=seed),
    )
    specs = [SessionSpec(session_id=f"sess{j}", queries=["q1", "q2"]) for j in range(4)]
    trajs = sched.run_step(specs)
    # 4 sessions × 8 slots × 2 queries = 64
    assert len(trajs) == 64
    assert {t.meta["session_id"] for t in trajs} == {f"sess{j}" for j in range(4)}
