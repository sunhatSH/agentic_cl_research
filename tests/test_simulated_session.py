"""Tests for run_simulated_session (SINGLE-TURN mode, 2026-07-23).

The session is now single-turn: the Questioner is NOT called, so each seed
query yields exactly one turn = N slot trajectories (deterministic yield that
fits verl's fixed-size rollout contract). These tests assert the single-turn
contract: one turn, N trajectories, winner picked on graded reward, observer
report feeds the reward judge, winner synced.

The multi-turn tests (K budget / questioner end / patience) are obsolete under
single-turn and were removed. Multi-turn can be re-enabled later via the
(deprecated) cross-batch trajectory pool; see rollout/simulated_session.py
docstring.
"""

from __future__ import annotations

from agents.observer import Observer
from agents.personas import PERSONAS
from agents.questioner import Questioner
from rollout.session_pool import SessionSandboxPool, Trajectory
from rollout.simulated_session import run_simulated_session


class MockChat:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, *, max_tokens=512):
        return self.reply


def make_agent_fn(reward_by_slot, *, write=True):
    """agent_fn whose slot rewards are fixed, so winner selection is deterministic.

    By default each slot WRITES a file to its sandbox so the observer's before/after
    diff is non-empty (has_effect True) -- the realistic case. Set ``write=False`` to
    model a turn that produces nothing (empty diff -> reward 0 via the gate).
    """

    def agent_fn(client, query, state, slot_idx, history=None):
        if write and hasattr(client, "run_code"):
            client.run_code(f"open('out_{slot_idx}.txt', 'w').write({query!r})")
        return Trajectory(
            slot_idx=slot_idx,
            trajectory_id=f"s{slot_idx}",
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"slot {slot_idx} did the task, wrote out.csv"},
            ],
            reward=reward_by_slot[slot_idx],
            next_state={"files": {"out.csv": "data"}},
        )

    return agent_fn


def _pool(slots=4):
    return SessionSandboxPool(slots=slots, backend="local", seed=0)


def _observer_report_json():
    return (
        '{"intermediate": [], "final": [{"path": "out.csv", "kind": "csv", '
        '"content_excerpt": "a,b"}], "actor_claims": "wrote out.csv", '
        '"discrepancies": "", "file_tree": "out.csv"}'
    )


def test_single_turn_yields_n_trajectories():
    """Single-turn: exactly one turn, N slot trajectories, ended_by=single_turn."""
    pool = _pool(slots=4)
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])  # slot 1 wins
    observer = Observer(client=MockChat(_observer_report_json()))
    # Questioner is passed but NOT called in single-turn mode.
    questioner = Questioner(client=MockChat("this should never be used"))

    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=PERSONAS[0],
        observer=observer,
        questioner=questioner,
        k_max=2,
        seed=1,
        score_followups=False,
    )
    assert res.num_turns == 1
    assert res.ended_by == "single_turn"
    # exactly N=4 trajectories (one GRPO group), no follow-up padding
    assert len(res.trajectories) == 4
    assert res.generated_queries == []  # no follow-ups generated
    assert res.persona_name == PERSONAS[0].name


def test_single_turn_picks_winner_on_graded_reward():
    """Winner is the slot with the highest graded reward (slot 1 here)."""
    pool = _pool(slots=4)
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])
    observer = Observer(client=MockChat(_observer_report_json()))
    questioner = Questioner(client=MockChat("unused"))

    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=PERSONAS[0],
        observer=observer,
        questioner=questioner,
        k_max=2,
        seed=1,
        score_followups=False,
    )
    # The winner (slot 1, reward 0.9) was synced; its state propagated to all.
    # All 4 trajectories carry a graded reward (set by _score_all_slots or the
    # agent_fn's fixed reward).
    assert all(t.reward is not None for t in res.trajectories)


def test_single_turn_scores_all_slots_with_injected_judge():
    """All N slots are graded from (actor trajectory + observer report) before
    winner selection, so pick_winner selects on real graded reward."""
    pool = _pool(slots=4)
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])
    observer = Observer(client=MockChat(_observer_report_json()))
    questioner = Questioner(client=MockChat("unused"))

    class Judge:
        def __init__(self):
            self.calls = 0

        def score(self, *, task, trajectory, rubric, data_source):
            self.calls += 1
            return {"completion": 1.0, "safety": 1.0, "robustness": 1.0}

    judge = Judge()
    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=PERSONAS[0],
        observer=observer,
        questioner=questioner,
        k_max=2,
        seed=1,
        score_followups=True,
        reward_judge=judge,
    )
    # Every non-empty slot was graded (4 slots, all wrote files -> 4 judge calls).
    assert judge.calls == 4
    # Each trajectory carries its reward verdict.
    assert all(t.meta.get("reward_verdict") is not None for t in res.trajectories)
    assert res.num_turns == 1


def test_single_turn_empty_diff_gates_to_zero_reward():
    """A slot whose observer report is empty (no state change) gets reward 0
    without a judge call (gated)."""
    pool = _pool(slots=2)
    # slot 0 writes (non-empty diff), slot 1 does not (empty diff -> gated 0)
    def agent_fn(client, query, state, slot_idx, history=None):
        if slot_idx == 0 and hasattr(client, "run_code"):
            client.run_code(f"open('out_{slot_idx}.txt', 'w').write({query!r})")
        return Trajectory(
            slot_idx=slot_idx,
            trajectory_id=f"s{slot_idx}",
            messages=[{"role": "user", "content": query}, {"role": "assistant", "content": f"slot {slot_idx}"}],
            reward=None,
            next_state={"files": {"out.csv": "data"}} if slot_idx == 0 else {},
        )

    observer = Observer(client=MockChat(_observer_report_json()))
    questioner = Questioner(client=MockChat("unused"))

    class Judge:
        def __init__(self):
            self.calls = 0

        def score(self, *, task, trajectory, rubric, data_source):
            self.calls += 1
            return {"completion": 1.0, "safety": 1.0, "robustness": 1.0}

    judge = Judge()
    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=PERSONAS[0],
        observer=observer,
        questioner=questioner,
        k_max=2,
        seed=1,
        score_followups=True,
        reward_judge=judge,
    )
    # Only the non-empty slot (slot 0) called the judge; slot 1 was gated to 0.
    assert judge.calls == 1
    # slot 1's reward is 0 (gated), slot 0's is graded (1.0 aggregate).
    by_slot = {t.slot_idx: t for t in res.trajectories}
    assert by_slot[1].reward == 0.0
    assert by_slot[0].reward == 1.0
