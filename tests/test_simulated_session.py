"""Tests for run_simulated_session (Algorithm 1, doc §5).

Off-network / off-verl: uses the existing SessionSandboxPool with a mock
agent_fn, mock Observer/Questioner, and an injected reward judge.
"""

from __future__ import annotations

import random

from agents.observer import Observer
from agents.personas import PERSONAS
from agents.questioner import Questioner
from agents.schema import ObservationReport
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
    model a turn that produces nothing (empty diff -> failed turn).
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


def test_session_runs_k_turns_then_stops_on_budget():
    pool = _pool()
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])  # slot 1 wins each turn
    observer = Observer(client=MockChat(_observer_report_json()))
    # questioner always asks a follow-up -> session bounded by K budget
    questioner = Questioner(client=MockChat("can you also add a summary row?"))

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
    # K ~ U{1..2}; turns = 1 (seed) + up to K follow-ups
    assert res.num_turns >= 1
    assert res.ended_by in {"k_budget", "end_session"}
    # every turn returns 4 slot trajectories for bucket ingest
    assert len(res.trajectories) == res.num_turns * 4
    assert res.persona_name == PERSONAS[0].name


def test_session_ends_when_questioner_says_end():
    pool = _pool()
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])
    observer = Observer(client=MockChat(_observer_report_json()))
    questioner = Questioner(client=MockChat("<end_session>"))

    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=PERSONAS[0],
        observer=observer,
        questioner=questioner,
        k_max=3,
        seed=1,
        score_followups=False,
    )
    assert res.ended_by == "end_session"
    # seed turn succeeded, then questioner ended before a 2nd rollout
    assert res.num_turns == 1


def test_session_failure_triggers_patience_path():
    pool = _pool()
    # all slots fail (reward 0) -> winner failed -> patience decides
    agent_fn = make_agent_fn([0.0, 0.0, 0.0, 0.0])
    # observer reports empty (no artifacts) -> _winner_failed True
    observer = Observer(client=MockChat('{"intermediate": [], "final": [], "actor_claims": ""}'))
    questioner = Questioner(client=MockChat("redo please"))

    # impatient persona -> gives up quickly
    impatient = PERSONAS[3]  # Tom, patience 0.6 decay 0.3
    res = run_simulated_session(
        pool,
        seed_query="make out.csv",
        agent_fn=agent_fn,
        persona=impatient,
        observer=observer,
        questioner=questioner,
        k_max=3,
        seed=2,
        score_followups=False,
    )
    # ends by patience exhaustion (or scorer error if all-equal winner pick differs)
    assert res.ended_by in {"patience", "scorer_error"}


def test_session_scores_followups_with_injected_judge():
    pool = _pool()
    agent_fn = make_agent_fn([0.2, 0.9, 0.5, 0.1])
    observer = Observer(client=MockChat(_observer_report_json()))
    questioner = Questioner(client=MockChat("add a total row"))

    class Judge:
        def score(self, *, task, trajectory, rubric, data_source):
            return {"completion": 1.0, "safety": 1.0, "robustness": 1.0}

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
        reward_judge=Judge(),
    )
    # follow-up turns (turn>1) carry an observation-grounded reward
    followup_winners = [
        t for t in res.trajectories if t.meta.get("followup_reward") is not None
    ]
    if res.num_turns > 1:
        assert followup_winners, "expected at least one scored follow-up winner"
        assert followup_winners[0].meta["followup_reward"]["score"] == 1.0
