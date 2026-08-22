"""Multi-turn user-sim rollout collection WITHOUT GRPO / winner / reward.

This is the cold-rollout variant the project asked for (2026-06-13):

  - 1 query == 1 rollout (NO 8-slot group, NO winner-sync, NO GRPO).
  - Each session: a single sandbox runs a seed query, then the loop is driven
    by the two agents -- observer (objective report on the single trajectory)
    and questioner (persona-driven next query) -- for up to K turns.
  - NO reward / judge this stage (observer + questioner only).
  - N sessions run in parallel (each = its own sandbox + its own seed query),
    so "N sandboxes run N queries" as required.

Contrast with rollout/simulated_session.run_simulated_session, which is built
for 8-slot GRPO (run_query -> pick_winner -> sync_to_winner). Here there is a
single trajectory per turn, so there is no winner to pick or sync.

The observer / questioner LLM backends are REQUIRED (remote models): if they
are not configured the caller must abort -- this stage cannot proceed without
the observer (no report) or questioner (no follow-up). See scripts/collect_rollout.sh.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agents.observer import Observer
from agents.questioner import Questioner
from agents.schema import ObservationReport, Persona
from rollout.session_pool import AgentFn, SessionSandboxPool, Trajectory


@dataclass
class UserSimSessionResult:
    """All trajectories + telemetry from one multi-turn session."""

    trajectories: list[Trajectory] = field(default_factory=list)
    reports: list[ObservationReport] = field(default_factory=list)
    generated_queries: list[str] = field(default_factory=list)
    persona_name: str = ""
    num_turns: int = 0
    ended_by: str = "k_budget"  # k_budget | end_session | agent_error


def run_usersim_session(
    pool: SessionSandboxPool,
    seed_query: str,
    agent_fn: AgentFn,
    *,
    persona: Persona,
    observer: Observer,
    questioner: Questioner,
    k_max: int = 3,
    seed: int = 0,
) -> UserSimSessionResult:
    """One multi-turn session: single-rollout per turn, observer + questioner.

    Args:
        pool: SessionSandboxPool with slots=1 (single trajectory per turn).
        seed_query: q1, the real reflow seed (only q1 is real; rest generated).
        agent_fn: per-slot ReAct agent (rollout.collect.make_react_agent_fn).
        persona: session-fixed persona (agents.personas.sample_persona).
        observer / questioner: REQUIRED remote LLM agents.
        k_max: follow-up budget upper bound; actual K ~ U{1..k_max}.
    """
    rng = random.Random(seed)
    k = rng.randint(1, max(1, k_max))
    result = UserSimSessionResult(persona_name=persona.name)

    pool.spawn()
    query: str | None = seed_query
    turn = 0
    prev_post: dict | None = None  # #2: previous turn's post-snapshot = this turn's baseline
    try:
        while query is not None:
            turn += 1
            sandbox = pool._slots[0].client if pool._slots else None
            # #2: reuse last turn's post as baseline; only snapshot fresh on turn 1.
            baseline = prev_post if prev_post is not None else observer.snapshot(sandbox)
            # slots=1 -> exactly one trajectory.
            trajs = pool.run_query(query, agent_fn)
            pool.query_index += 1
            result.trajectories.extend(trajs)
            if not trajs:
                result.ended_by = "agent_error"
                break
            traj = trajs[0]

            # The single trajectory IS the session history (no winner to pick).
            pool.session_history.extend(traj.messages)
            # advance the (single) slot state already handled by run_query.

            # Observe (diff-driven): observer MODEL sees STATE only; trajectory carried
            # pass-through on the report. One post-snapshot, carried forward (#2).
            post = observer.snapshot(sandbox)
            report = observer.observe(sandbox, actor_trajectory=traj.messages, baseline=baseline, post=post)
            prev_post = post
            result.reports.append(report)

            if turn > k:
                result.ended_by = "k_budget"
                break

            # Persona-driven next query from the report + history.
            query = questioner.next_query(persona, report, pool.session_history)
            if query is None:
                result.ended_by = "end_session"
                break
            result.generated_queries.append(query)
    finally:
        pool.destroy_all()

    result.num_turns = turn
    return result
