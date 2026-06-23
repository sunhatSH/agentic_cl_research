"""User-sim simulated session: Algorithm 1 of doc/UserSim_多轮Query在线生成.md.

Drives a session where follow-up queries are generated online by the three-agent
pipeline (observer → questioner / reward) after each winner is synced, instead of
iterating a static query list. This is the §7.5 ``run_simulated_session`` that
sits beside the existing static ``SessionSandboxPool.run_session`` (which is kept
for compatibility / debugging).

Flow per turn (Algorithm 1 lines 5-17):
    parallel_rollout 8 slots ─► pick winner ─► sync_to_winner
        ─► Observer(winner) ─► ObservationReport R_t
              ├─► Reward(R_t, winner traj)            (score the winner)
              └─► Questioner(persona, R_t, history)    (next query / <end_session>)
        ─► on winner failure: PatienceTracker decides redo vs end (§3.6.5)

The session draws ONE persona at random (session-fixed) and a random follow-up
budget K ~ U{1..K_max} (§3.5). All 8 slot trajectories of every turn are returned
for per-query bucket ingest (unchanged from the static path).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agents.observer import Observer
from agents.questioner import PatienceTracker, Questioner
from agents.reward import score_followup
from agents.schema import ObservationReport, Persona
from rollout.session_pool import AgentFn, SessionSandboxPool, Trajectory


@dataclass
class SimulatedSessionResult:
    """Outcome of one simulated session."""

    trajectories: list[Trajectory] = field(default_factory=list)
    reports: list[ObservationReport] = field(default_factory=list)
    generated_queries: list[str] = field(default_factory=list)
    persona_name: str = ""
    num_turns: int = 0
    ended_by: str = "k_budget"  # k_budget | end_session | patience | scorer_error


def _winner_failed(report: ObservationReport, winner: Trajectory) -> bool:
    """A turn failed when the winner produced no usable artifacts (§3.6.5).

    Either the scorer/reward marked it failed, or the observer found nothing.
    """
    if winner.reward is not None and winner.reward <= 0.0:
        return True
    return report.is_empty()


def run_simulated_session(
    pool: SessionSandboxPool,
    seed_query: str,
    agent_fn: AgentFn,
    *,
    persona: Persona,
    observer: Observer,
    questioner: Questioner,
    k_max: int = 3,
    seed: int = 0,
    score_followups: bool = True,
    reward_judge: Any = None,
) -> SimulatedSessionResult:
    """Run one user-sim session (Algorithm 1). Returns all trajectories + telemetry.

    Args:
        pool: a SessionSandboxPool (provides spawn / run_query / winner / sync).
        seed_query: q1, the real seed from reflow data (only q1 is real).
        agent_fn: the per-slot ReAct agent (rollout/collect.make_react_agent_fn).
        persona: session-fixed persona (draw with agents.personas.sample_persona).
        observer / questioner: the two LLM agents (reward is called inline).
        k_max: follow-up budget upper bound; actual K ~ U{1..k_max} (§3.5).
        score_followups: score generated follow-ups via observation-grounded judge.
        reward_judge: inject a judge (tests); else env-resolved JudgeClient.
    """
    rng = random.Random(seed)
    k = rng.randint(1, max(1, k_max))
    result = SimulatedSessionResult(persona_name=persona.name)
    patience = PatienceTracker(persona, rng)

    pool.spawn()
    query: str | None = seed_query
    turn = 0
    prev_post: dict | None = None  # #2: prior winner's post-snapshot = next turn's baseline
    try:
        while query is not None:
            turn += 1
            # #2: reuse prior winner's post as baseline; snapshot fresh only on turn 1
            # (all 8 slots are bit-identical at turn start, synced to the prior winner).
            baseline = (
                prev_post
                if prev_post is not None
                else (observer.snapshot(pool._slots[0].client) if pool._slots else None)
            )
            trajs = pool.run_query(query, agent_fn)
            result.trajectories.extend(trajs)
            pool.query_index += 1

            winner_idx = pool.pick_winner(trajs)
            if winner_idx is None:
                result.ended_by = "scorer_error"
                break
            # capture the winner's live sandbox BEFORE sync (real backends may
            # recycle loser slots during winner-sync).
            winner_client = pool._slots[winner_idx].client if winner_idx < len(pool._slots) else None
            pool.sync_to_winner(winner_idx, trajs)
            winner = trajs[winner_idx]

            # Observe winner (diff-driven): the observer MODEL sees STATE only; the
            # winner trajectory is carried PASS-THROUGH on the report for reward (it is
            # not fed to the observer LLM). One post-snapshot, carried forward (#2).
            post = observer.snapshot(winner_client)
            report = observer.observe(
                winner_client, actor_trajectory=winner.messages, baseline=baseline, post=post
            )
            prev_post = post
            result.reports.append(report)

            # Score follow-up from the one R_t packet: completion grounded in the
            # state diff, safety/robustness from the pass-through trajectory.
            # Gated: no-effect turns skip the judge call.
            if score_followups and turn > 1:
                verdict = score_followup(query=query, report=report, judge=reward_judge)
                winner.meta["followup_reward"] = verdict

            # Failure path: patience decides redo vs end (§3.6.5). Redo turns do
            # not consume the K budget.
            if _winner_failed(report, winner):
                if patience.on_failure():
                    query = query  # ask for a redo: re-issue the same query
                    continue
                result.ended_by = "patience"
                break

            if turn > k:
                result.ended_by = "k_budget"
                break

            # Generate the next follow-up from the persona's view of the report.
            query = questioner.next_query(persona, report, pool.session_history)
            if query is None:
                result.ended_by = "end_session"
                break
            result.generated_queries.append(query)
    finally:
        pool.destroy_all()

    result.num_turns = turn
    return result
