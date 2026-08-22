"""User-sim simulated session (SINGLE-TURN mode, 2026-07-23).

WHY SINGLE-TURN:
  verl 0.8.0's rollout contract is FIXED-SIZE: ``generate_sequences`` must
  return exactly ``gen_batch_size * n`` rows (ray_trainer.py:1397-1398), with
  ``drop_last=True`` discarding any non-multiple tail. A multi-turn UserSim
  session produces a VARIABLE number of trajectories per seed query (K turns x 8,
  K driven by Questioner satisfaction, 1..20 random). That variable yield cannot
  be cleanly packed into verl's fixed 512-row batch without pad (fake data,
  breaks GRPO groups), trim (drops ~80% of multi-turn data), or a cross-batch
  trajectory pool (changes generate_sequences semantics + pool steady-state is
  fragile). The pool approach is recorded as a future option (deprecated for
  now). The pragmatic resolution: DISABLE Questioner follow-ups so every seed
  query yields exactly one turn = 8 trajectories, giving a deterministic
  64 x 8 = 512 that fits verl's contract exactly.

WHAT SURVIVES in single-turn mode:
  - 8-slot GRPO group per seed query (unchanged).
  - Observer still runs (diff-driven, deterministic, no LLM by default): its
    report is the ground-truth evidence fed to the reward judge. The observer
    is NOT wasted -- it anchors reward in real state, not actor self-report.
  - Reward = judge over (actor trajectory + observer state_diff report).
  - Winner selection on graded reward; winner enters the replay buffer AND
    is persisted to $ROLLOUT_DATA_DIR/winners.jsonl.

WHAT IS DISABLED (code kept, just not called):
  - Questioner.next_query -- no follow-up query generation. The session ends
    after turn 1. The Questioner class and persona machinery are untouched so
    multi-turn can be re-enabled later via the (deprecated) pool approach.

Flow (single turn):
    spawn 8 slots -> run_query(seed) -> 8 trajectories
        -> Observer(winner) -> ObservationReport R_t   (state diff, ground truth)
        -> Reward(actor traj + R_t) -> graded reward per trajectory
        -> pick_winner -> sync_to_winner -> winner -> buffer
        -> _save_winner_trajectory -> $ROLLOUT_DATA_DIR/winners.jsonl
"""

from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.observer import Observer, _last_assistant_reply, flatten_trajectory
from agents.questioner import Questioner  # kept for signature compat; not called in single-turn
from agents.reward import score_followup
from agents.schema import ObservationReport, Persona
from rollout.session_pool import AgentFn, SessionSandboxPool, Trajectory

# Sessions run on a ThreadPoolExecutor (rollout/scheduler.py); serialise the
# shared winners.jsonl append so large `messages` payloads (> PIPE_BUF) don't
# interleave into corrupt lines across threads.
_WINNER_WRITE_LOCK = threading.Lock()


@dataclass
class SimulatedSessionResult:
    """Outcome of one simulated session."""

    trajectories: list[Trajectory] = field(default_factory=list)
    reports: list[ObservationReport] = field(default_factory=list)
    generated_queries: list[str] = field(default_factory=list)
    persona_name: str = ""
    num_turns: int = 0
    ended_by: str = "single_turn"  # single_turn | scorer_error


def _score_all_slots(
    trajs: list[Trajectory],
    reports: list[ObservationReport | None],
    *,
    query: str,
    reward_judge: Any,
) -> None:
    """Grade every slot's trajectory from (actor trajectory + observer report).

    Sets ``t.reward`` in place so ``pick_winner`` selects on real graded reward
    instead of falling back to random. The observer report is the ground-truth
    state diff; the actor trajectory is carried pass-through on the report for
    trajectory/safety. A slot whose observer report is empty/None (no state
    change) gets reward 0 without a judge call (gated, per agents.reward).

    Discard policy (2026-08-03): a judge failure that survived the retry sets the
    slot's ``discard`` flag; ``resolve_group_rewards`` then maps such rows to
    reward=None (masked, not a fake 0) and drops the WHOLE 8-slot group to None
    when more than half its slots were discarded. reward=None routes through
    pick_winner's scorer-error fallback (keep previous state, no sync).
    """
    from trainer.model_reward import resolve_group_rewards

    verdicts: list[dict] = []
    for t, rep in zip(trajs, reports, strict=True):
        # Stash the observer's diff evidence on the trajectory so it survives
        # back to the rollout manager, which forwards it (non_tensor
        # ``observer_report``) to the TRAINING judge -- the observer never
        # scores, it only supplies ground-truth state evidence the judge reads.
        t.meta["observer_report"] = "" if rep is None else (rep.state_diff or "")
        if rep is None or rep.is_empty():
            # 空 report：不武断判 0。有些任务不改系统状态（QA/纯对话/只读），observer
            # 无 FS/sys diff 给不出 report，但 agent 可能回答得好——若 actor 有最终回复，
            # 构造兜底 report（把回复 fold 进 final）交给 judge 用 trajectory 打分；
            # 只有 agent 真没产出（空轨迹 / 崩溃 / timeout）才判 0。
            last_reply = _last_assistant_reply(t.messages)
            if last_reply:
                rep = ObservationReport(
                    actor_trajectory=flatten_trajectory(t.messages),
                    final=[{"path": "(assistant reply)", "kind": "text", "content_excerpt": last_reply}],
                )
                verdict = score_followup(query=query, report=rep, judge=reward_judge)
            else:
                # agent 真没产出 → 真 0（gated，非 discard）。
                verdict = {"score": 0.0, "gated": 1.0, "discard": 0.0}
        else:
            verdict = score_followup(query=query, report=rep, judge=reward_judge)
        t.meta["reward_verdict"] = verdict
        verdicts.append(verdict)

    # Apply discard + >half-group-drop policy across the whole group at once.
    rewards = resolve_group_rewards(verdicts)
    for t, r in zip(trajs, rewards, strict=True):
        # None (discard / group-dropped) -> pick_winner treats it as a scorer
        # error: keep previous state, no sync, failure stays visible in the
        # verdict's discard/judge_error fields.
        t.reward = None if r is None else float(r)


def _save_winner_trajectory(traj: Trajectory, query: str) -> None:
    """Persist winner trajectory to AFS JSONL for permanent record.

    Written to $ROLLOUT_DATA_DIR/winners.jsonl (one JSON object per line).
    Serialised across threads via _WINNER_WRITE_LOCK so concurrent sessions'
    appends don't interleave into corrupt lines.
    """
    out_dir = os.environ.get("ROLLOUT_DATA_DIR", "")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "trajectory_id": traj.trajectory_id,
        "reward": traj.reward,
        "bucket": traj.bucket,
        "messages": traj.messages,
        "reward_verdict": traj.meta.get("reward_verdict", {}),
    }
    path = os.path.join(out_dir, "winners.jsonl")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _WINNER_WRITE_LOCK, open(path, "a") as f:
        f.write(line)


def run_simulated_session(
    pool: SessionSandboxPool,
    seed_query: str,
    agent_fn: AgentFn,
    *,
    persona: Persona,
    observer: Observer,
    questioner: Questioner,  # noqa: ARG001 -- kept for call-site compat; unused in single-turn
    k_max: int = 3,  # noqa: ARG001 -- unused in single-turn; kept for call-site compat
    seed: int = 0,
    score_followups: bool = True,  # noqa: ARG001 -- always scores the single turn
    reward_judge: Any = None,
    success_threshold: float = 0.8,  # noqa: ARG001 -- unused in single-turn
) -> SimulatedSessionResult:
    """Run one SINGLE-TURN user-sim session. Returns 8 trajectories + telemetry.

    Single-turn (2026-07-23): the Questioner is NOT called, so the session runs
    exactly one turn and yields a deterministic 8 trajectories. This makes the
    per-seed yield fixed (8), so ``gen_batch_size`` seeds produce a fixed
    ``gen_batch_size * 8`` rows that fit verl's fixed-size rollout contract.
    See module docstring for the full rationale.

    Args:
        pool: a SessionSandboxPool (provides spawn / run_query / winner / sync).
        seed_query: q1, the real seed from reflow data.
        agent_fn: the per-slot ReAct agent (rollout/collect.make_react_agent_fn).
        persona: session-fixed persona (drawn but unused in single-turn; kept
            for telemetry / future multi-turn re-enable).
        observer: diff-driven observer; its report feeds the reward judge.
        questioner: UNUSED in single-turn (no follow-up). Kept in the signature
            so call sites and tests don't break; multi-turn can be re-enabled.
        reward_judge: inject a judge (tests); else env-resolved JudgeClient.
    """
    rng = random.Random(seed)
    result = SimulatedSessionResult(persona_name=persona.name)
    _ = rng  # no random turn budget in single-turn; kept for future multi-turn

    pool.spawn()
    try:
        # Single turn: snapshot baseline (all 8 slots bit-identical at start),
        # run the 8 slots on the seed query, observe + score, pick winner, sync.
        baseline = observer.snapshot(pool._slots[0].client) if pool._slots else None
        trajs = pool.run_query(seed_query, agent_fn)
        result.trajectories.extend(trajs)
        pool.query_index += 1

        # Observe each slot's resulting state (diff-driven, deterministic). The
        # observer MODEL sees STATE only; each trajectory is carried pass-through
        # on its report for the reward judge (safety/robustness).
        slot_reports: list[ObservationReport | None] = []
        for slot_idx, t in enumerate(trajs):
            client = pool._slots[slot_idx].client if slot_idx < len(pool._slots) else None
            post = observer.snapshot(client)
            rep = observer.observe(
                client, actor_trajectory=t.messages, baseline=baseline, post=post
            )
            slot_reports.append(rep)
            result.reports.append(rep)

        # Grade all 8 slots from (actor trajectory + observer report) BEFORE
        # winner selection, so pick_winner selects on real graded reward.
        _score_all_slots(trajs, slot_reports, query=seed_query, reward_judge=reward_judge)

        winner_idx = pool.pick_winner(trajs)
        if winner_idx is None:
            result.ended_by = "scorer_error"
        else:
            pool.sync_to_winner(winner_idx, trajs)
            _save_winner_trajectory(trajs[winner_idx], seed_query)
            result.ended_by = "single_turn"
    finally:
        pool.destroy_all()

    result.num_turns = 1
    return result
