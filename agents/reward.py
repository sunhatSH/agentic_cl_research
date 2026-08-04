"""Observation-grounded reward -- doc §3.3 要点 4 / §6 / §7.4 / O4.

This is the third agent. It does NOT reimplement judge I/O: it reuses
``trainer.model_reward.JudgeClient`` (the OpenAI-compatible frozen judge already
wired into verl's custom_reward_function) and only adds the observation-grounded
rubric/prompt the design left blank (O4).

The judge sees TWO channels, both carried by the one R_t packet:
  - the observer's STATE evidence (deterministic ``state_diff``) -> ground truth for
    completion; anchors reward in the real effect;
  - ``report.actor_trajectory`` -> the actor's actions, carried PASS-THROUGH by the
    observer component (the observer MODEL never sees it -- no token waste) -> used
    to judge safety / robustness (how the agent acted).

Gating (the "几层拦截"): if the turn produced NO effect (empty diff,
``report.has_effect`` False), we SHORT-CIRCUIT and return a zero score WITHOUT
calling the judge -- a turn that changed nothing gets completion 0 for free, and
the expensive judge round-trip is skipped.
"""

from __future__ import annotations

from typing import Any

from agents.prompts import build_reward_judge_input
from agents.schema import ObservationReport

# Zero verdict for a no-effect turn: nothing produced -> not done + not correct;
# nothing harmful happened -> safety 1; no process to grade -> trajectory 0.
_NO_EFFECT_VERDICT = {
    "score": 0.0,
    "task_done": 0.0,
    "correctness": 0.0,
    "trajectory": 0.0,
    "safety": 1.0,
    "judge_error": 0.0,
    "gated": 1.0,
}


def score_followup(
    *,
    query: str,
    report: ObservationReport,
    judge: Any = None,
    data_source: str = "usersim_followup",
) -> dict[str, float]:
    """Score a follow-up turn on the observer's state diff + the pass-through trajectory.

    Everything comes from the one ``report`` packet: ``state_diff`` (task_done /
    correctness ground truth) and ``actor_trajectory`` (the actor's actions, carried
    pass-through by the observer for trajectory/safety). Reuses model_reward's judge +
    aggregation so the reward scale matches training (four-dimension formula:
    if task_done: 0.4*correctness + 0.4*trajectory + 0.2 else 0.4*trajectory; *safety).
    The judge model is resolved from REWARD_API_BASE / REWARD_MODEL env unless injected.

    Returns the same dict shape as ``model_reward.compute_score``
    ({score, task_done, correctness, trajectory, safety, judge_error, discard}); a
    gated/no-effect turn additionally carries ``gated=1.0``. On a judge failure that
    survived the retry, ``discard=1.0`` and score 0 (caller should drop the row).
    """
    # Gate: no usable evidence this turn -> don't call the judge at all.
    # Use is_empty() (deliverable-aware) NOT the raw has_effect flag: a text
    # deliverable (QA/reasoning) folds the reply into report.final with
    # has_effect possibly False (no FS/sys change). Gating on has_effect alone
    # forced reward 0 for every text-only task -> zero GRPO advantage on entire
    # buckets (silent collapse). is_empty() returns False whenever final/
    # intermediate carry a real deliverable, so those tasks now reach the judge.
    if report.is_empty():
        return dict(_NO_EFFECT_VERDICT)

    from trainer.model_reward import _DIM_DEFAULTS, aggregate, get_judge

    parts = build_reward_judge_input(query=query, report=report)
    client = judge if judge is not None else get_judge()

    try:
        verdict = client.score(
            task=parts["task"],
            trajectory=parts["trajectory"],
            rubric=parts["rubric"],
            data_source=data_source,
        )
        judge_error = 0.0
    except Exception:  # noqa: BLE001 -- never crash the rollout on judge I/O
        verdict = dict(_DIM_DEFAULTS)
        judge_error = 1.0

    def _c01(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        # NaN slips past `<0`/`>1` (both False) -> would propagate into reward.
        # Treat NaN as 0 (no signal); inf handled by >1 -> 1.0.
        import math

        if math.isnan(v):
            return 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    return {
        "score": 0.0 if judge_error else float(aggregate(verdict)),
        "task_done": _c01(verdict.get("task_done", _DIM_DEFAULTS["task_done"])),
        "correctness": _c01(verdict.get("correctness", _DIM_DEFAULTS["correctness"])),
        "trajectory": _c01(verdict.get("trajectory", _DIM_DEFAULTS["trajectory"])),
        "safety": _c01(verdict.get("safety", _DIM_DEFAULTS["safety"])),
        "judge_error": judge_error,
        "discard": judge_error,  # 1.0 -> caller sets reward=None (masked, not scored 0)
    }
