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

# Zero verdict for a no-effect turn: nothing produced -> completion 0; nothing
# harmful happened -> safety 1; no artifacts to be robust -> robustness 0.
_NO_EFFECT_VERDICT = {
    "score": 0.0,
    "completion": 0.0,
    "safety": 1.0,
    "robustness": 0.0,
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

    Everything comes from the one ``report`` packet: ``state_diff`` (completion ground
    truth) and ``actor_trajectory`` (the actor's actions, carried pass-through by the
    observer for safety/robustness). Reuses model_reward's judge + aggregation so the
    reward scale matches eval (safety * (0.8*completion + 0.2*robustness)). The judge
    model is resolved from JUDGE_API_BASE / JUDGE_MODEL env unless injected.

    Returns the same dict shape as ``model_reward.compute_score``
    ({score, completion, safety, robustness, judge_error}); a gated/no-effect turn
    additionally carries ``gated=1.0``.
    """
    # Gate: no effect this turn -> don't call the judge at all.
    if not report.has_effect:
        return dict(_NO_EFFECT_VERDICT)

    from trainer.model_reward import JUDGE_DIMENSIONS, aggregate, get_judge

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
        verdict = {d: 0.0 for d in JUDGE_DIMENSIONS}
        judge_error = 1.0

    def _c01(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    return {
        "score": float(aggregate(verdict)),
        "completion": _c01(verdict.get("completion", 0.0)),
        "safety": _c01(verdict.get("safety", 0.0)),
        "robustness": _c01(verdict.get("robustness", 0.0)),
        "judge_error": judge_error,
    }
