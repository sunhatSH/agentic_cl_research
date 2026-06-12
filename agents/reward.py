"""Observation-grounded reward -- doc §3.3 要点 4 / §6 / §7.4 / O4.

This is the third agent. It does NOT reimplement judge I/O: it reuses
``trainer.model_reward.JudgeClient`` (the OpenAI-compatible frozen judge already
wired into verl's custom_reward_function) and only adds the observation-grounded
rubric/prompt the design left blank (O4).

The judge scores the assistant on the REAL effect recorded by the observer,
not on what the actor claimed -- the observation report's 'discrepancies' field
is the anti reward-hacking signal (§3.3 要点 4).
"""

from __future__ import annotations

from typing import Any

from agents.prompts import build_reward_judge_input
from agents.schema import ObservationReport


def score_followup(
    *,
    query: str,
    report: ObservationReport,
    trajectory: str,
    judge: Any = None,
    data_source: str = "usersim_followup",
) -> dict[str, float]:
    """Score a generated follow-up turn using the observation report as evidence.

    Reuses model_reward's judge + aggregation so the reward scale matches eval
    (safety * (0.8*completion + 0.2*robustness)). The judge model is resolved
    from JUDGE_API_BASE / JUDGE_MODEL env unless injected (tests / custom).

    Returns the same dict shape as ``model_reward.compute_score``:
        {score, completion, safety, robustness, judge_error}.
    """
    from trainer.model_reward import JUDGE_DIMENSIONS, aggregate, get_judge

    parts = build_reward_judge_input(query=query, report=report, trajectory=trajectory)
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
