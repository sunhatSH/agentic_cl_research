"""Tests for trainer.model_reward (dual-judge: 3-dim main + 5-dim trajectory; no GPU)."""

import math

import pytest

from trainer.model_reward import (
    _TRAJ_DEFAULTS,
    JUDGE_DIMENSIONS,
    TRAJECTORY_DIMENSIONS,
    aggregate,
    aggregate_trajectory,
    build_judge_prompt,
    compute_score,
    get_judge,
    parse_judge_output,
    resolve_group_rewards,
    set_judge,
)


class MockJudge:
    """Returns ``verdict`` for the main (system=None) call and ``traj_verdict`` for
    the trajectory (system=...) call, recording each into its own list."""

    def __init__(self, verdict, traj_verdict=None, record=None):
        self.verdict = verdict
        self.traj_verdict = traj_verdict if traj_verdict is not None else verdict
        self.record = record if record is not None else []  # main calls
        self.traj_record = []  # trajectory calls

    def score(self, *, task, trajectory, rubric, data_source, system=None):
        rec = {"task": task, "trajectory": trajectory, "rubric": rubric}
        if system is not None:
            self.traj_record.append(rec)
            return self.traj_verdict
        self.record.append(rec)
        return self.verdict


def test_judge_dimensions_are_split():
    assert JUDGE_DIMENSIONS == ("task_done", "correctness", "safety")
    assert TRAJECTORY_DIMENSIONS == ("tool", "efficiency", "planning", "consistency", "recovery")


def test_aggregate_trajectory_weights():
    # 0.20*tool + 0.20*efficiency + 0.25*planning + 0.25*consistency + 0.10*recovery
    assert math.isclose(
        aggregate_trajectory(
            {"tool": 1.0, "efficiency": 1.0, "planning": 1.0, "consistency": 1.0, "recovery": 1.0}
        ),
        1.0,
    )
    assert math.isclose(
        aggregate_trajectory(
            {"tool": 0.5, "efficiency": 0.5, "planning": 0.5, "consistency": 0.5, "recovery": 0.5}
        ),
        0.5,
    )
    # planning/consistency dominate: zeroing them costs 0.5
    assert math.isclose(
        aggregate_trajectory(
            {"tool": 1.0, "efficiency": 1.0, "planning": 0.0, "consistency": 0.0, "recovery": 1.0}
        ),
        0.5,
    )
    # missing dims default to 0.0
    assert aggregate_trajectory({}) == 0.0


def test_build_judge_prompt_includes_task_rubric_trajectory():
    msgs = build_judge_prompt(task="do X", trajectory="agent did X", rubric="must do X")
    assert msgs[0]["role"] == "system"
    # system prompt fixes the THREE-key output format
    assert "task_done" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "do X" in user and "must do X" in user and "agent did X" in user


def test_parse_judge_output_three_keys_and_task_done_binary():
    # task_done AND safety coerced to 0/1 (0.9 -> 1.0, 0.3 -> 0.0); correctness clamped
    v, parsed = parse_judge_output('{"task_done": 0.9, "correctness": 1, "safety": 0.3}')
    assert v == {"task_done": 1.0, "correctness": 1.0, "safety": 0.0}
    assert parsed is True
    # task_done below 0.5 -> 0.0
    v, _ = parse_judge_output('{"task_done": 0.4}')
    assert v["task_done"] == 0.0
    # embedded in prose; safety binarised (-1 -> 0.0), correctness clamped
    v, _ = parse_judge_output('verdict: {"correctness": 2, "safety": -1} done')
    assert v["correctness"] == 1.0 and v["safety"] == 0.0
    # safety >= 0.5 -> 1.0
    v, _ = parse_judge_output('{"safety": 0.7}')
    assert v["safety"] == 1.0
    # garbage -> defaults (task_done/correctness 0, safety 1), parsed False
    v, parsed = parse_judge_output("no json")
    assert v == {"task_done": 0.0, "correctness": 0.0, "safety": 1.0}
    assert parsed is False


def test_parse_judge_output_trajectory_dims():
    # the trajectory judge is parsed with TRAJECTORY_DIMENSIONS, no binary coercion
    v, parsed = parse_judge_output(
        '{"tool": 0.8, "efficiency": 0.6, "planning": 0.5, "consistency": 0.2, "recovery": 0.9}',
        dimensions=TRAJECTORY_DIMENSIONS,
        defaults=_TRAJ_DEFAULTS,
        binary_dims=(),
    )
    assert v == {"tool": 0.8, "efficiency": 0.6, "planning": 0.5, "consistency": 0.2, "recovery": 0.9}
    assert parsed is True
    # missing keys -> defaults
    v, _ = parse_judge_output(
        '{"tool": 0.5}',
        dimensions=TRAJECTORY_DIMENSIONS,
        defaults=_TRAJ_DEFAULTS,
        binary_dims=(),
    )
    assert v["tool"] == 0.5 and v["recovery"] == 0.0


def test_parse_judge_output_missing_safety_defaults_to_one():
    v, parsed = parse_judge_output('{"task_done": 1, "correctness": 0.9}')
    assert v == {"task_done": 1.0, "correctness": 0.9, "safety": 1.0}
    assert parsed is True


def test_aggregate_formula_done_branch():
    # done: 0.4*c + 0.4*t + 0.2, then *safety
    # all 1 -> 0.4+0.4+0.2 = 1.0
    assert math.isclose(
        aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0, "safety": 1.0}), 1.0
    )
    # done, c=0.5, t=0.5, s=1 -> 0.2+0.2+0.2 = 0.6
    assert math.isclose(
        aggregate({"task_done": 1, "correctness": 0.5, "trajectory": 0.5, "safety": 1.0}), 0.6
    )
    # +0.2 base bonus even with c=t=0 (done alone) -> 0.2
    assert math.isclose(
        aggregate({"task_done": 1, "correctness": 0.0, "trajectory": 0.0, "safety": 1.0}), 0.2
    )


def test_aggregate_formula_not_done_branch():
    # not done: 0.4*trajectory only (no correctness credit, no +0.2), then *safety
    assert math.isclose(
        aggregate({"task_done": 0, "correctness": 1.0, "trajectory": 1.0, "safety": 1.0}), 0.4
    )
    # not done + zero trajectory -> 0
    assert math.isclose(
        aggregate({"task_done": 0, "correctness": 1.0, "trajectory": 0.0, "safety": 1.0}), 0.0
    )
    # correctness is IGNORED when not done
    assert aggregate({"task_done": 0, "correctness": 1.0, "trajectory": 0.5, "safety": 1.0}) == (
        aggregate({"task_done": 0, "correctness": 0.0, "trajectory": 0.5, "safety": 1.0})
    )


def test_aggregate_safety_is_multiplicative():
    # safety = 0 zeroes the whole reward regardless of the rest
    assert aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0, "safety": 0.0}) == 0.0
    # safety is BINARY: 0.5 binarises to 1 (>=0.5) -> reward unchanged (all-1 done = 1.0)
    assert math.isclose(
        aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0, "safety": 0.5}), 1.0
    )
    # safety = 0.4 binarises to 0 -> zeroed
    assert aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0, "safety": 0.4}) == 0.0
    # missing safety defaults to 1.0 -> no penalty
    assert math.isclose(aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0}), 1.0)


def test_aggregate_nan_inf_are_finite():
    """NaN/inf judge fields must NOT propagate into reward (would NaN verl loss)."""
    from trainer.model_reward import _clamp01

    nan, inf = float("nan"), float("inf")
    assert _clamp01(nan) == 0.0
    assert _clamp01(inf) == 1.0
    assert _clamp01(-inf) == 0.0
    for field in ("task_done", "correctness", "trajectory", "safety"):
        v = aggregate({"task_done": 1, "correctness": 1.0, "trajectory": 1.0, "safety": 1.0, field: nan})
        assert math.isfinite(v), f"aggregate leaked nan via {field}"


def test_compute_score_with_injected_judge():
    judge = MockJudge(
        {"task_done": 1, "correctness": 1.0, "safety": 1.0},
        traj_verdict={"tool": 1.0, "efficiency": 1.0, "planning": 1.0, "consistency": 1.0, "recovery": 1.0},
    )
    out = compute_score(
        "agentic_cl",
        "agent solved it",
        "",
        {"queries": ["help me deploy"], "checkers": [{"type": "regex"}]},
        judge=judge,
    )
    # trajectory = 1.0, done -> 0.4*1.0 + 0.4*1.0 + 0.2 = 1.0
    assert math.isclose(out["score"], 1.0)
    assert out["judge_error"] == 0.0
    assert out["discard"] == 0.0
    # main 3 dims + trajectory + 5 sub-dims surfaced
    for d in (
        "task_done",
        "correctness",
        "safety",
        "trajectory",
        "tool",
        "efficiency",
        "planning",
        "consistency",
        "recovery",
    ):
        assert d in out
    # task text came from queries; main rubric from REWARD_RUBRIC + legacy checkers
    assert "help me deploy" in judge.record[0]["task"]
    assert judge.record[0]["rubric"]
    # two concurrent calls fired: one main + one trajectory
    assert len(judge.record) == 1 and len(judge.traj_record) == 1
    # trajectory rubric carries the TRAJECTORY_RUBRIC anchor text (not the main rubric)
    assert "工具使用质量" in judge.traj_record[0]["rubric"]


def test_compute_score_judge_error_flags_discard_not_raised():
    class BoomJudge:
        def score(self, **kw):
            raise RuntimeError("endpoint down")

    out = compute_score("agentic_cl", "x", "", {}, judge=BoomJudge())
    assert out["judge_error"] == 1.0
    assert out["discard"] == 1.0
    assert out["score"] == 0.0


def test_compute_score_folds_observer_report_into_both_rubrics():
    judge = MockJudge(
        {"task_done": 1, "correctness": 1.0, "safety": 1.0},
        traj_verdict={"tool": 0.5, "efficiency": 0.5, "planning": 0.5, "consistency": 0.5, "recovery": 0.5},
    )
    compute_score(
        "agentic_cl",
        "agent said it wrote out.csv",
        "",
        {"queries": ["make out.csv"], "observer_report": "ADDED out.csv (csv, 42B): a,b\\n1,2"},
        judge=judge,
    )
    # observer diff lands in the MAIN rubric (task_done/correctness ground truth)...
    assert "out.csv (csv, 42B)" in judge.record[0]["rubric"]
    assert "observer ground truth" in judge.record[0]["rubric"].lower()
    # ...AND the trajectory rubric (consistency checks claims against the diff)
    assert "out.csv (csv, 42B)" in judge.traj_record[0]["rubric"]


def test_compute_score_no_observer_report_is_naive():
    judge = MockJudge(
        {"task_done": 0, "correctness": 0.5, "safety": 1.0},
        traj_verdict={"tool": 0.0, "efficiency": 0.0, "planning": 0.0, "consistency": 0.0, "recovery": 0.0},
    )
    compute_score("agentic_cl", "x", "", {"queries": ["q"]}, judge=judge)
    assert "observer ground truth" not in judge.record[0]["rubric"].lower()


# --- discard + group-drop policy ---------------------------------------------


def test_resolve_group_rewards_keeps_good_rows():
    scored = [
        {"score": 0.8, "discard": 0.0},
        {"score": 0.2, "discard": 0.0},
        {"score": 0.5, "discard": 0.0},
    ]
    assert resolve_group_rewards(scored) == [0.8, 0.2, 0.5]


def test_resolve_group_rewards_discarded_row_is_none():
    scored = [
        {"score": 0.0, "discard": 1.0},  # judge failed -> None, not a fake 0
        {"score": 0.8, "discard": 0.0},
        {"score": 0.5, "discard": 0.0},
        {"score": 0.5, "discard": 0.0},
    ]
    assert resolve_group_rewards(scored) == [None, 0.8, 0.5, 0.5]


def test_resolve_group_rewards_drops_whole_group_over_half():
    # 3 of 4 discarded -> more than half -> entire group None
    scored = [
        {"score": 0.0, "discard": 1.0},
        {"score": 0.0, "discard": 1.0},
        {"score": 0.0, "discard": 1.0},
        {"score": 0.9, "discard": 0.0},
    ]
    assert resolve_group_rewards(scored) == [None, None, None, None]


def test_resolve_group_rewards_exactly_half_not_dropped():
    # 2 of 4 discarded -> NOT strictly more than half -> group survives
    scored = [
        {"score": 0.0, "discard": 1.0},
        {"score": 0.0, "discard": 1.0},
        {"score": 0.7, "discard": 0.0},
        {"score": 0.6, "discard": 0.0},
    ]
    assert resolve_group_rewards(scored) == [None, None, 0.7, 0.6]


def test_resolve_group_rewards_falls_back_to_judge_error_flag():
    # older rows may carry judge_error instead of discard
    scored = [{"score": 0.0, "judge_error": 1.0}, {"score": 0.5}]
    assert resolve_group_rewards(scored) == [None, 0.5]


def test_resolve_group_rewards_empty():
    assert resolve_group_rewards([]) == []


def test_get_judge_requires_env(monkeypatch):
    set_judge(None)
    monkeypatch.delenv("REWARD_API_BASE", raising=False)
    monkeypatch.delenv("REWARD_MODEL", raising=False)
    from agents.config import _reload_config

    _reload_config("/nonexistent/agents.yaml")
    try:
        with pytest.raises(RuntimeError, match="Reward not configured|REWARD_API_BASE"):
            get_judge()
    finally:
        set_judge(None)
        _reload_config(None)
