"""Tests for trainer.model_reward (model judge; judge injected/mocked, no GPU)."""

import math

import pytest

from trainer.model_reward import (
    aggregate,
    build_judge_prompt,
    compute_score,
    get_judge,
    parse_judge_output,
    set_judge,
)


class MockJudge:
    def __init__(self, verdict, record=None):
        self.verdict = verdict
        self.record = record if record is not None else []

    def score(self, *, task, trajectory, rubric, data_source):
        self.record.append({"task": task, "trajectory": trajectory, "rubric": rubric})
        return self.verdict


def test_build_judge_prompt_includes_task_rubric_trajectory():
    msgs = build_judge_prompt(task="do X", trajectory="agent did X", rubric="must do X")
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "do X" in user and "must do X" in user and "agent did X" in user


def test_parse_judge_output_plain_and_embedded():
    v, parsed = parse_judge_output('{"completion": 1, "safety": 1, "robustness": 0.5}')
    assert v == {
        "completion": 1.0,
        "safety": 1.0,
        "robustness": 0.5,
    }
    assert parsed is True
    # embedded in prose + clamping out-of-range
    v, _ = parse_judge_output('verdict: {"completion": 2, "safety": -1, "robustness": 0.3} done')
    assert v["completion"] == 1.0 and v["safety"] == 0.0 and v["robustness"] == 0.3
    # garbage -> all zero, parsed False
    v, parsed = parse_judge_output("no json")
    assert v == {"completion": 0.0, "safety": 0.0, "robustness": 0.0}
    assert parsed is False


def test_aggregate_formula():
    # safety gate is multiplicative
    assert aggregate({"completion": 1.0, "safety": 0.0, "robustness": 1.0}) == 0.0
    assert math.isclose(aggregate({"completion": 1.0, "safety": 1.0, "robustness": 0.0}), 0.8)
    assert math.isclose(
        aggregate({"completion": 0.5, "safety": 1.0, "robustness": 0.5}), 0.8 * 0.5 + 0.2 * 0.5
    )


def test_compute_score_with_injected_judge():
    judge = MockJudge({"completion": 1.0, "safety": 1.0, "robustness": 1.0})
    out = compute_score(
        "agentic_cl",
        "agent solved it",
        "",
        {"queries": ["help me deploy"], "checkers": [{"type": "regex"}]},
        judge=judge,
    )
    assert math.isclose(out["score"], 1.0)
    assert out["judge_error"] == 0.0
    # task text came from queries; rubric from checkers
    assert "help me deploy" in judge.record[0]["task"]
    assert judge.record[0]["rubric"]


def test_compute_score_judge_error_is_flagged_not_raised():
    class BoomJudge:
        def score(self, **kw):
            raise RuntimeError("endpoint down")

    out = compute_score("agentic_cl", "x", "", {}, judge=BoomJudge())
    assert out["judge_error"] == 1.0
    assert out["score"] == 0.0


def test_get_judge_requires_env(monkeypatch):
    set_judge(None)
    monkeypatch.delenv("REWARD_API_BASE", raising=False)
    monkeypatch.delenv("REWARD_MODEL", raising=False)
    # Also point config at a missing file so yaml resolution doesn't fill in
    from agents.config import _reload_config

    _reload_config("/nonexistent/agents.yaml")
    try:
        with pytest.raises(RuntimeError, match="Reward not configured|REWARD_API_BASE"):
            get_judge()
    finally:
        set_judge(None)  # cleanup
        _reload_config(None)  # restore default config
