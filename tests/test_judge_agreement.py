"""Tests for eval.judge_agreement (judge↔human agreement + judge ranking)."""

import math

from eval.judge_agreement import (
    binary_classification,
    cohen_kappa,
    evaluate_judge,
    mae,
    pearson,
    rank_judges,
)


def test_mae_and_pearson():
    assert mae([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert math.isclose(mae([1.0, 0.0], [0.0, 1.0]), 1.0)
    assert math.isclose(pearson([1, 2, 3], [1, 2, 3]), 1.0)
    assert math.isclose(pearson([1, 2, 3], [3, 2, 1]), -1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0  # zero variance


def test_cohen_kappa():
    assert math.isclose(cohen_kappa([True, False, True], [True, False, True]), 1.0)
    # chance-level: kappa near 0
    k = cohen_kappa([True, False, True, False], [False, True, False, True])
    assert k < 0.0 or math.isclose(k, -1.0)


def test_binary_classification():
    m = binary_classification([True, True, False, False], [True, False, True, False])
    assert math.isclose(m["accuracy"], 0.5)
    assert math.isclose(m["precision"], 0.5)
    assert math.isclose(m["recall"], 0.5)


def _sample(bucket, c, s, r):
    return {
        "task": "t",
        "trajectory": "x",
        "rubric": "",
        "bucket": bucket,
        "human": {"completion": c, "safety": s, "robustness": r},
    }


def test_evaluate_judge_perfect_and_bucketed():
    samples = [
        _sample("SysOps", 1.0, 1.0, 1.0),
        _sample("SysOps", 0.0, 1.0, 0.0),
        _sample("Communication", 0.5, 1.0, 0.5),
    ]
    preds = [dict(s["human"]) for s in samples]  # perfect judge
    rep = evaluate_judge(samples, preds, pass_threshold=0.5)
    assert rep["n"] == 3
    assert rep["score_mae"] == 0.0
    assert math.isclose(rep["pass"]["accuracy"], 1.0)
    assert math.isclose(rep["pass"]["kappa"], 1.0)
    assert set(rep["per_bucket"]) == {"SysOps", "Communication"}
    assert rep["per_bucket"]["SysOps"]["n"] == 2


def test_evaluate_judge_imperfect():
    samples = [_sample("SysOps", 1.0, 1.0, 1.0), _sample("SysOps", 0.0, 1.0, 0.0)]
    # judge inverts: predicts low where human high and vice versa
    preds = [
        {"completion": 0.0, "safety": 1.0, "robustness": 0.0},
        {"completion": 1.0, "safety": 1.0, "robustness": 1.0},
    ]
    rep = evaluate_judge(samples, preds, pass_threshold=0.5)
    assert rep["score_mae"] > 0.0
    assert rep["pass"]["accuracy"] < 1.0


def test_rank_judges_picks_smallest_clearing():
    reports = {
        "small14b": {"pass": {"kappa": 0.62}},
        "mid32b": {"pass": {"kappa": 0.80}},
        "big72b": {"pass": {"kappa": 0.85}},
    }
    sizes = {"small14b": 14, "mid32b": 32, "big72b": 72}
    out = rank_judges(reports, sizes_b=sizes, min_kappa=0.6)
    assert out["recommended"] == "small14b"  # smallest clearing 0.6
    assert out["cleared_threshold"] is True
    assert out["ranking"][0]["judge"] == "big72b"  # ranked by kappa desc


def test_rank_judges_none_clears():
    reports = {"a": {"pass": {"kappa": 0.3}}, "b": {"pass": {"kappa": 0.4}}}
    out = rank_judges(reports, sizes_b={"a": 7, "b": 14}, min_kappa=0.6)
    assert out["cleared_threshold"] is False
    assert out["recommended"] == "b"  # best available
