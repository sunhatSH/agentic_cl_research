"""Tests for the ClawEval manifest interface + Pass^N aggregation."""

from __future__ import annotations

import json

import pytest

from eval.run_eval import (
    aggregate,
    evaluate,
    load_tasks,
    score_task,
    validate_task_record,
)


def _write(tmp_path, records):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(records))
    return str(p)


def test_load_tasks_filters_to_text(tmp_path):
    path = _write(
        tmp_path,
        [
            {"task_id": "t1", "split": "General", "modality": "text", "bucket": "SysOps"},
            {"task_id": "t2", "split": "Multimodal", "modality": "multimodal"},
            {"task_id": "t3", "split": "Multi-turn", "modality": "text", "category": "Finance"},
        ],
    )
    tasks = load_tasks(path)
    ids = [t["task_id"] for t in tasks]
    assert ids == ["t1", "t3"]  # multimodal dropped


def test_load_tasks_keeps_multimodal_when_requested(tmp_path):
    path = _write(
        tmp_path,
        [
            {"task_id": "t1", "split": "General", "modality": "text"},
            {"task_id": "t2", "split": "Multimodal", "modality": "multimodal"},
        ],
    )
    assert len(load_tasks(path, text_only=False)) == 2


def test_load_tasks_none_raises():
    with pytest.raises(NotImplementedError):
        load_tasks(None)


def test_validate_task_record_rejects_missing_and_bad_fields():
    with pytest.raises(ValueError):
        validate_task_record({"split": "General", "modality": "text"})  # no task_id
    with pytest.raises(ValueError):
        validate_task_record({"task_id": "t", "split": "Nope", "modality": "text"})
    with pytest.raises(ValueError):
        validate_task_record({"task_id": "t", "split": "General", "modality": "video"})
    # valid -> no raise
    validate_task_record({"task_id": "t", "split": "General", "modality": "text"})


def test_load_tasks_propagates_bad_record(tmp_path):
    path = _write(
        tmp_path,
        [
            {"task_id": "ok", "split": "General", "modality": "text"},
            {"task_id": "bad", "split": "X", "modality": "text"},
        ],
    )
    with pytest.raises(ValueError):
        load_tasks(path)


def test_score_task_pass_n_aggregation():
    runs = [
        {"task_id": "t", "passed": True, "safety": 1.0, "completion": 0.8, "robustness": 0.5, "reward": 0.74},
        {
            "task_id": "t",
            "passed": False,
            "safety": 1.0,
            "completion": 0.6,
            "robustness": 0.5,
            "reward": 0.58,
        },
    ]
    out = score_task(runs)
    assert out["passed_all"] is False  # one run failed -> Pass^N fails
    assert out["completion"] == pytest.approx(0.7)


def test_evaluate_with_fake_rollout(monkeypatch):
    tasks = [{"task_id": "t1", "split": "General", "modality": "text"}]

    def fake_rollout(ckpt, task):
        return {
            "task_id": task["task_id"],
            "passed": True,
            "safety": 1.0,
            "completion": 1.0,
            "robustness": 1.0,
            "reward": 1.0,
        }

    monkeypatch.setattr("eval.run_eval.rollout_one_task", fake_rollout)
    results = evaluate("ckpt", tasks, num_runs=3)
    assert results[0]["passed_all"] is True
    summary = aggregate(results, baseline_results=None)
    assert summary["new_task_perf"] >= 0.0
    assert "cl_score" in summary
