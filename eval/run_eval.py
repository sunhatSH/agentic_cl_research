"""Evaluation entry: run a checkpoint over the 195-task text subset of ClawEval.

Outputs:
    - per-task pass/fail with safety / completion / robustness sub-scores
    - aggregated New Task Perf, Old Task Forgetting, CL Score
    - dump JSON report to eval/results/<run_id>/

Pass^N: a task counts as passed only if it passes in all N independent runs
(N defaults to 3, matching the project's Pass^3 standard).

Usage:
    python -m eval.run_eval --ckpt <path> --baseline-ckpt <path>

The model rollout step is intentionally factored out (``rollout_one_task``)
so the harness can be tested without a real model loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import cl_score, new_task_performance, old_task_forgetting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--baseline-ckpt", type=str, default=None, help="Reference ckpt for forgetting computation."
    )
    parser.add_argument("--output-dir", type=str, default="eval/results")
    parser.add_argument("--num-runs", type=int, default=3, help="Pass^N standard.")
    parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="ClawEval task manifest (JSON list; see load_tasks docstring).",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include multimodal tasks (default: text-only 195 subset).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# ClawEval task manifest interface (frozen ahead of the @杨益博 data drop)      #
# --------------------------------------------------------------------------- #
# A manifest is a JSON list of task records. Required / optional fields:
#
#   task_id   (str, REQUIRED)  unique task identifier, used for Pass^N + forgetting join
#   split     (str, REQUIRED)  "General" | "Multimodal" | "Multi-turn"
#   modality  (str, REQUIRED)  "text" | "multimodal"  -- only "text" is evaluated (195 subset)
#   bucket    (str, optional)  one of the 7 capability buckets; if absent, falls back to `category`
#   category  (str, optional)  ClawEval category label (used to derive bucket downstream)
#   prompt / messages (optional) task input; consumed by rollout_one_task on the cluster
#
# A record is "text-evaluable" iff modality == "text". The 7 buckets are the
# project's capability buckets (see CLAUDE.md / doc/BucketDesign.md).
REQUIRED_MANIFEST_FIELDS = ("task_id", "split", "modality")
VALID_SPLITS = ("General", "Multimodal", "Multi-turn")
VALID_MODALITIES = ("text", "multimodal")


def validate_task_record(rec: dict, index: int = -1) -> None:
    """Raise ValueError if a manifest record is malformed."""
    where = f"manifest[{index}]" if index >= 0 else "manifest record"
    if not isinstance(rec, dict):
        raise ValueError(f"{where} must be an object, got {type(rec).__name__}")
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in rec:
            raise ValueError(f"{where} missing required field {field!r}")
    if rec["split"] not in VALID_SPLITS:
        raise ValueError(f"{where}: split {rec['split']!r} not in {VALID_SPLITS}")
    if rec["modality"] not in VALID_MODALITIES:
        raise ValueError(f"{where}: modality {rec['modality']!r} not in {VALID_MODALITIES}")


def load_tasks(tasks_file: str | None, text_only: bool = True) -> list[dict]:
    """Load + validate a ClawEval task manifest.

    Args:
        tasks_file: path to the JSON manifest. ``None`` raises (no bundled
            manifest yet -- pending the @杨益博 data drop).
        text_only: keep only ``modality == "text"`` records (the 195-task
            text subset the project currently trains/evaluates on).

    Returns the (optionally filtered) list of validated task records.
    """
    if tasks_file is None:
        raise NotImplementedError(
            "ClawEval task manifest path not configured. Pass --tasks-file or "
            "wait for ClawEval integration (see doc/ClawEval_Metadata.md)."
        )
    with open(tasks_file) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("manifest must be a JSON list of task records")
    for i, rec in enumerate(records):
        validate_task_record(rec, index=i)
    if text_only:
        records = [r for r in records if r["modality"] == "text"]
    return records


def rollout_one_task(ckpt_path: str, task: dict) -> dict:
    """Run the model on a single task and return per-task scores.

    Pending verl inference engine integration. Expected return shape:
        {
            'task_id': str,
            'safety':     float in [0, 1],
            'completion': float in [0, 1],
            'robustness': float in [0, 1],
            'reward': float,  # = safety * (0.8*completion + 0.2*robustness)
            'passed': bool,
        }
    """
    raise NotImplementedError("Per-task rollout pending verl inference integration.")


def score_task(per_run_results: list[dict]) -> dict[str, Any]:
    """Aggregate Pass^N over per-run results for one task.

    Args:
        per_run_results: list of length N, each is the rollout_one_task output.

    Returns:
        {'task_id', 'passed_all', 'safety', 'completion', 'robustness', 'reward'}
        with sub-scores averaged across runs; passed_all = all(r.passed).
    """
    if not per_run_results:
        return {}
    n = len(per_run_results)
    return {
        "task_id": per_run_results[0]["task_id"],
        "passed_all": all(r["passed"] for r in per_run_results),
        "safety": sum(r["safety"] for r in per_run_results) / n,
        "completion": sum(r["completion"] for r in per_run_results) / n,
        "robustness": sum(r["robustness"] for r in per_run_results) / n,
        "reward": sum(r["reward"] for r in per_run_results) / n,
    }


def evaluate(ckpt: str, tasks: list[dict], num_runs: int) -> list[dict]:
    """Run num_runs independent rollouts per task, return Pass^N-aggregated scores."""
    results = []
    for task in tasks:
        per_run = [rollout_one_task(ckpt, task) for _ in range(num_runs)]
        results.append(score_task(per_run))
    return results


def aggregate(current_results: list[dict], baseline_results: list[dict] | None) -> dict[str, float]:
    """Compute new_task_perf, forgetting, cl_score from per-task results."""
    new_perf = new_task_performance(current_results)
    if baseline_results:
        cur = {r["task_id"]: r["reward"] for r in current_results}
        base = {r["task_id"]: r["reward"] for r in baseline_results}
        forgetting = old_task_forgetting(cur, base)
    else:
        forgetting = 0.0
    return {
        "new_task_perf": new_perf,
        "forgetting": forgetting,
        "cl_score": cl_score(new_perf, forgetting, alpha=1.0),
    }


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks_file, text_only=not args.include_multimodal)

    current = evaluate(args.ckpt, tasks, args.num_runs)
    baseline = evaluate(args.baseline_ckpt, tasks, args.num_runs) if args.baseline_ckpt else None

    summary = aggregate(current, baseline)

    run_id = Path(args.ckpt).name
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_task.json").write_text(json.dumps(current, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
