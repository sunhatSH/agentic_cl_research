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
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Self-contained experiment dir (e.g. runs/phase3/r4); eval goes to "
        "<exp-dir>/eval/. Overrides --output-dir. This is the runs/ layout.",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Pass^N standard.")
    parser.add_argument(
        "--tasks-file",
        type=str,
        default="eval/claweval_manifest.json",
        help="ClawEval task manifest (JSON list; 默认 eval/claweval_manifest.json, "
             "由 scripts/build_eval_manifest.py 生成, 195 纯文本任务按 9 能力桶分组).",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include multimodal tasks (default: text-only 195 subset).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# ClawEval task manifest interface (frozen ahead of the eval data drop)         #
# --------------------------------------------------------------------------- #
# A manifest is a JSON list of task records. Required / optional fields:
#
#   task_id   (str, REQUIRED)  unique task identifier, used for Pass^N + forgetting join
#   split     (str, REQUIRED)  "General" | "Multimodal" | "Multi-turn"
#   modality  (str, REQUIRED)  "text" | "multimodal"  -- only "text" is evaluated (195 subset)
#   bucket    (str, optional)  one of the 9 capability buckets; if absent, falls back to `category`
#   category  (str, optional)  ClawEval category label (used to derive bucket downstream)
#   prompt / messages (optional) task input; consumed by rollout_one_task on the cluster
#
# A record is "text-evaluable" iff modality == "text". The 9 buckets are the
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
            manifest yet -- pending the eval data drop).
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
            'reward': float,  # = completion * (0.8*safety + 0.2*robustness)
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


def _bucket_of(task_id: str, task_bucket: dict[str, str]) -> str:
    """Bucket for a task id; 'Unlabeled' if the manifest carried no bucket."""
    return task_bucket.get(task_id, "Unlabeled")


def aggregate_by_bucket(
    current: list[dict],
    baseline: list[dict] | None,
    task_bucket: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Per-bucket score table (总账). One row per bucket.

    Each row carries the bucket's mean reward / pass_rate / sub-scores for the
    current model, AND — when a baseline is given — the baseline's same numbers
    plus the delta, so every 过程评测 shows BOTH ends (CL vs baseline) without
    waiting for the final showdown.
    """
    base_reward = {r["task_id"]: r["reward"] for r in (baseline or [])}
    base_pass = {r["task_id"]: r["passed_all"] for r in (baseline or [])}

    buckets: dict[str, list[dict]] = {}
    for r in current:
        buckets.setdefault(_bucket_of(r["task_id"], task_bucket), []).append(r)

    table: dict[str, dict[str, Any]] = {}
    for bucket, rows in sorted(buckets.items()):
        n = len(rows)
        row = {
            "n_tasks": n,
            "reward": sum(r["reward"] for r in rows) / n,
            "pass_rate": sum(1 for r in rows if r["passed_all"]) / n,
            "safety": sum(r["safety"] for r in rows) / n,
            "completion": sum(r["completion"] for r in rows) / n,
            "robustness": sum(r["robustness"] for r in rows) / n,
        }
        if baseline is not None:
            b_rows = [r for r in rows if r["task_id"] in base_reward]
            if b_rows:
                bn = len(b_rows)
                b_reward = sum(base_reward[r["task_id"]] for r in b_rows) / bn
                b_pass = sum(1 for r in b_rows if base_pass[r["task_id"]]) / bn
                row["baseline_reward"] = b_reward
                row["baseline_pass_rate"] = b_pass
                # delta > 0 = CL 比 baseline 好（该桶防遗忘生效）
                row["reward_delta"] = row["reward"] - b_reward
                row["pass_rate_delta"] = row["pass_rate"] - b_pass
        table[bucket] = row
    return table


def _dump_bucket_reports(
    out_dir: Path,
    current: list[dict],
    task_bucket: dict[str, str],
) -> None:
    """细则: one JSONL per bucket, one line per task with full sub-scores."""
    per_bucket_dir = out_dir / "per_bucket"
    per_bucket_dir.mkdir(parents=True, exist_ok=True)
    by_bucket: dict[str, list[dict]] = {}
    for r in current:
        by_bucket.setdefault(_bucket_of(r["task_id"], task_bucket), []).append(r)
    for bucket, rows in by_bucket.items():
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        (per_bucket_dir / f"{bucket}.jsonl").write_text(lines + "\n", encoding="utf-8")


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks_file, text_only=not args.include_multimodal)
    # task_id -> bucket (prefer explicit 'bucket', fall back to 'category').
    task_bucket = {
        t["task_id"]: t.get("bucket") or t.get("category") or "Unlabeled" for t in tasks
    }

    current = evaluate(args.ckpt, tasks, args.num_runs)
    baseline = evaluate(args.baseline_ckpt, tasks, args.num_runs) if args.baseline_ckpt else None

    summary = aggregate(current, baseline)
    bucket_table = aggregate_by_bucket(current, baseline, task_bucket)

    # Output layout: --exp-dir <runs/phaseN/exp> puts eval under <exp-dir>/eval/
    # (self-contained per-experiment dir); else legacy eval/results/<run_id>/.
    if args.exp_dir:
        out_dir = Path(args.exp_dir) / "eval"
    else:
        run_id = Path(args.ckpt).name
        out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 总账: per-bucket score table (+ CL vs baseline 两端) + global summary.
    scores = {
        "checkpoint": args.ckpt,
        "baseline_checkpoint": args.baseline_ckpt,
        "num_runs": args.num_runs,
        "summary": summary,          # new_task_perf / forgetting / cl_score
        "per_bucket": bucket_table,  # 每桶得分 + baseline 对比 + delta
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    # 细则: 桶内逐题
    _dump_bucket_reports(out_dir, current, task_bucket)
    # 保留平铺 per_task 供旧工具/复查
    (out_dir / "per_task.json").write_text(json.dumps(current, indent=2, ensure_ascii=False))
    print(json.dumps({"summary": summary, "buckets": list(bucket_table)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
