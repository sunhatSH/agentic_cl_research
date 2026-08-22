#!/usr/bin/env python3
"""taskspec.yaml -> queries.jsonl (data pipeline Stage A).

把上游 taskspec（datasources/taskspecs/s_<id>/taskspec.yaml）转成 queries JSONL，
每行含 record_id + queries + bucket（--classify 时 LLM 打桶）。

Input  : datasources/taskspecs/s_<id>/taskspec.yaml
Output : queries.jsonl，每行一个 JSON 对象：
             {"record_id": "<task_id>", "queries": ["<seed>", "<follow_up>", ...],
              "bucket": "ops", "sub_bucket": "operations"}

Usage:
    # 无打桶（快速，桶全标 unknown）
    python scripts/data/taskspec_to_queries.py --output datasets/queries.jsonl

    # LLM 打桶（需 TOKENHUB_API_KEY 或 BUCKET_CLASSIFIER_* env）
    python scripts/data/taskspec_to_queries.py --classify --output datasets/queries.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 9 桶（与 replay_buffer / configs/base.yaml 对齐，仅用于统计打印）
CANONICAL_BUCKETS = [
    "workflow", "ops", "qa", "finance", "office",
    "communication", "safety", "coding", "research",
]

# task_family → bucket 静态映射。这些 task_family 语义确定、不会跨桶，
# 直接用代码打标，不走 LLM。其余 task_family 必须 LLM 分类。
_TASK_FAMILY_TO_BUCKET: dict[str, str] = {
    "file_organize": "ops",
    "file_move_rename": "ops",
    "data_merge": "ops",
    "table_process": "office",
    "risky_op": "safety",
    "process_design": "workflow",
}


def _load_taskspec(path: Path) -> dict[str, Any] | None:
    """Load one taskspec.yaml; return None on parse error."""
    try:
        ts = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        print(f"[WARN] parse failed: {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(ts, dict):
        print(f"[WARN] not a mapping: {path}", file=sys.stderr)
        return None
    return ts


def _extract_queries(ts: dict[str, Any]) -> list[str]:
    """seed_query (顶层) + user_profile.follow_ups (list[str]) -> queries 列表。

    seed_query 是真实首条 query (q1)，follow_ups 是原始会话的后续 query。
    两者拼成 queries 列表；collect_cold 只取 queries[0]。
    """
    queries: list[str] = []
    seed = ts.get("seed_query")
    if isinstance(seed, str) and seed.strip():
        queries.append(seed.strip())
    # follow_ups 在 user_profile 下（见用户贴的 taskspec.yaml 结构）
    profile = ts.get("user_profile") or {}
    follow_ups = profile.get("follow_ups") if isinstance(profile, dict) else None
    if isinstance(follow_ups, list):
        for q in follow_ups:
            if isinstance(q, str) and q.strip():
                queries.append(q.strip())
    return queries


def _record_id(ts: dict[str, Any], fallback_dir: str) -> str:
    """task_id 优先，缺失时用目录名。"""
    tid = ts.get("task_id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return fallback_dir


def _task_family(ts: dict[str, Any]) -> str:
    """task_family 用于统计（不进 queries，仅打印分布）。"""
    tf = ts.get("task_family")
    return tf if isinstance(tf, str) else "unknown"


def convert(taskspecs_dir: Path, output: Path, limit: int | None, *, classify: bool = False) -> dict[str, Any]:
    """遍历 taskspecs 目录，每个 task 输出一行 queries JSONL。

    当 ``classify=True`` 时，对每条 seed_query 调用 LLM 分类模型打桶
    （``data_pipeline/classify.py::classify_query``），结果写入 ``bucket`` / ``sub_bucket`` 字段。
    """
    import json

    client = None
    if classify:
        # Ensure TOKENHUB_API_KEY is set (reads AGENT_MODEL_KEY from runtime.env)
        import os as _os
        if not _os.environ.get("TOKENHUB_API_KEY", "").strip():
            env_file = Path(__file__).resolve().parent.parent / "docker" / "sandbox" / "runtime.env"
            if env_file.is_file():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip() == "AGENT_MODEL_KEY" and v.strip().strip('"').strip("'"):
                        _os.environ["TOKENHUB_API_KEY"] = v.strip().strip('"').strip("'")
                        break

        from data_pipeline.classify import classify_query, make_default_client

        client = make_default_client()
        print(f"[classify] LLM bucket classifier: model={client.model}", flush=True)

    if not taskspecs_dir.is_dir():
        raise FileNotFoundError(f"taskspecs dir not found: {taskspecs_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "total_dirs": 0,
        "parsed": 0,
        "skipped_no_seed": 0,
        "written": 0,
        "classified_ok": 0,
        "classified_unknown": 0,
        "classified_static": 0,
        "classify_errors": 0,
        "task_families": {},
        "queries_per_task": [],
        "buckets": {},
    }

    # 按 task_id 排序，保证输出稳定可复现
    subdirs = sorted(d for d in taskspecs_dir.iterdir() if d.is_dir())
    with output.open("w", encoding="utf-8") as fh:
        for d in subdirs:
            stats["total_dirs"] += 1
            ts_path = d / "taskspec.yaml"
            if not ts_path.is_file():
                continue
            ts = _load_taskspec(ts_path)
            if ts is None:
                continue
            stats["parsed"] += 1

            queries = _extract_queries(ts)
            if not queries:
                stats["skipped_no_seed"] += 1
                print(f"[WARN] no seed_query: {d.name}", file=sys.stderr)
                continue

            record_id = _record_id(ts, fallback_dir=d.name)
            tf = _task_family(ts)
            stats["task_families"][tf] = stats["task_families"].get(tf, 0) + 1
            stats["queries_per_task"].append(len(queries))

            row: dict[str, Any] = {"record_id": record_id, "queries": queries}

            # Bucket classification: static mapping > LLM
            static_bucket = _TASK_FAMILY_TO_BUCKET.get(tf, "")
            if static_bucket:
                row["bucket"] = static_bucket
                row["sub_bucket"] = None
                row["bucket_rationale"] = f"static: task_family={tf}"
                stats["classified_static"] += 1
            elif client is not None:
                verdict = classify_query(queries[0], client)
                row["bucket"] = verdict.get("bucket", "unknown")
                row["sub_bucket"] = None  # taskspecs 无官方 category，子桶留空
                row["bucket_rationale"] = verdict.get("rationale", "")
                if verdict.get("ok"):
                    stats["classified_ok"] += 1
                else:
                    stats["classified_unknown"] += 1
            stats["buckets"][row.get("bucket", "unknown")] = stats["buckets"].get(row.get("bucket", "unknown"), 0) + 1

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1

            if limit is not None and stats["written"] >= limit:
                break

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taskspecs", default="datasources/taskspecs", help="taskspecs 根目录")
    ap.add_argument("--output", default="datasets/queries.jsonl", help="输出 queries JSONL")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 task")
    ap.add_argument("--classify", action="store_true", help="LLM 打桶（需 TOKENHUB_API_KEY 或 BUCKET_CLASSIFIER_* env）")
    args = ap.parse_args()

    stats = convert(Path(args.taskspecs), Path(args.output), args.limit, classify=args.classify)
    n = stats["queries_per_task"]
    avg_q = (sum(n) / len(n)) if n else 0.0
    print("=== taskspec_to_queries stats ===")
    print(f"taskspec dirs : {stats['total_dirs']}")
    print(f"parsed        : {stats['parsed']}")
    print(f"skipped (no seed_query): {stats['skipped_no_seed']}")
    print(f"written       : {stats['written']}")
    if args.classify:
        print(f"classified (static): {stats['classified_static']}")
        print(f"classified (LLM ok): {stats['classified_ok']}")
        print(f"classified (LLM unk): {stats['classified_unknown']}")
        print("bucket distribution:")
        for b, c in sorted(stats["buckets"].items(), key=lambda x: -x[1]):
            print(f"  {b:15s} {c}")
    print(f"avg queries/task      : {avg_q:.1f}")
    print("task_family distribution:")
    for tf, c in sorted(stats["task_families"].items(), key=lambda x: -x[1]):
        print(f"  {tf:30s} {c}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
