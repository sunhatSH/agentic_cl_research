#!/usr/bin/env python3
"""taskspec.yaml -> queries.jsonl (Step 3 of the data pipeline).

把吴健给的 taskspec（data/taskspecs/s_<id>/taskspec.yaml）转成 collect_cold.py
能消费的 queries JSONL：每个 task 一行，含 record_id + queries 列表。

Input  : data/taskspecs/s_<id>/taskspec.yaml
         每个 task 一份，字段见 doc/训练与推理流程.md / Paper_Method_draft_CN.md。
Output : queries.jsonl，每行一个 JSON 对象：
             {"record_id": "<task_id>", "queries": ["<seed_query>", "<follow_up_1>", ...]}
         collect_cold.py 的 iter_seeds() 只取 queries[0]（= seed_query）作冷启动种子；
         后续 follow_ups 留在列表里供正式训练态的 questioner 在线生成参考（或对照）。

为什么需要这一步：
  - collect_cold.py / collect_rollout.py 的 iter_seeds() 读的是 JSONL
    （{"record_id","queries":[...]}），不直接读 taskspec.yaml。
  - taskspec.yaml 的 seed_query 在顶层、follow_ups 在 user_profile 下，需要摘出来
    拍平成 queries 列表。
  - 真实 follow_ups 在论文方法里被丢弃（前提漂移，§3.2），但这里仍写进 queries
    列表——是否使用由下游决定（collect_cold 只取 q1；正式训练的 questioner
    在线生成，不复用这些）。保留是为了 pipeline 完整性 + 对照实验可选。

Usage:
    python scripts/taskspec_to_queries.py \\
        --taskspecs data/taskspecs \\
        --output datasets/queries.jsonl
    # 只跑前 5 个 task 调试：
    python scripts/taskspec_to_queries.py --limit 5 --output /tmp/queries.jsonl

Notes:
  - 缺 seed_query 的 task 跳过（打 WARNING）。
  - follow_ups 缺失或非 list 时 queries 只含 seed_query。
  - record_id 取 task_id（taskspec 顶层字段）；task_id 缺失时 fallback 目录名。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# 9 桶（与 replay_buffer / configs/base.yaml 对齐，仅用于统计打印）
CANONICAL_BUCKETS = [
    "workflow", "ops", "qa", "finance", "office",
    "communication", "safety", "coding", "research",
]


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


def convert(taskspecs_dir: Path, output: Path, limit: int | None) -> dict[str, Any]:
    """遍历 taskspecs 目录，每个 task 输出一行 queries JSONL。"""
    import json

    if not taskspecs_dir.is_dir():
        raise FileNotFoundError(f"taskspecs dir not found: {taskspecs_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_dirs": 0,
        "parsed": 0,
        "skipped_no_seed": 0,
        "written": 0,
        "task_families": {},
        "queries_per_task": [],
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

            fh.write(json.dumps({"record_id": record_id, "queries": queries}, ensure_ascii=False) + "\n")
            stats["written"] += 1

            if limit is not None and stats["written"] >= limit:
                break

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--taskspecs",
        default="data/taskspecs",
        help="taskspecs 根目录（每个子目录 s_<id>/ 含 taskspec.yaml）",
    )
    ap.add_argument(
        "--output",
        default="datasets/queries.jsonl",
        help="输出 queries JSONL 路径（collect_cold.py --queries 指向它）",
    )
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 task（调试用）")
    args = ap.parse_args()

    stats = convert(Path(args.taskspecs), Path(args.output), args.limit)
    n = stats["queries_per_task"]
    avg_q = (sum(n) / len(n)) if n else 0.0
    print("=== taskspec_to_queries stats ===")
    print(f"taskspec dirs : {stats['total_dirs']}")
    print(f"parsed        : {stats['parsed']}")
    print(f"skipped (no seed_query): {stats['skipped_no_seed']}")
    print(f"written       : {stats['written']}")
    print(f"avg queries/task      : {avg_q:.1f}  (seed + follow_ups)")
    print("task_family distribution:")
    for tf, c in sorted(stats["task_families"].items(), key=lambda x: -x[1]):
        print(f"  {tf:30s} {c}")
    print(f"-> {args.output}")
    print(f"  下一步: python scripts/collect_cold.py --queries {args.output} ...")


if __name__ == "__main__":
    main()
