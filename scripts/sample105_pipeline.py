#!/usr/bin/env python3
"""sample105_v2 采集数据 → 7 桶 Replay Buffer 数据管道 CLI。

三步（可单独跑，也可串联）：

  1. extract   — 从 OpenClaw 采集目录抽每个主会话的首 query
                 → data/first_queries.jsonl
  2. classify  — LLM 给首 query 分桶 + 子桶
                 → data/first_queries_classified.jsonl
  3. route     — 按桶把完整轨迹重建入桶
                 → data/buckets/<bucket>/<record_id>.jsonl

典型串联::

    # 全流程（extract → classify → route）
    python scripts/sample105_pipeline.py all \\
        --root /path/to/sample105_v2 --out-dir data

    # 仅抽首 query（不联网）
    python scripts/sample105_pipeline.py extract \\
        --root /path/to/sample105_v2 --output data/first_queries.jsonl

    # 仅分桶（需 tokenhub key / .env；输入是上一步的 jsonl）
    python scripts/sample105_pipeline.py classify \\
        --input data/first_queries.jsonl --output data/first_queries_classified.jsonl

    # 仅入桶
    python scripts/sample105_pipeline.py route \\
        --input data/first_queries_classified.jsonl --out-dir data/buckets

凭证：classify 走 tokenhub（``.env`` 的 TOKENHUB_API_KEY），端点缺省用
``configs/agents.yaml`` 的 observer 配置；可用 ``BUCKET_CLASSIFIER_API_BASE /
_BUCKET`` env 覆盖。无网络时 classify 会把每条标 unknown（不崩，可后续重跑）。

产物落 ``data/``（gitignored），不入库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_pipeline.classify import classify_queries, make_default_client  # noqa: E402
from data_pipeline.dag import build_session_dag, dag_summary  # noqa: E402
from data_pipeline.extract import extract_first_queries  # noqa: E402
from data_pipeline.route import route_subagents, route_trajectories  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_extract(args) -> int:
    records = extract_first_queries(args.root, limit=args.limit)
    _write_jsonl(Path(args.output), records)
    total_q = sum(1 for r in records if r["first_query"].strip())
    print(f"[extract] sessions={len(records)} with_first_query={total_q} -> {args.output}")
    return 0


def cmd_classify(args) -> int:
    # 先加载 .env 让 TOKENHUB_API_KEY 进环境，再构造 client（resolve 读 env）
    _maybe_load_env()
    client = make_default_client(args.config)
    records = _load_jsonl(Path(args.input))
    classified = classify_queries(records, client, max_tokens=args.max_tokens)
    _write_jsonl(Path(args.output), classified)
    from collections import Counter

    buckets = Counter(r["bucket"] for r in classified)
    ok = sum(1 for r in classified if r.get("ok"))
    print(f"[classify] total={len(classified)} ok={ok} -> {args.output}")
    print("  桶分布:")
    for b, c in buckets.most_common():
        print(f"    {b:14s} {c}")
    return 0


def cmd_route(args) -> int:
    classified = _load_jsonl(Path(args.input))
    stats = route_trajectories(classified, args.out_dir)
    print(f"[route] total={stats['total']} unknown={stats['unknown']} skipped={stats['skipped']}")
    print("  入桶:")
    for b, c in sorted(stats["per_bucket"].items()):
        print(f"    {b:14s} {c}")
    print(f"  -> {args.out_dir}")
    return 0


def cmd_route_subagents(args) -> int:
    """子会话事件流原样转 OpenAI chat、独立成训练样本（与主轨迹各自单独训练）。

    规定：格式=原数据，只按文件顺序转 chat，不增减字段、不改内容、不重排；
    主轨迹何时拿子数据看 Agent + 原数据，我们不编排。DAG 仅用于找存在的子日志。
    """
    nodes = build_session_dag(args.root)
    summary = dag_summary(nodes)
    print(f"[dag] {summary}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = route_subagents(nodes, out_dir)
    print(f"[route-subagents] total={stats['total']} written={stats['written']} skipped={stats['skipped']}")
    print(f"  -> {out_dir}")
    return 0


def _maybe_load_env() -> None:
    """加载 .env 让 TOKENHUB_API_KEY 进环境（与 scripts/load_training_env.sh 对齐）。"""
    import os

    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def cmd_all(args) -> int:
    # extract
    records = extract_first_queries(args.root, limit=args.limit)
    fq_path = Path(args.out_dir) / "first_queries.jsonl"
    _write_jsonl(fq_path, records)
    print(f"[extract] sessions={len(records)} -> {fq_path}")

    # classify
    _maybe_load_env()
    client = make_default_client(args.config)
    classified = classify_queries(records, client, max_tokens=args.max_tokens)
    cls_path = Path(args.out_dir) / "first_queries_classified.jsonl"
    _write_jsonl(cls_path, classified)
    from collections import Counter

    buckets = Counter(r["bucket"] for r in classified)
    ok = sum(1 for r in classified if r.get("ok"))
    print(f"[classify] total={len(classified)} ok={ok} -> {cls_path}")
    for b, c in buckets.most_common():
        print(f"    {b:14s} {c}")

    # route
    bucket_root = Path(args.out_dir) / "buckets"
    stats = route_trajectories(classified, bucket_root)
    print(f"[route] total={stats['total']} unknown={stats['unknown']} skipped={stats['skipped']}")
    for b, c in sorted(stats["per_bucket"].items()):
        print(f"    {b:14s} {c}")
    print(f"  -> {bucket_root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="抽首 query")
    p_extract.add_argument("--root", required=True, help="OpenClaw 采集根目录（含 000XXX/）")
    p_extract.add_argument("--output", default="data/first_queries.jsonl")
    p_extract.add_argument("--limit", type=int, default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_classify = sub.add_parser("classify", help="LLM 分桶")
    p_classify.add_argument("--input", default="data/first_queries.jsonl")
    p_classify.add_argument("--output", default="data/first_queries_classified.jsonl")
    p_classify.add_argument("--config", default=None, help="agents.yaml 路径")
    p_classify.add_argument("--max-tokens", type=int, default=256)
    p_classify.set_defaults(func=cmd_classify)

    p_route = sub.add_parser("route", help="按桶入完整轨迹")
    p_route.add_argument("--input", default="data/first_queries_classified.jsonl")
    p_route.add_argument("--out-dir", default="data/buckets")
    p_route.set_defaults(func=cmd_route)

    p_sub = sub.add_parser("route-subagents", help="子会话事件流原样转 chat、独立成样本")
    p_sub.add_argument("--root", required=True, help="OpenClaw 采集根目录（含 000XXX/）")
    p_sub.add_argument("--out-dir", default="data/subagent_trajectories")
    p_sub.set_defaults(func=cmd_route_subagents)

    p_all = sub.add_parser("all", help="extract → classify → route 串联")
    p_all.add_argument("--root", required=True, help="OpenClaw 采集根目录")
    p_all.add_argument("--out-dir", default="data")
    p_all.add_argument("--config", default=None)
    p_all.add_argument("--limit", type=int, default=None)
    p_all.add_argument("--max-tokens", type=int, default=256)
    p_all.set_defaults(func=cmd_all)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
