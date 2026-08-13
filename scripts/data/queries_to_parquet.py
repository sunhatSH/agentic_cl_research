#!/usr/bin/env python3
"""queries_train.jsonl -> verl rl_dataset parquet (single-turn baseline).

2026-07-23: 全链路单轮化后，baseline 训练数据直接从已打标的
`datasets/queries_train.jsonl`（{record_id, queries, bucket, persona_name}）
转成 verl rl_dataset parquet。单轮 = 每个 query 一条轨迹，所以训练集大小
= query 数。按桶配额采样：小桶全取、大桶截断，保证 9 桶都有。

verl row schema (data.return_raw_chat=true):
    prompt        list[{role, content}]  -- system + first user query
    data_source   str                    -- "agentic_cl"
    reward_model  {ground_truth: str}
    extra_info    {record_id, bucket, queries, persona_name, num_user_turns}

排除被冷采集借走的 record_id（读 <rollouts>/borrowed_ids.json），与冷采集
保持一致。第一轮 baseline 无借用 → borrowed_ids 为空 → 不排除，逻辑一致。

用法:
    python scripts/data/queries_to_parquet.py \\
        --input datasets/queries_train.jsonl \\
        --out-dir datasets \\
        --per-bucket-cap 3600          # 每桶上限（大桶截断）
        # --exclude borrowed_ids.json   # 可选：排除被借走的 id
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

DATA_SOURCE = "agentic_cl"
DEFAULT_BUCKETS = [
    "workflow", "ops", "qa", "finance", "office",
    "communication", "safety", "coding", "research",
]


def _load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_bucket(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {b: [] for b in DEFAULT_BUCKETS}
    other: list[dict] = []
    for r in rows:
        b = r.get("bucket")
        if b in grouped:
            grouped[b].append(r)
        else:
            other.append(r)
    if other:
        grouped["unknown"] = other  # type: ignore[assignment]
    return grouped


def split_assignment(record_id: str, val_fraction: float = 0.02) -> str:
    """Stable train/val split by hashing record_id (reproducible)."""
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return "val" if (int(h[:8], 16) % 10000) / 10000.0 < val_fraction else "train"


def query_to_row(rec: Mapping) -> dict | None:
    """Build one verl rl_dataset row from a queries_train record."""
    queries = rec.get("queries")
    if not isinstance(queries, list) or not queries:
        return None
    seed = str(queries[0]).strip()
    if not seed:
        return None
    record_id = str(rec.get("record_id", ""))
    bucket = str(rec.get("bucket", "unknown"))
    persona = str(rec.get("persona_name", ""))
    prompt = [
        {"role": "system", "content": "You are a helpful agentic assistant."},
        {"role": "user", "content": seed},
    ]
    return {
        "prompt": prompt,
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": {
            "record_id": record_id,
            "bucket": bucket,
            "queries": json.dumps([seed], ensure_ascii=False),
            "persona_name": persona,
            "num_user_turns": 1,
            "checkers": "[]",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="queries_train.jsonl -> verl parquet (single-turn).")
    ap.add_argument("--input", default="datasets/queries_train.jsonl")
    ap.add_argument("--out-dir", default="datasets")
    ap.add_argument("--per-bucket-cap", type=int, default=0,
                    help="每桶上限：0=不截断（全量训练，测不均分布防遗忘）；"
                         ">0 时小桶全取、大桶截断到此值。默认 0=全量。")
    ap.add_argument("--exclude", default=None,
                    help="可选：borrowed_ids.json 路径，排除被冷采集借走的 record_id。")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    args = ap.parse_args()

    rows = _load(Path(args.input))
    excluded: set[str] = set()
    if args.exclude:
        ep = Path(args.exclude)
        if ep.exists():
            excluded = set(json.load(open(ep)))
            print(f"[queries_to_parquet] 排除 {len(excluded)} 个被借走的 record_id")

    # 按 record_id 去重（queries_train 可能有重复）
    seen: set[str] = set()
    grouped = _by_bucket(rows)
    cap = args.per_bucket_cap

    rows_split: dict[str, list[dict]] = {"train": [], "val": []}
    stats: Counter = Counter()
    print(f"{'bucket':14} {'have':>6} {'cap':>5} {'used':>5}")
    for b in DEFAULT_BUCKETS + (["unknown"] if "unknown" in grouped else []):
        have = len(grouped.get(b, []))
        used = 0
        for r in grouped.get(b, []):
            rid = r.get("record_id")
            if rid and (rid in excluded or rid in seen):
                continue
            if cap > 0 and used >= cap:   # cap=0 = 不截断（全量）
                break
            row = query_to_row(r)
            if row is None:
                continue
            if rid:
                seen.add(rid)
            used += 1
            split = split_assignment(str(rid), args.val_fraction)
            rows_split[split].append(row)
            stats[f"{b}/{split}"] += 1
        print(f"{b:14} {have:>6} {cap:>5} {used:>5}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in rows_split.items():
        table = pa.Table.from_pylist([
            {
                "prompt": r["prompt"],
                "data_source": r["data_source"],
                "reward_model": r["reward_model"],
                "bucket": r["extra_info"]["bucket"],
                "extra_info": r["extra_info"],
            }
            for r in data
        ])
        out = out_dir / f"{split}.parquet"
        pq.write_table(table, out)
        print(f"[queries_to_parquet] {split}: {len(data)} rows -> {out}")

    total = len(rows_split["train"]) + len(rows_split["val"])
    print(f"[queries_to_parquet] total={total} (train={len(rows_split['train'])} val={len(rows_split['val'])})")


if __name__ == "__main__":
    main()
