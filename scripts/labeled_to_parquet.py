#!/usr/bin/env python3
"""打标 jsonl (label_capability 产物) -> verl rl_dataset parquet。

Input  : data/labeled/taskspecs_labeled.jsonl，每行:
           {record_id, bucket, queries:[seed, *follow_ups], hidden_goal,
            persona, available_tools, missing_info_slots, safety_constraints, difficulty}
Output : train.parquet + val.parquet，verl rl_dataset 列:
           prompt        list[{role,content}]  -- system + 首个 user query(rollout 起点)
           data_source   str                   -- "agentic_cl"
           reward_model   {ground_truth}        -- 空(judge 在线打分)
           bucket        str                   -- 9 桶能力标签（top-level，供 trajectory_adapter 读取）
           extra_info    {record_id, bucket, queries, persona, available_tools,
                          missing_info_slots, safety_constraints, difficulty}

设计(与 doc/训练与推理流程.md 路径 A 一致):
  - parquet 只装 prompt(对话起点 = queries 合并，句号分割),verl 拿 prompt -> lightllm
    rollout -> 产 trajectory -> 入 buffer。不装 rollout 轨迹(此时还没有)。
  - bucket(能力桶)进 extra_info,供 buffer 分桶 + 训练 + 评测按桶对齐。
  - follow_ups / persona / tools / safety 进 extra_info,供三 Agent UserSim 消费。
  - bucket=="unknown" 的行跳过(buffer 会 skip,不入训练)。

用法:
  python scripts/labeled_to_parquet.py \
      --input data/labeled/taskspecs_labeled.jsonl \
      --out-dir datasets --val-fraction 0.02
  # 测试小批(测完就丢):
  python scripts/labeled_to_parquet.py --input <jsonl> --out-dir /tmp/_pq_test --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.convert_dataset import split_assignment  # noqa: E402

DATA_SOURCE = "agentic_cl"

_SYSTEM_PROMPT = (
    "You are a capable autonomous agent. Complete the user's task using the available tools. "
    "Work independently — never ask the user for input, confirmation, or clarification. "
    "When faced with ambiguity or multiple options, pick the most reasonable or first option "
    "and proceed without hesitation."
)


def _to_row(rec: dict) -> dict | None:
    rid = rec.get("record_id", "")
    bucket = rec.get("bucket", "")
    queries = rec.get("queries") or []
    if bucket in ("", "unknown") or not queries:
        return None
    # 多个 query(seed + follow_ups + correction)合并成一条 user 内容，用句号分割。
    parts = []
    for q in queries:
        s = str(q).strip()
        if not s:
            continue
        # 已以中/英句末标点结尾则不再补句号，避免 "。。"
        if s[-1] not in "。.!?！？":
            s += "。"
        parts.append(s)
    user_content = "".join(parts).strip()
    if not user_content:
        return None
    prompt = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return {
        "prompt": prompt,
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": ""},
        "bucket": bucket,
        "extra_info": {
            "record_id": rid,
            "bucket": bucket,
            "queries": queries,
            "persona": rec.get("persona", ""),
            "available_tools": rec.get("available_tools", []),
            "missing_info_slots": rec.get("missing_info_slots", []),
            "safety_constraints": rec.get("safety_constraints", []),
            "difficulty": rec.get("difficulty", ""),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help=">0: 只取前 N 条(测试用)")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("ERROR: 需要 pandas + pyarrow (pip install pandas pyarrow)")

    lines = [x for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    train_rows, val_rows = [], []
    skipped = 0
    for line in lines:
        rec = json.loads(line)
        row = _to_row(rec)
        if row is None:
            skipped += 1
            continue
        if split_assignment(row["extra_info"]["record_id"], args.val_fraction) == "val":
            val_rows.append(row)
        else:
            train_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)

    print(f"[parquet] 输入 {len(lines)} 行 | 跳过(unknown/空) {skipped} | "
          f"train {len(train_rows)} + val {len(val_rows)} -> {out_dir}")
    # 桶分布
    from collections import Counter
    c = Counter(r["extra_info"]["bucket"] for r in train_rows + val_rows)
    for b, n in c.most_common():
        print(f"    {b:14s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
