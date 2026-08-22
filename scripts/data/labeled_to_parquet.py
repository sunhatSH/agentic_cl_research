#!/usr/bin/env python3
"""打标 jsonl (label_capability 产物) -> verl rl_dataset parquet。

Input  : datasources/labeled/taskspecs_labeled.jsonl，每行:
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
  python scripts/datasources/labeled_to_parquet.py \
      --input datasources/labeled/taskspecs_labeled.jsonl \
      --out-dir datasets --val-fraction 0.02
  # 测试小批(测完就丢):
  python scripts/datasources/labeled_to_parquet.py --input <jsonl> --out-dir /tmp/_pq_test --limit 50
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


def _to_row(rec: dict, *, no_system: bool = False) -> dict | None:
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
    # 走沙箱 Hermes rollout 时 system 由沙箱内 Hermes 注入 → prompt 不带 system。
    # verl 内置 rollout 才需要 parquet 自带 system。
    if no_system:
        prompt = [{"role": "user", "content": user_content}]
    else:
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
    ap.add_argument("--no-system", action="store_true",
                    help="prompt 不带 system（走沙箱 Hermes rollout 时用，system 由沙箱注入）")
    ap.add_argument("--per-bucket-out", action="store_true",
                    help="每桶输出 train_<bucket>.parquet（供 train.sh --buckets 按桶顺序训）")
    ap.add_argument("--proportional", type=int, default=0,
                    help=">0: 按桶比例抽样到总量 ~N；某桶不足其占比配额时用该桶全部（不足用其本身最多）")
    ap.add_argument("--scale", type=float, default=0.0,
                    help=">0: 每桶按该比例缩放取前 N×scale 条（如 0.0625=1/16）；与 --proportional 二选一")
    ap.add_argument("--keep-full", default="",
                    help="逗号分隔的桶名，这些桶不缩放、用全部（如 communication,qa）")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("ERROR: 需要 pandas + pyarrow (pip install pandas pyarrow)")

    lines = [x for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    # 解析所有行为 row（无效跳过）
    all_rows, skipped = [], 0
    for line in lines:
        row = _to_row(json.loads(line), no_system=args.no_system)
        if row is None:
            skipped += 1
        else:
            all_rows.append(row)

    # 按桶比例抽样：目标总量 N，各桶配额 = round(N × 桶占比)，不足配额用该桶全部。
    if args.proportional > 0:
        from collections import defaultdict
        by_b: dict[str, list] = defaultdict(list)
        for r in all_rows:
            by_b[r["extra_info"]["bucket"]].append(r)
        total = len(all_rows)
        sampled = []
        print(f"[parquet] 按比例抽样 目标~{args.proportional}（总 {total}）：")
        for b, rows in sorted(by_b.items(), key=lambda x: -len(x[1])):
            quota = max(1, round(args.proportional * len(rows) / total))
            take = min(quota, len(rows))     # 不足配额 → 用该桶本身最多的（全部）
            sampled.extend(rows[:take])
            print(f"    {b:14s} 有 {len(rows):>5}  配额 {quota:>4}  取 {take:>4}")
        all_rows = sampled
        print(f"[parquet] 抽样后合计 {len(all_rows)}")

    # 按固定比例缩放：每桶取 round(len×scale) 条；keep_full 里的桶用全部。
    if args.scale > 0:
        from collections import defaultdict
        keep = {b.strip() for b in args.keep_full.split(",") if b.strip()}
        by_b: dict[str, list] = defaultdict(list)
        for r in all_rows:
            by_b[r["extra_info"]["bucket"]].append(r)
        scaled = []
        print(f"[parquet] 按 scale={args.scale} 缩放（keep-full={sorted(keep)}）：")
        for b, rows in sorted(by_b.items(), key=lambda x: -len(x[1])):
            take = len(rows) if b in keep else max(1, round(len(rows) * args.scale))
            scaled.extend(rows[:take])
            tag = "  (全量)" if b in keep else ""
            print(f"    {b:14s} 有 {len(rows):>5}  取 {take:>4}{tag}")
        all_rows = scaled
        print(f"[parquet] 缩放后合计 {len(all_rows)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _split(rows):
        tr, va = [], []
        for r in rows:
            (va if split_assignment(r["extra_info"]["record_id"], args.val_fraction) == "val"
             else tr).append(r)
        return tr, va

    if args.per_bucket_out:
        # 每桶一个 train_<bucket>.parquet（--buckets 顺序训用）+ 一份合并 val
        from collections import defaultdict
        by_b: dict[str, list] = defaultdict(list)
        for r in all_rows:
            by_b[r["extra_info"]["bucket"]].append(r)
        all_val = []
        print(f"[parquet] per-bucket 输出 → {out_dir}")
        for b, rows in sorted(by_b.items(), key=lambda x: -len(x[1])):
            tr, va = _split(rows)
            pd.DataFrame(tr).to_parquet(out_dir / f"train_{b}.parquet", index=False)
            all_val.extend(va)
            print(f"    train_{b}.parquet  {len(tr)} (+{len(va)} val)")
        pd.DataFrame(all_val).to_parquet(out_dir / "val.parquet", index=False)
        print(f"[parquet] val.parquet {len(all_val)} | 跳过(unknown/空) {skipped} | no_system={args.no_system}")
    else:
        train_rows, val_rows = _split(all_rows)
        pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
        pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)
        print(f"[parquet] 输入 {len(lines)} | 跳过 {skipped} | train {len(train_rows)} + "
              f"val {len(val_rows)} -> {out_dir} | no_system={args.no_system}")
        from collections import Counter
        c = Counter(r["extra_info"]["bucket"] for r in train_rows + val_rows)
        for b, n in c.most_common():
            print(f"    {b:14s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
