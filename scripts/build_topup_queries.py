#!/usr/bin/env python3
"""构造「补采就绪」的 queries 文件：按桶分块排列 + finance/safety 从训练集 borrow 去重。

冷启动 pipeline 的补采循环靠 run_cold_start 的 incremental 去重（query_index/行号 + 同一
out_file）实现幂等 —— 前提是 queries 文件行序永不变，只单调提高 --num-queries。本脚本产出
这样一份文件：

  - 按 DEFAULT_BUCKETS 顺序把 cold queries **分块排列**（桶1全部 → 桶2全部 → …），
    这样「前 N 行」随 N 增大自然覆盖到后面的桶，间接给缺量桶补候选。
  - 对每桶，若 cold 候选 < floor × overshoot（预留质检淘汰），从 queries_train **借足量候选**：
    按 record_id 排除 cold 已含的、排除该桶 cold 已用的，追加到该桶块尾部（保持分块）。
  - 输出 queries_topup.jsonl + bucket_offsets.json（每桶 [start,end) 行范围，供循环按桶定位/统计）。

用法：
  python scripts/build_topup_queries.py \
      --cold datasets/queries_cold.jsonl \
      --borrow-from datasets/queries_train.jsonl \
      --floors 163,145,131,97,72,72,65,30,53 \
      --overshoot 1.4 \
      --out <OUT>/queries_topup.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.domain_tagging import DEFAULT_BUCKETS  # noqa: E402

# base.yaml:154 bucket_floors，与 DEFAULT_BUCKETS 一一对齐
DEFAULT_FLOORS = [163, 145, 131, 97, 72, 72, 65, 30, 53]


def _load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_bucket(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {b: [] for b in DEFAULT_BUCKETS}
    for r in rows:
        b = r.get("bucket")
        if b in grouped:
            grouped[b].append(r)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description="按桶分块 + borrow 去重构造补采 queries。")
    ap.add_argument("--cold", default="datasets/queries_cold.jsonl")
    ap.add_argument("--borrow-from", default="datasets/queries_train.jsonl")
    ap.add_argument("--floors", default=",".join(map(str, DEFAULT_FLOORS)),
                    help="逗号分隔，顺序同 DEFAULT_BUCKETS")
    ap.add_argument("--overshoot", type=float, default=1.4,
                    help="每桶候选目标 = floor × overshoot（预留质检淘汰）")
    ap.add_argument("--out", required=True, help="queries_topup.jsonl 输出路径")
    ap.add_argument("--no-borrow", action="store_true", help="禁用 train borrow（只用 cold）")
    args = ap.parse_args()

    floors = dict(zip(DEFAULT_BUCKETS, (int(x) for x in args.floors.split(","))))
    cold = _load(Path(args.cold))
    cold_by_b = _by_bucket(cold)
    # 全局已用 record_id（跨桶去重：cold 里出现过的一律不从 train 借）
    used_ids = {r.get("record_id") for r in cold if r.get("record_id")}

    borrow_by_b: dict[str, list[dict]] = {b: [] for b in DEFAULT_BUCKETS}
    if not args.no_borrow:
        train_by_b = _by_bucket(_load(Path(args.borrow_from)))
        for b in DEFAULT_BUCKETS:
            target = int(round(floors[b] * args.overshoot))
            have = len(cold_by_b[b])
            need = target - have
            if need <= 0:
                continue
            for r in train_by_b[b]:
                rid = r.get("record_id")
                if rid and rid in used_ids:
                    continue  # 已在 cold 或已借过
                borrow_by_b[b].append(r)
                if rid:
                    used_ids.add(rid)
                if len(borrow_by_b[b]) >= need:
                    break

    # 按桶分块写出，记录每桶 [start,end) 行范围
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[str, dict] = {}
    line_no = 0
    print(f"{'bucket':14} {'floor':>6} {'cold':>6} {'borrow':>7} {'total':>6} {'target':>7}")
    with open(out_path, "w") as w:
        for b in DEFAULT_BUCKETS:
            start = line_no
            block = cold_by_b[b] + borrow_by_b[b]
            for r in block:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                line_no += 1
            offsets[b] = {"start": start, "end": line_no,
                          "cold": len(cold_by_b[b]), "borrow": len(borrow_by_b[b])}
            target = int(round(floors[b] * args.overshoot))
            print(f"{b:14} {floors[b]:>6} {len(cold_by_b[b]):>6} "
                  f"{len(borrow_by_b[b]):>7} {len(block):>6} {target:>7}")

    off_path = out_path.with_name("bucket_offsets.json")
    with open(off_path, "w") as w:
        json.dump({"floors": floors, "overshoot": args.overshoot,
                   "total_rows": line_no, "offsets": offsets}, w, ensure_ascii=False, indent=2)

    print(f"\n[build_topup] {line_no} 行 → {out_path}")
    print(f"[build_topup] offsets → {off_path}")


if __name__ == "__main__":
    main()
