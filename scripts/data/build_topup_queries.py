#!/usr/bin/env python3
"""冷启动 queries 构造：两种模式。

冷启动 pipeline 的补采靠 run_cold_start 的 incremental 去重（query_index/行号 + 同一
out_file）实现幂等 —— 前提是 queries 文件行序永不变，新增行只能【追加到尾部】。

正确的 borrow 时机是「先采 cold、质检、确认真缺，才 borrow」，所以本脚本不再采集前预拼
borrow。两种模式：

  模式① cold-only（默认，无 --gaps）：
    只把 cold queries 按 DEFAULT_BUCKETS 顺序【分块排列】（桶1全部 → 桶2全部 → …）输出。
    不 borrow。这样「前 N 行」随 N 增大自然覆盖后面的桶。
    输出：<out> + <out>同目录/bucket_offsets.json（每桶 [start,end) 行范围）。

  模式② borrow 补缺（--gaps GAPS_JSON --append-to EXISTING）：
    采集+质检后，用每桶缺口 {bucket: 还缺N} 只为【缺口桶】从 train borrow：
      借 N×overshoot 条候选（留质检淘汰余量），record_id 去重排除 cold 已用 + 已 borrow
      （已 borrow 记 <out>同目录/borrowed_ids.json，跨轮不重复借）。
    borrow 行【追加到 EXISTING 文件尾部】（不改前面行 index，incremental 幂等成立）。
    输出：追加写入 EXISTING；更新 borrowed_ids.json；打印本轮实际 borrow 数。

用法：
  # ① cold-only
  python scripts/data/build_topup_queries.py --cold datasets/queries_cold.jsonl \
      --out <OUT>/queries_topup.jsonl
  # ② 采集后按缺口 borrow，追加到已有 queries
  python scripts/data/build_topup_queries.py --cold datasets/queries_cold.jsonl \
      --borrow-from datasets/queries_train.jsonl --gaps <OUT>/gaps_roundN.json \
      --append-to <OUT>/queries_topup.jsonl --overshoot 1.4
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


def _build_cold_only(args) -> None:
    """模式①：cold 按桶分块，不 borrow。"""
    cold = _load(Path(args.cold))
    cold_by_b = _by_bucket(cold)
    floors = dict(zip(DEFAULT_BUCKETS, (int(x) for x in args.floors.split(","))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[str, dict] = {}
    line_no = 0
    print(f"{'bucket':14} {'floor':>6} {'cold':>6}")
    with open(out_path, "w") as w:
        for b in DEFAULT_BUCKETS:
            start = line_no
            for r in cold_by_b[b]:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                line_no += 1
            offsets[b] = {"start": start, "end": line_no, "cold": len(cold_by_b[b]), "borrow": 0}
            print(f"{b:14} {floors[b]:>6} {len(cold_by_b[b]):>6}")

    off_path = out_path.with_name("bucket_offsets.json")
    with open(off_path, "w") as w:
        json.dump({"floors": floors, "total_rows": line_no, "offsets": offsets},
                  w, ensure_ascii=False, indent=2)
    # 初始化空的 borrowed_ids（供后续 --gaps 轮累积）
    bid_path = out_path.with_name("borrowed_ids.json")
    if not bid_path.exists():
        json.dump([], open(bid_path, "w"))
    print(f"\n[build_topup] cold-only {line_no} 行 → {out_path}")
    print(f"[build_topup] offsets → {off_path}")


def _append_borrow(args) -> None:
    """模式②：按缺口只为缺口桶 borrow，追加到已有 queries 尾部（跨轮去重）。"""
    gaps = json.load(open(args.gaps))          # {bucket: 还缺N}
    if not gaps:
        print("[build_topup] 无缺口，跳过 borrow"); return

    existing_path = Path(args.append_to)
    cold_ids = {r.get("record_id") for r in _load(Path(args.cold)) if r.get("record_id")}
    bid_path = existing_path.with_name("borrowed_ids.json")
    borrowed_ids = set(json.load(open(bid_path))) if bid_path.exists() else set()
    used = cold_ids | borrowed_ids            # 已 cold 或已借过的，一律不再借

    train_by_b = _by_bucket(_load(Path(args.borrow_from)))
    new_rows: list[dict] = []
    print(f"{'bucket':14} {'gap':>5} {'want':>6} {'got':>5}")
    for b, gap in gaps.items():
        want = int(round(gap * args.overshoot))   # 借 gap×overshoot，留质检淘汰余量
        got = 0
        for r in train_by_b.get(b, []):
            rid = r.get("record_id")
            if rid and rid in used:
                continue
            new_rows.append(r)
            if rid:
                used.add(rid); borrowed_ids.add(rid)
            got += 1
            if got >= want:
                break
        print(f"{b:14} {gap:>5} {want:>6} {got:>5}")

    if not new_rows:
        print("[build_topup] ⚠ train 无可借候选（缺口桶已借尽）"); return

    # 追加到已有 queries 尾部（保持前面行 index 不变 → incremental 幂等）
    with open(existing_path, "a") as w:
        for r in new_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(sorted(borrowed_ids), open(bid_path, "w"))
    print(f"\n[build_topup] borrow {len(new_rows)} 行 → 追加至 {existing_path}")
    print(f"[build_topup] 累计已借 {len(borrowed_ids)} → {bid_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="冷启动 queries：cold-only 或按缺口 borrow 补缺。")
    ap.add_argument("--cold", default="datasets/queries_cold.jsonl")
    ap.add_argument("--borrow-from", default="datasets/queries_train.jsonl")
    ap.add_argument("--floors", default=",".join(map(str, DEFAULT_FLOORS)),
                    help="逗号分隔，顺序同 DEFAULT_BUCKETS")
    ap.add_argument("--overshoot", type=float, default=1.4,
                    help="borrow 时每桶借 gap×overshoot（预留质检淘汰）")
    ap.add_argument("--out", help="模式①：cold-only queries 输出路径")
    ap.add_argument("--gaps", help="模式②：缺口 JSON（{bucket: 还缺N}）")
    ap.add_argument("--append-to", help="模式②：把 borrow 行追加到这个已有 queries 文件")
    args = ap.parse_args()

    if args.gaps:
        if not args.append_to:
            ap.error("--gaps 需配 --append-to")
        _append_borrow(args)
    else:
        if not args.out:
            ap.error("模式① 需 --out")
        _build_cold_only(args)


if __name__ == "__main__":
    main()
