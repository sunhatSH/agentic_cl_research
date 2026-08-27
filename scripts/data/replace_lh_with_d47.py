#!/usr/bin/env python3
"""把数据集里 LH 行中【非 d4-7】的替换成全量打分池里未用的【d4-7】任务。

背景:带原始轨迹重打难度后(lh_difficulty.jsonl,521 条),发现当前进 train_cl.parquet
的 241 个 LH 里有 76 个不是 d4-7(d8:68/d9:5/d2:1/d3:1/+1 未打分)。用户要求:进数据集
的 LH 全部是 d4-7。

做法(保 12800 行 / batch32 / coding=6400 不变):
  1. 保留当前 LH 里已是 d4-7 的 165 个。
  2. 剔除 76 个非 d4-7。
  3. 从池里【未用的 d4-7】(170 个)按难度取 76 个补上(优先 d5-6 中等,再 d4,再 d7,
     使补进来的和保留的合起来 d4-7 分布尽量均衡)。
  4. LH 行的 difficulty 用【轨迹打分】(lh_difficulty.jsonl),不再用 completion_score。
  5. 顺带:把保留的 165 行 difficulty 也刷成轨迹打分值(之前回填漏了 1 条 LH_000170)。

复用 replace_swe_with_longhorizon 的 build_lh_tasks(query/GT/answer_key 构建逻辑)。

用法:
  python3 scripts/data/replace_lh_with_d47.py            # 预览
  python3 scripts/data/replace_lh_with_d47.py --apply     # 落地
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from replace_swe_with_longhorizon import TASKSPECS, build_lh_tasks  # noqa: E402

PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"
DIFF_JSONL = ROOT / "datasources" / "labeled" / "lh_difficulty.jsonl"


def load_traj_difficulty() -> dict[str, int]:
    """rid → 轨迹打分难度(lh_difficulty.jsonl)。"""
    out = {}
    with open(DIFF_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["rid"]] = int(d["difficulty"])
    return out


def build_lh_row(task: dict, difficulty: int) -> dict:
    """按 replace_swe_with_longhorizon 的格式建一行 LH parquet row + 写 answer_key.json。"""
    rid = task["rid"]
    q = task["query"]
    ak_path = TASKSPECS / rid / "answer_key.json"
    ak_path.parent.mkdir(parents=True, exist_ok=True)
    ak_path.write_text(json.dumps(task["answer_key"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prompt": [{"role": "user", "content": q}],
        "data_source": "agentic_cl",
        "reward_model": {
            "ground_truth": "",
            "style": "rule",
            "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
        },
        "bucket": "coding",
        "extra_info": {
            "record_id": rid,
            "bucket": "coding",
            "queries": [q],
            "persona": "",
            "available_tools": [],
            "missing_info_slots": [],
            "safety_constraints": [],
            "difficulty": str(difficulty),
            "gen_task_id": rid,
            "ws_dir": "",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    traj_diff = load_traj_difficulty()
    d47_pool = {r for r, d in traj_diff.items() if 4 <= d <= 7}

    # 当前 parquet
    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()
    lh_rows = [r for r in rows if r["extra_info"].get("gen_task_id", "").startswith("LH_")]
    lh_rids = {r["extra_info"]["gen_task_id"] for r in lh_rows}
    n_lh = len(lh_rows)
    print(f"当前: {len(rows)} 行, LH {n_lh} 个", flush=True)

    keep_rids = lh_rids & d47_pool          # 已是 d4-7,保留
    drop_rids = lh_rids - d47_pool          # 非 d4-7,剔除
    need = len(drop_rids)
    print(f"LH 中 d4-7 保留 {len(keep_rids)}, 剔除非 d4-7 {need}", flush=True)

    # 池里未用的 d4-7,按"优先中等 d5-6 → d4 → d7"排序取 need 个
    unused = d47_pool - lh_rids
    order = {5: 0, 6: 0, 4: 1, 7: 2}
    unused_sorted = sorted(unused, key=lambda r: (order.get(traj_diff[r], 9), r))
    add_rids = set(unused_sorted[:need])
    if len(add_rids) < need:
        print(f"⚠️ 池里未用 d4-7 仅 {len(unused)}, 不足 {need}", flush=True)
        return 1
    print(f"从池补入 {len(add_rids)} 个 d4-7 (难度: {dict(sorted(Counter(traj_diff[r] for r in add_rids).items()))})", flush=True)

    # 最终 LH rid 集合 = 保留 + 新增
    final_lh_rids = keep_rids | add_rids
    assert len(final_lh_rids) == n_lh, f"LH 数量变了: {len(final_lh_rids)} != {n_lh}"

    # 构建所有需要用到的 LH 任务(build_lh_tasks 一次解析 523 条,按 rid 取)
    all_lh = {task["rid"]: task for task in build_lh_tasks()}

    # 新的 parquet 行:非 LH 行原样保留;LH 行整体重建为 final_lh_rids(难度用轨迹打分)
    non_lh = [r for r in rows if not r["extra_info"].get("gen_task_id", "").startswith("LH_")]
    new_lh_rows = []
    for rid in sorted(final_lh_rids):
        task = all_lh.get(rid)
        if task is None:
            print(f"⚠️ {rid} 不在源文件,跳过", flush=True)
            continue
        new_lh_rows.append(build_lh_row(task, traj_diff[rid]))

    final_rows = non_lh + new_lh_rows
    assert len(final_rows) == len(rows), f"总行数变了: {len(final_rows)} != {len(rows)}"

    # 统计
    bc = Counter(r["bucket"] for r in final_rows)
    lh_diff = Counter(int(r["extra_info"]["difficulty"]) for r in new_lh_rows)
    print(f"\n=== 最终 ===", flush=True)
    print(f"总行数 {len(final_rows)} (batch32 对齐: {len(final_rows) % 32 == 0})", flush=True)
    print(f"coding={bc['coding']} office={bc['office']}", flush=True)
    print(f"LH {len(new_lh_rows)} 个, 难度分布: {dict(sorted(lh_diff.items()))}", flush=True)
    d47 = sum(v for k, v in lh_diff.items() if 4 <= k <= 7)
    print(f"LH d4-7: {d47}/{len(new_lh_rows)} (应=全部)", flush=True)

    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    new_t = pa.Table.from_pylist(final_rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    tmp_j = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_j, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_j.replace(JSONL)
    print(f"\n✅ 落地 {PARQUET.name} + {JSONL.name} ({len(final_rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
