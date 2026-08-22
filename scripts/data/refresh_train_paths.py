#!/usr/bin/env python3
"""增量修正 train_cl.parquet：重跑 path_normalize（修 ws→workspace / bare /workspace）+ 补 ws_dir。

不做全量重建（build_train.py 的 batch/难度均衡逻辑与当前 64-batch 实验可能漂移）——
保持现有行序/step 结构完全不变，只对每行：
  1. 用【原始 seed_query】(labeled) 重跑 normalize_paths → 修正 query 路径
     （旧 F4 把采集 ws 错误归到 ./outputs，新规则归到 /home/user/workspace）；
  2. extract_ws_dir 补进 extra_info.ws_dir（供 cl_agent_dataset 注入 ws 快照）。
prompt[user] 与 extra_info.queries[0] 同步更新（二者都是 query 文本）。system prompt 不动。

依赖 record_id 关联 labeled（现有 parquet 12800 行 100% 命中）。同步写 .jsonl。

用法:
  python3 scripts/data/refresh_train_paths.py            # 干跑(统计变更)
  python3 scripts/data/refresh_train_paths.py --apply    # 落地
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from path_normalize import extract_ws_dir, normalize_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"


def _load_labeled_seed() -> dict[str, str]:
    m: dict[str, str] = {}
    for line in open(LABELED, encoding="utf-8"):
        d = json.loads(line)
        rid = d.get("record_id")
        if rid:
            m[rid] = d.get("seed_query", "")
    return m


def refresh(parquet_path: Path, apply: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    seed = _load_labeled_seed()
    t = pq.read_table(str(parquet_path))
    rows = t.to_pylist()

    n_query_changed = 0
    n_ws_added = 0
    n_no_labeled = 0
    for r in rows:
        ei = r.get("extra_info") or {}
        rid = ei.get("record_id")
        raw = seed.get(rid)
        if raw is None:
            n_no_labeled += 1
            continue
        new_q = normalize_paths(raw)
        # 修正 query 文本（prompt 末条 user + extra_info.queries[0]）
        old_q = ei.get("queries", [None])[0] if ei.get("queries") else None
        if new_q != old_q:
            n_query_changed += 1
        ei["queries"] = [new_q]
        # prompt 里 role=user 的 content 同步
        for msg in r.get("prompt") or []:
            if msg.get("role") == "user":
                msg["content"] = new_q
        # 补 ws_dir
        ws = extract_ws_dir(raw) or ""
        if ws and not ei.get("ws_dir"):
            n_ws_added += 1
        ei["ws_dir"] = ws
        r["extra_info"] = ei

    print(f"[{'apply' if apply else 'dry'}] {parquet_path.name}: {len(rows)} 行")
    print(f"  query 路径被修正: {n_query_changed}")
    print(f"  补 ws_dir(非空): {n_ws_added}")
    print(f"  record_id 不在 labeled: {n_no_labeled}")

    if apply:
        # extra_info 是固定 struct schema，需扩展加入 ws_dir 字段（原 schema 无此键）。
        ei_field = t.schema.field("extra_info")
        ei_type = ei_field.type
        if "ws_dir" not in [ei_type.field(i).name for i in range(ei_type.num_fields)]:
            new_ei_type = pa.struct(
                [ei_type.field(i) for i in range(ei_type.num_fields)]
                + [pa.field("ws_dir", pa.string())]
            )
            new_schema = pa.schema(
                [f if f.name != "extra_info" else pa.field("extra_info", new_ei_type)
                 for f in t.schema]
            )
        else:
            new_schema = t.schema
        new_t = pa.Table.from_pylist(rows, schema=new_schema)
        tmp = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(new_t, tmp)
        tmp.replace(parquet_path)
        # 同步 jsonl
        jsonl = parquet_path.with_suffix(".jsonl")
        if jsonl.exists():
            tmpj = jsonl.with_suffix(".jsonl.tmp")
            with open(tmpj, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmpj.replace(jsonl)
        print(f"  ✅ 写回 {parquet_path.name} + .jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--parquet", default=str(ROOT / "datasets" / "train_cl.parquet"))
    args = ap.parse_args()
    refresh(Path(args.parquet), args.apply)
    if not args.apply:
        print("确认无误后加 --apply 落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
