#!/usr/bin/env python3
"""数据层修复：把任务指令里的根目录路径 /workspace 改写为可写位置 /home/user/workspace。

背景（2026-08-22）：Hermes write_file 的 _atomic_write 把临时文件 .hermes-tmp 建在【目标
父目录】。沙箱用户 user(uid1000) 无权限在根目录 / 下建目录，故任务指令里引用 /workspace/...
的（1661 条）全部写盘失败 → task_done=0 → 污染 GRPO 组基线。真机验证见 scripts/verify_write_fix.py。

修复不用非必要 sudo，从数据层解决：/workspace → /home/user/workspace（默认 cwd、可写）。
- 只改【路径根】/workspace，保留后缀（/workspace/app.py → /home/user/workspace/app.py）。
- 后瞻含中文标点，排除 /workspaces 复数别词（如 API 路径 /workspaces/subscriptions）。
- 同步改 train_cl.parquet（训练实际读）+ train_cl.jsonl（源），prompt + extra_info 两处。

用法:
  python3 scripts/data/rewrite_workspace_path.py            # 干跑(默认)
  python3 scripts/data/rewrite_workspace_path.py --apply    # 落地
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OLD = "/workspace"
NEW = "/home/user/workspace"
# 路径根：/workspace 后面是 / 或 结束/标点（含中文标点）；s 不在后瞻集 → /workspaces 天然排除。
PAT = re.compile(r'/workspace(?=$|[/\s.,;:\'"`)\]}，。、；：）】！？])')

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "datasets/train_cl.parquet"
JSONL = ROOT / "datasets/train_cl.jsonl"


def rewrite_obj(obj):
    """递归改写任意嵌套结构里的字符串。"""
    if isinstance(obj, str):
        return PAT.sub(NEW, obj)
    if isinstance(obj, list):
        return [rewrite_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: rewrite_obj(v) for k, v in obj.items()}
    return obj


def rewrite_jsonl(apply: bool) -> int:
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    changed = 0
    out = []
    for r in rows:
        nr = rewrite_obj(r)
        if nr != r:
            changed += 1
        out.append(nr)
    if apply:
        tmp = JSONL.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(JSONL)
    return changed


def rewrite_parquet(apply: bool) -> int:
    import pyarrow.parquet as pq
    import pyarrow as pa

    t = pq.read_table(PARQUET)
    rows = t.to_pylist()
    changed = 0
    out = []
    for r in rows:
        nr = rewrite_obj(r)
        if nr != r:
            changed += 1
        out.append(nr)
    if apply:
        new_t = pa.Table.from_pylist(out, schema=t.schema)
        tmp = PARQUET.with_suffix(".parquet.tmp")
        pq.write_table(new_t, tmp)
        tmp.replace(PARQUET)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="落地写回（默认干跑）")
    args = ap.parse_args()

    jc = rewrite_jsonl(args.apply)
    pc = rewrite_parquet(args.apply)
    mode = "已落地" if args.apply else "干跑(未写)"
    print(f"[{mode}] train_cl.jsonl 改写 {jc} 行；train_cl.parquet 改写 {pc} 行")
    if not args.apply:
        print("确认无误后加 --apply 落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
