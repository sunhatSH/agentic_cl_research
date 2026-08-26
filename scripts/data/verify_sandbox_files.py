#!/usr/bin/env python3
"""全量验证 train_cl.parquet:每个任务的文件能否正确进入沙箱 + 路径正确。

复现 build_agent_assets 的查找逻辑(4 条分支),对每行:
  - query 引用读文件路径 → 文件源必须能找到(否则 agent 读不到)
  - 文件源找到 → 注入沙箱的目标目录(inputs/workspace)必须和 query 路径对齐
  - 不读文件的题(产出型)→ 无需文件,跳过

输出:全量结果 + 找不到文件源的清单 + 路径不对齐的清单。
"""
import json, os, sys, re
from collections import Counter
import pyarrow.parquet as pq
sys.path.insert(0, "scripts/pipeline")
from path_normalize import extract_gen_task_id

ROOT = os.getcwd()
idx = json.load(open("datasources/review_ws/index.json")) if os.path.exists("datasources/review_ws/index.json") else {}

def build_assets(rid, gid):
    """复现 build_agent_assets:返回 (源目录, 沙箱目标) or (None, None)。"""
    if gid:
        dp = gid.split("_")[0]
        p = os.path.join("datasources/generated_tasks_hermes", dp, gid, "inputs")
        if os.path.isdir(p) and os.listdir(p):
            return (p, "inputs")
    p = os.path.join("datasources/taskspecs_w3", rid, "files")
    if os.path.isdir(p) and os.listdir(p):
        return (p, "inputs")
    p = os.path.join("datasources/seed2traj_taskspecs", rid, "files")
    if os.path.isdir(p) and os.listdir(p):
        return (p, "inputs")
    rel = idx.get(rid)
    if rel:
        p = os.path.join("datasources/review_ws", rel, "ws")
        if os.path.isdir(p) and os.listdir(p):
            return (p, "workspace")
    return (None, None)

# 读文件路径模式(宽口径:./inputs/ /home/user/workspace/ inputs/ data/ files/ origin/)
READ_PAT = re.compile(
    r'\./inputs/|/home/user/workspace/|(?:^|[^./\w])inputs/|inputs\\|'
    r'(?:^|[^./\w])data/|(?:^|[^./\w])files/|origin/'
)

rows = pq.read_table("datasets/train_cl.parquet").to_pylist()
total = len(rows)
no_read = 0
found = Counter()
missing = []
misalign = []

for r in rows:
    ei = r["extra_info"]
    rid = ei["record_id"]
    gid = ei.get("gen_task_id") or ""
    q = str((ei.get("queries") or [""])[0])
    if not READ_PAT.search(q):
        no_read += 1
        continue
    src, target = build_assets(rid, gid)
    if src:
        found[target] += 1
        # 路径对齐检查:query 引用 /home/user/workspace → target 应是 workspace;引用 ./inputs → target 应是 inputs
        if "/home/user/workspace" in q and target != "workspace":
            misalign.append((rid, "引用workspace但注入"+target, q[:80]))
        elif "./inputs" in q and target != "inputs":
            misalign.append((rid, "引用./inputs但注入"+target, q[:80]))
    else:
        missing.append((rid, gid, q[:100]))

print(f"=== 全量验证 {total} 行 ===")
print(f"  不读文件(产出型,无需注入): {no_read}")
print(f"  读文件且文件源找到: {sum(found.values())} (注入目标: {dict(found)})")
print(f"  读文件但文件源找不到: {len(missing)}")
print(f"  路径不对齐: {len(misalign)}")
print(f"\n=== 找不到文件源的 {len(missing)} 个 ===")
for rid, gid, q in missing[:15]:
    idt = "非unk" if not rid.startswith("unk_") else "unk"
    print(f"  [{rid}]({idt}) gid={gid}: {q}")
if len(missing) > 15:
    print(f"  ... 还有 {len(missing)-15} 个")
print(f"\n=== 路径不对齐的 {len(misalign)} 个 ===")
for rid, why, q in misalign[:10]:
    print(f"  [{rid}] {why}: {q}")
