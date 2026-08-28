#!/usr/bin/env python3
"""AI 核验:974 个无 files/ 的任务是否【合法】(真不需要外部输入文件)。

无 files/ 的两类:797 个 D `_s`(产出型/from-scratch) + 177 LH(自包含迁移,源码内嵌 query)。
让 luna 逐条判:query 是否真的【不依赖任何需预置的外部输入文件】(自包含=合法),
还是【引用了必须先存在的输入文件却缺失】(不合法=真缺文件)。

产出型标志:从零构建、query 内嵌全部数据/源码、只要求写产出文件。
不合法标志:query 说"读取/基于/使用 xxx.csv 分析",但那文件没内嵌、也不在 files/。

用法: python3 scripts/data/verify_nofiles_legit.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "datasources" / "taskspecs_w3"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
OUT = ROOT / "datasources" / "labeled" / "nofiles_legit_check.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "gpt-5.6-luna"
WORKERS = 64

PROMPT = """你在核验一个 agent 编程任务是否【自包含】(不需要任何预先放置的外部输入文件)。

给你任务描述(query)。判断:完成这个任务,agent 是否需要读取【某个必须由外部预置、但没有出现在任务描述里】的输入文件?

- **自包含(合法)**:满足任一即是——① 从零构建/from-scratch(自己造数据或生成样例);② 所需源码/数据【已内嵌在任务描述里】(如"读以下 Python 代码:...");③ 只要求产出文件(写报告/写脚本),不读外部文件;④ agent 自己先创建再处理。
- **缺文件(不合法)**:任务明确要求"读取/基于/使用 某文件(csv/xlsx/json 等)"来完成,但该文件内容【既没内嵌在描述里】、任务也没让 agent 自己造。

只输出 JSON:{"self_contained": true 或 false, "reason": "一句话依据"}"""


def load_key():
    key = os.environ.get("TOKENHUB_API_KEY", "") or os.environ.get("AGENT_MODEL_KEY", "")
    if not key:
        for envf in (ROOT / ".env", ROOT / "docker" / "sandbox" / "runtime.env"):
            if envf.is_file():
                for line in envf.read_text(encoding="utf-8").splitlines():
                    for k in ("TOKENHUB_API_KEY=", "AGENT_MODEL_KEY="):
                        if line.strip().startswith(k):
                            key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        sys.exit("no key")
    return key


def parse_json(t):
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if len(t.split("```")) > 1 else t
        if t.startswith("json"):
            t = t[4:]
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        t = t[s:e + 1]
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        return {}


def main():
    key = load_key()
    rows = pq.read_table(str(PARQUET)).to_pylist()
    tasks = []
    for r in rows:
        gid = r["extra_info"]["gen_task_id"]
        fd = TS / gid / "files"
        if not (fd.is_dir() and any(fd.iterdir())):
            tasks.append((gid, (r["extra_info"].get("queries") or [""])[0]))
    print(f"无 files/ 待核验: {len(tasks)}", flush=True)

    done = {}
    if OUT.is_file():
        for l in open(OUT, encoding="utf-8"):
            try:
                d = json.loads(l)
                done[d["gid"]] = d
            except Exception:  # noqa: BLE001
                pass
    todo = [t for t in tasks if t[0] not in done]
    print(f"待跑 {len(todo)} ({len(done)} 已跑)", flush=True)

    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(90.0))
    fout = open(OUT, "a", encoding="utf-8")
    ok = 0
    t0 = time.time()

    def check(t):
        gid, q = t
        for _ in range(3):
            try:
                resp = client.post(f"{API_BASE}/chat/completions", json={
                    "model": MODEL, "messages": [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f"任务描述:\n{q[:8000]}"}],
                    "max_tokens": 256, "temperature": 0.0}, timeout=90.0)
                resp.raise_for_status()
                p = parse_json(resp.json()["choices"][0]["message"]["content"])
                if "self_contained" in p:
                    return gid, bool(p["self_contained"]), p.get("reason", "")
            except Exception:  # noqa: BLE001
                time.sleep(1.5)
        return gid, None, "judge_failed"

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for gid, sc, reason in ex.map(check, todo):
            rec = {"gid": gid, "self_contained": sc, "reason": reason}
            done[gid] = rec
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            if ok % 100 == 0 or ok == len(todo):
                print(f"  {ok}/{len(todo)} rate={ok/(time.time()-t0):.1f}/s", flush=True)
    fout.close()

    # 汇总
    vals = list(done.values())
    legit = sum(1 for v in vals if v["self_contained"] is True)
    illegit = [v["gid"] for v in vals if v["self_contained"] is False]
    failed = sum(1 for v in vals if v["self_contained"] is None)
    print(f"\n=== 核验汇总(共 {len(vals)}) ===", flush=True)
    print(f"  合法(自包含): {legit}", flush=True)
    print(f"  不合法(疑缺文件): {len(illegit)}", flush=True)
    print(f"  judge 失败: {failed}", flush=True)
    if illegit:
        print(f"  不合法清单(前20): {illegit[:20]}", flush=True)
        # 来源分布
        c = Counter("LH" if g.startswith("LH_") else "D" for g in illegit)
        print(f"  不合法来源: {dict(c)}", flush=True)
    print(f"\n清单: {OUT}", flush=True)


if __name__ == "__main__":
    main()
