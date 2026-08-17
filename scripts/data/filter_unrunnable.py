#!/usr/bin/env python3
"""Filter queries that CANNOT run in a pure-Linux sandbox (no user local disk,
no live external channels like 飞书/微信 push, no Windows drives).

LLM judges each seed query: given that the sandbox is a fresh Linux container
with ONLY the taskspec's uploaded workspace files (no user's E:\\ drive, no live
飞书/webhook connection, no host mounts), can the task produce a MEANINGFUL
artifact? Tasks that merely READ bundled files / analyze / write reports = OK.
Tasks that REQUIRE a live external system the sandbox lacks = DROP.

Reads datasets/queries.jsonl, writes:
  - datasets/queries.jsonl           (kept rows, in place)
  - datasets/queries_dropped.jsonl   (dropped rows + reason, for audit)

Usage:
    python scripts/data/filter_unrunnable.py --workers 32 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_QUERIES = _REPO / "datasets" / "queries.jsonl"
_DROPPED = _REPO / "datasets" / "queries_dropped.jsonl"

_SYSTEM = (
    "你是一个任务可执行性判定器。判定一个给 agent 的任务，在如下环境里能否产出"
    "有意义的结果：\n"
    "环境 = 一个全新的 Linux 沙箱容器，里面只有本任务预置的工作区文件（若有），"
    "没有用户本机磁盘（如 Windows E:\\ / C:\\ 盘）、没有连接实时外部系统"
    "（真实飞书/微信/钉钉消息通道、真实 webhook、用户本机挂载盘、真实 cron 守护进程）。\n\n"
    "判定标准：\n"
    "- RUNNABLE：任务主要是读取/分析预置文件、写文档、处理数据、生成代码、回答问题。"
    "即使提到‘心跳/cron/飞书/HEARTBEAT.md’，只要核心工作是读预置文件+产出文件，就算 RUNNABLE。\n"
    "- UNRUNNABLE：任务的核心成果必须依赖沙箱没有的东西——必须访问用户本机具体磁盘路径里的"
    "真实文件、必须真实推送飞书/微信消息、必须连真实 webhook/外部服务验证、必须操作真实的"
    "常驻定时守护进程。没有这些就无法产出任何有意义结果。\n\n"
    "只输出一个 JSON：{\"runnable\": true/false, \"reason\": \"一句话\"}"
)


def _judge(query: str, client) -> dict:
    msgs = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"任务：\n{query}\n\n输出 JSON。"},
    ]
    try:
        raw = client.chat(msgs, max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        return {"runnable": True, "reason": f"judge_error_keep: {exc}"}  # 出错保守保留
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {"runnable": True, "reason": "parse_fail_keep"}
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {"runnable": True, "reason": "parse_fail_keep"}
    return {"runnable": bool(obj.get("runnable", True)), "reason": str(obj.get("reason", ""))[:150]}


def _ensure_key() -> None:
    if os.environ.get("TOKENHUB_API_KEY", "").strip():
        return
    env = _REPO / "docker" / "sandbox" / "runtime.env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("AGENT_MODEL_KEY") and "=" in line:
                os.environ["TOKENHUB_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="only report, do not modify queries.jsonl")
    args = ap.parse_args()

    _ensure_key()
    from data_pipeline.classify import make_default_client
    client = make_default_client()

    rows = []
    with open(_QUERIES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[filter] {len(rows)} queries, judging runnability ({args.workers} workers)")

    verdicts: dict[int, dict] = {}

    def _one(i: int) -> tuple[int, dict]:
        return i, _judge(rows[i]["queries"][0], client)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, i): i for i in range(len(rows))}
        with tqdm(total=len(rows), desc="Judging", unit="q", smoothing=0.01) as pbar:
            drop = 0
            for fut in as_completed(futs):
                i, v = fut.result()
                verdicts[i] = v
                if not v["runnable"]:
                    drop += 1
                pbar.set_postfix(drop=drop, refresh=False)
                pbar.update(1)

    kept = [rows[i] for i in range(len(rows)) if verdicts[i]["runnable"]]
    dropped = [{**rows[i], "drop_reason": verdicts[i]["reason"]}
               for i in range(len(rows)) if not verdicts[i]["runnable"]]

    print(f"[filter] kept={len(kept)}  dropped={len(dropped)}")
    print("[filter] dropped 示例:")
    for r in dropped[:10]:
        print(f"  [{r['record_id']}] {r['queries'][0][:50]} → {r['drop_reason']}")

    if args.dry_run:
        print("[filter] DRY RUN — 未修改文件")
        return

    # 备份 + 写 kept + 写 dropped 审计
    import shutil
    shutil.copy(_QUERIES, str(_QUERIES) + ".bak_prefilter")
    with open(_QUERIES, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(_DROPPED, "w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[filter] queries.jsonl → {len(kept)} 行 (备份 .bak_prefilter)")
    print(f"[filter] 丢弃审计 → {_DROPPED}")


if __name__ == "__main__":
    main()
