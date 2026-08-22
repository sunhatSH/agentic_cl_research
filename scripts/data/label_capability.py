#!/usr/bin/env python3
"""给 taskspec 打【能力桶】标签 → 带桶标的 queries JSONL（数据 pipeline 打标步）。

单层能力桶（无子桶）。桶定义读自 capability_bucket_discovery.py 的产物 buckets.json。
对每条 taskspec，让 GPT 从固定的能力桶里选【唯一一个】最匹配的能力桶。

Input  : taskspec 目录（每个 s_<id>/taskspec.yaml），默认 datasources/seed2traj_taskspecs
         能力桶定义 runs/_analysis/capability_buckets/buckets.json
Output : 带桶标 + 训练字段的 JSONL，每行一个任务：
           {"record_id","bucket","queries":[seed, *follow_ups],
            "hidden_goal","persona","available_tools","missing_info_slots",
            "safety_constraints","difficulty"}
         —— queries[0]=seed_query（rollout 起点）；bucket=能力桶（进 buffer/训练/评测）。

走 tokenhub（本地）。断点续跑：已打标的 record_id 跳过（--resume）。

用法：
  python scripts/data/label_capability.py --limit 5           # 冒烟
  python scripts/data/label_capability.py                     # 全量 4941
  python scripts/data/label_capability.py --resume            # 断点续
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKSPECS = ROOT / "datasources" / "seed2traj_taskspecs"
BUCKETS_JSON = ROOT / "runs" / "_analysis" / "capability_buckets" / "buckets.json"
OUT_DIR = ROOT / "datasources" / "labeled"
API_BASE = os.environ.get("DISCOVERY_API_BASE", "https://tokenhub.sensetime.com/v1")


def _load_key() -> str:
    key = os.environ.get("DISCOVERY_API_KEY", "") or os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("TOKENHUB_API_KEY=") or s.startswith("DISCOVERY_API_KEY="):
                    key = s.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        sys.exit("ERROR: TOKENHUB_API_KEY not set (see .env)")
    return key


def _chat(messages: list[dict], model: str, key: str, max_tokens: int = 256, retries: int = 3) -> str:
    import time

    import httpx

    last = None
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {key}"},
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    end = text.rfind("}")
    if end > 0:
        text = text[: end + 1]
    return json.loads(text)


def _load_buckets() -> tuple[list[str], str]:
    if not BUCKETS_JSON.is_file():
        sys.exit(f"ERROR: {BUCKETS_JSON} 不存在，先跑 capability_bucket_discovery.py")
    d = json.loads(BUCKETS_JSON.read_text(encoding="utf-8"))
    names = [b["name"] for b in d["buckets"]]
    defs = "\n".join(f"- {b['name']} ({b.get('cn', '')}): {b.get('definition', '')}" for b in d["buckets"])
    return names, defs


def _load_taskspecs(limit: int) -> list[dict]:
    out = []
    for dpath in sorted(TASKSPECS.iterdir()):
        if not dpath.is_dir():
            continue
        f = dpath / "taskspec.yaml"
        if not f.is_file():
            continue
        try:
            ts = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(ts, dict):
            out.append(ts)
    if limit > 0:
        out = out[:limit]
    return out


def _queries(ts: dict) -> list[str]:
    qs = []
    seed = ts.get("seed_query")
    if isinstance(seed, str) and seed.strip():
        qs.append(seed.strip())
    prof = ts.get("user_profile") or {}
    fus = prof.get("follow_ups") if isinstance(prof, dict) else None
    if isinstance(fus, list):
        qs += [q.strip() for q in fus if isinstance(q, str) and q.strip()]
    cm = ts.get("correction_message")
    if isinstance(cm, str) and cm.strip():
        qs.append(cm.strip())
    return qs


_SYSTEM = textwrap.dedent("""\
    你是能力分类器。给你一个 agent 任务，请判断【完成它主要依赖哪一种能力】，
    从下面的固定能力桶里选【唯一一个】最匹配的。

    ## 能力桶（按能力划分，不是话题）
    {bucket_defs}

    ## 规则
    - 只选一个 bucket，必须是上面列表里的英文桶名之一。
    - 按"完成任务所需的核心本领"选，不要按业务话题（金融/邮件/医疗等都是话题，不是依据）。
    - 只输出 JSON: {{"bucket": "<桶名>", "reason": "一句话理由"}}，不要 markdown。
""")

_USER = textwrap.dedent("""\
    seed_query: {seed}
    hidden_goal: {goal}
    available_tools: {tools}

    这个任务主要依赖哪个能力桶？只回 JSON。
""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过已打标的 record_id")
    ap.add_argument("--model", default=os.environ.get("DISCOVERY_MODEL", "claude-opus-4-8"))
    ap.add_argument("--out", default=str(OUT_DIR / "taskspecs_labeled.jsonl"))
    ap.add_argument("--workers", type=int, default=16, help="并发线程数(tokenhub 并发打标)")
    args = ap.parse_args()

    key = _load_key()
    names, defs = _load_buckets()
    print(f"[label] 能力桶 {len(names)} 个: {names}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["record_id"])
        print(f"[label] resume: 已打标 {len(done)} 条，跳过", flush=True)

    tasks = _load_taskspecs(args.limit)
    todo = [ts for ts in tasks if (ts.get("task_id") or "") not in done]
    print(f"[label] 载入 {len(tasks)} 条 taskspec，待打标 {len(todo)} 条 "
          f"(源 {TASKSPECS}, 并发 {args.workers})", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    system = _SYSTEM.format(bucket_defs=defs)
    valid = set(names)
    lock = threading.Lock()
    fout = out_path.open("a", encoding="utf-8")
    counters = {"ok": 0, "fail": 0}

    def _label_one(ts: dict) -> None:
        rid = ts.get("task_id") or ""
        user = _USER.format(
            seed=(ts.get("seed_query", "") or "")[:1200],
            goal=(ts.get("hidden_goal", "") or "")[:1200],
            tools=ts.get("available_tools", []),
        )
        try:
            out = _chat([{"role": "system", "content": system},
                         {"role": "user", "content": user}], args.model, key)
            bucket = _parse_json(out).get("bucket", "")
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"FAIL {rid}: {exc}")
            with lock:
                counters["fail"] += 1
            return
        # 空/非法桶 -> 强制重问一次;仍不合法才标 unknown
        if bucket not in valid:
            try:
                retry_user = user + (
                    f"\n\n上次未给出有效桶。必须从 {names} 里选【唯一一个】最匹配的英文桶名，"
                    "不许为空、不许编造新名。只回 JSON。"
                )
                out = _chat([{"role": "system", "content": system},
                             {"role": "user", "content": retry_user}], args.model, key)
                bucket = _parse_json(out).get("bucket", "")
            except Exception:  # noqa: BLE001
                bucket = ""
        if bucket not in valid:
            tqdm.write(f"WARN {rid}: 重问后仍无效 bucket={bucket!r}，标 unknown")
            bucket = "unknown"
        prof = ts.get("user_profile") or {}
        rec = {
            "record_id": rid,
            "bucket": bucket,
            "queries": _queries(ts),
            "hidden_goal": ts.get("hidden_goal", ""),
            "persona": prof.get("style", "") if isinstance(prof, dict) else "",
            "available_tools": ts.get("available_tools", []),
            "missing_info_slots": ts.get("missing_info_slots", []),
            "safety_constraints": ts.get("safety_constraints", []),
            "difficulty": ts.get("difficulty", ""),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with lock:
            fout.write(line)
            fout.flush()
            counters["ok"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(_label_one, todo), total=len(todo), desc="labeling"))

    fout.close()
    print(f"\n[label] done: 成功 {counters['ok']} / 失败 {counters['fail']} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
