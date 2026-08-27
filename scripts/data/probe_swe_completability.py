#!/usr/bin/env python3
"""SWE 任务可完成性体检:沙箱 + luna 实跑 + rubric 评判。

背景:train_cl.parquet 里 2931 个 SWE 任务(sweCoding,git 类 coding)是稀疏抽取
(中位数 6 文件,全无 .git)。用户担心一部分无法完成(缺源码文件 / 无 git 历史 /
rubric 有歧义)。做法:每个 SWE 用【沙箱 + hermes agent(模型=luna,强模型)】实跑,
再用 luna 按 rubric 逐条打分。luna 都拿不到高分 → 判定"可能无法完成"。

三段:
  1. precheck(零成本): files/ 是否全测试文件、有无源码、rubric 是否含 git/历史词。
  2. 实跑(沙箱+luna): 起 e2b 沙箱 → 注入 files 到 /home/user/workspace → hermes(luna)
     跑 → 收 final answer + 文件 diff(注入后 vs 跑完)。
  3. 评判(luna): rubric + agent 产出/diff → 逐条 0/1 → score = 命中比例(0-1)。

复用:sandbox_grpo_collect._write_hermes_config / make_sandbox / _load_ground_truth。

结果写 datasources/labeled/swe_completability.jsonl(断点续跑)。
score < 阈值 → likely_uncompletable,和 precheck flags 交叉出分类清单。

用法:
  source scripts/env/load_tencent_env.sh
  python3 scripts/data/probe_swe_completability.py --precheck-only      # 秒级预检
  python3 scripts/data/probe_swe_completability.py --limit 20           # 小样实跑
  python3 scripts/data/probe_swe_completability.py --workers 64         # 全量
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "datasets" / "train_cl.parquet"
TASKSPECS = ROOT / "datasources" / "taskspecs_w3"
OUT = ROOT / "datasources" / "labeled" / "swe_completability.jsonl"
PRECHECK_OUT = ROOT / "datasources" / "labeled" / "swe_precheck.jsonl"

API_BASE = "https://tokenhub.sensetime.com/v1"
JUDGE_MODEL = "gpt-5.6-luna"
ACTOR_MODEL = "gpt-5.6-luna"
WS = "/home/user/workspace"

# rubric 里出现 → 疑似依赖"原始仓库/历史/未改动基线"才能验收(我们无 .git,agent 无从对比)
_GIT_WORDS = re.compile(
    r"\b(unchanged|passed?\s+through|pass[- ]?through|as[- ]?is|exactly what|remaining|"
    r"existing behaviou?r|previous|original|regression|backward[- ]?compat)\b"
    r"|保持|原状|不变|原有|原始|回归",
    re.IGNORECASE,
)
_TEST_NAME = re.compile(r"(^test_|_test\.|\.test\.|(^|/)tests?/|\.spec\.|_spec\.)", re.IGNORECASE)


def load_key() -> str:
    key = os.environ.get("TOKENHUB_API_KEY", "") or os.environ.get("AGENT_MODEL_KEY", "")
    if not key:
        for envf in (ROOT / ".env", ROOT / "docker" / "sandbox" / "runtime.env"):
            if envf.is_file():
                for line in envf.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    for k in ("TOKENHUB_API_KEY=", "AGENT_MODEL_KEY="):
                        if s.startswith(k):
                            key = s.split("=", 1)[1].strip().strip("\"'")
                            break
                if key:
                    break
    if not key:
        sys.exit("ERROR: no TOKENHUB_API_KEY / AGENT_MODEL_KEY (see .env or runtime.env)")
    return key


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def load_swe_tasks() -> list[dict]:
    """从 parquet 读全部 SWE 行 → [{swe_id, query}]。"""
    rows = pq.read_table(str(PARQUET)).to_pylist()
    tasks = []
    for r in rows:
        ei = r["extra_info"]
        gid = ei.get("gen_task_id", "")
        if gid.startswith("SWE_"):
            q = (ei.get("queries") or [""])[0] or ""
            tasks.append({"swe_id": gid, "query": str(q)})
    return tasks


def precheck_one(swe_id: str, rubric_text: str) -> dict:
    """零成本预检:文件构成 + rubric git 词。"""
    fdir = TASKSPECS / swe_id / "files"
    files = [p for p in fdir.rglob("*") if p.is_file()] if fdir.is_dir() else []
    names = [str(p.relative_to(fdir)) for p in files]
    n_test = sum(1 for n in names if _TEST_NAME.search(n))
    n_src = len(names) - n_test
    flags = []
    if not files:
        flags.append("no_files")
    elif n_src == 0:
        flags.append("only_tests")  # 全测试文件,没源码可改
    if _GIT_WORDS.search(rubric_text or ""):
        flags.append("git_history_rubric")  # rubric 依赖原始基线/历史
    return {
        "n_files": len(files),
        "n_test": n_test,
        "n_src": n_src,
        "precheck_flags": flags,
    }


def load_rubric(swe_id: str) -> tuple[str, int]:
    """读 answer_key.json → (rubric 文本, 条目数)。"""
    ak_path = TASKSPECS / swe_id / "answer_key.json"
    if not ak_path.is_file():
        return "", 0
    try:
        ak = json.loads(ak_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "", 0
    rubric = ak.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    lines = [f"{i + 1}. {str(x).strip()}" for i, x in enumerate(rubric)]
    return "\n".join(lines), len(rubric)


# ─────────────────────────── 沙箱实跑 ───────────────────────────

def upload_workspace(sb, swe_id: str) -> int:
    """注入 files/ 到沙箱 /home/user/workspace/(绝对路径,和 query 对齐)。"""
    src = TASKSPECS / swe_id / "files"
    if not src.is_dir():
        return 0
    entries = []
    for fpath in sorted(src.rglob("*")):
        if not fpath.is_file() or fpath.name in (".DS_Store", "Thumbs.db") or fpath.name.startswith("~$"):
            continue
        rel = fpath.relative_to(src)
        entries.append({"path": f"{WS}/{rel}", "data": fpath.read_bytes()})
    if entries:
        sb._sb.files.write_files(entries)
    return len(entries)


def snapshot_ws(sb) -> dict[str, str]:
    """沙箱 /home/user/workspace 下每个文件的 (size,mtime) 指纹,用于 diff。"""
    code = (
        "import os,json\n"
        f"base={WS!r}\n"
        "out={}\n"
        "for r,_,fs in os.walk(base):\n"
        "  for f in fs:\n"
        "    p=os.path.join(r,f)\n"
        "    try: st=os.stat(p); out[p]=f'{st.st_size}:{int(st.st_mtime)}'\n"
        "    except Exception: pass\n"
        "print('__SNAP__'+json.dumps(out))\n"
    )
    res = sb.run_code(code)
    txt = (res.stdout or "")
    i = txt.find("__SNAP__")
    if i < 0:
        return {}
    try:
        return json.loads(txt[i + 8:].strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return {}


def diff_ws(before: dict, after: dict, sb, max_files: int = 12, max_bytes: int = 4000) -> str:
    """算 before/after 差异,读变更文件内容(截断)拼成 diff 文本给 judge。"""
    changed = [p for p, v in after.items() if before.get(p) != v]
    created = [p for p in after if p not in before]
    deleted = [p for p in before if p not in after]
    lines = [f"created: {len(created)}, modified: {len(changed) - len(created)}, deleted: {len(deleted)}"]
    show = changed[:max_files]
    for p in show:
        res = sb.run_code(
            "import sys\n"
            f"p={p!r}\n"
            "try:\n"
            "  d=open(p,'r',errors='replace').read()\n"
            f"  sys.stdout.write('__F__'+p+'\\n'+d[:{max_bytes}])\n"
            "except Exception as e: sys.stdout.write('__F__'+p+'\\n<read err>')\n"
        )
        lines.append(f"\n=== {p} ===\n{(res.stdout or '').split('__F__', 1)[-1][:max_bytes]}")
    if deleted:
        lines.append(f"\ndeleted files: {deleted[:10]}")
    return "\n".join(lines)


JUDGE_PROMPT = """你是严格的代码任务验收评判器。给定任务描述、验收标准(rubric)、agent 的最终产出(final answer + 沙箱文件改动 diff)。

只评两个维度(其他一律不看):

1. task_done (0 或 1) — agent 是否真的【完成】了任务?
   - 按沙箱文件改动 diff 的【真实产物】判断,不看 agent 嘴上声称。
   - 需要改代码/产出文件的任务:diff 里没有出现要求的改动 → task_done=0。
   - 完成=真完成(产物齐全),不是"看起来努力过"。产物不全/半截/放弃 → 0。
   - 若任务【根本无法完成】(所需源码文件不在工作区 / 需对比原始仓库或 git 历史但沙箱没有)→ task_done=0。

2. correctness [0,1] — 产出是否【正确】?
   - 从 1.0 起扣分:对照 rubric 每条验收标准,不满足一条扣一部分;实现有 bug/跑不通/缺校验都扣。
   - 全错或完全跑偏 → 0。
   - 约束:task_done=0 时 correctness 最高只能 0.5(没完成不可能高正确)。

只输出 JSON(不要其他维度、不要逐条):
{"task_done":0或1,"correctness":<0到1的小数>,"note":"一句话,若无法完成写明原因(缺文件/无git基线/实现不全等)"}"""


def judge_one(client, key, task_query: str, rubric_text: str, n_rubric: int, answer: str, diff_text: str) -> dict:
    user = (
        f"# 任务描述\n{task_query[:4000]}\n\n"
        f"# 验收标准(rubric)\n{rubric_text}\n\n"
        f"# agent final answer\n{(answer or '<空>')[:4000]}\n\n"
        f"# 沙箱文件改动 diff\n{diff_text[:8000]}\n\n"
        "只评 task_done 和 correctness,只回 JSON。"
    )
    for attempt in range(3):
        try:
            resp = client.post(
                f"{API_BASE}/chat/completions",
                json={"model": JUDGE_MODEL, "messages": [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user}],
                    "max_tokens": 1024, "temperature": 0.0},
                timeout=120.0,
            )
            resp.raise_for_status()
            parsed = parse_json(resp.json()["choices"][0]["message"]["content"])
            if parsed and "task_done" in parsed:
                return parsed
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return {}



def run_one(task: dict, key: str, max_turns: int, slot_timeout: int) -> dict:
    """一个 SWE 的完整体检。返回结果 dict(含 precheck + score + verdict)。"""
    from rollout.sandbox_client import make_sandbox

    swe_id = task["swe_id"]
    query = task["query"]
    rubric_text, n_rubric = load_rubric(swe_id)
    pre = precheck_one(swe_id, rubric_text)
    result = {"swe_id": swe_id, "n_rubric": n_rubric, **pre}

    sb = None
    try:
        # 沙箱驱动复用 sandbox_grpo_collect 的 hermes 配置
        from scripts.collect.sandbox_grpo_collect import _hermes_chat, _write_hermes_config

        sb = make_sandbox("e2b", template="agentic-cl-sandbox", timeout=slot_timeout + 600)
        n_up = upload_workspace(sb, swe_id)
        result["n_uploaded"] = n_up
        cfg = _write_hermes_config(sb, ACTOR_MODEL, "")
        if not cfg.ok:
            result["error"] = f"hermes config: {cfg.stderr[:150]}"
            return result

        before = snapshot_ws(sb)
        # query 里路径已归一化为 /home/user/workspace 绝对路径,文件注入同处,hermes 直接可达
        stdout, stderr, ok, _sid = _hermes_chat(sb, query, ACTOR_MODEL, max_turns, slot_timeout)
        result["agent_ok"] = bool(ok)
        after = snapshot_ws(sb)
        diff_text = diff_ws(before, after, sb)
        result["n_changed"] = len([p for p, v in after.items() if before.get(p) != v])

        client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(120.0))
        verdict = judge_one(client, key, query, rubric_text, n_rubric, stdout[-4000:], diff_text)
        if not verdict:
            # 评判器没返回有效 JSON(截断/网络)——不是任务失败,不写 score,标 error 让 resume 重试
            result["error"] = "judge_failed"
        else:
            task_done = 1 if int(verdict.get("task_done", 0)) == 1 else 0
            correctness = max(0.0, min(1.0, float(verdict.get("correctness", 0.0))))
            if task_done == 0:
                correctness = min(correctness, 0.5)  # 没完成 correctness 封顶 0.5
            result["task_done"] = task_done
            result["correctness"] = round(correctness, 3)
            # score = 完成×正确(只看这两维);未完成→按封顶后的 correctness,天然低分
            result["score"] = round(correctness if task_done else min(correctness, 0.49), 3)
            result["judge_note"] = (verdict.get("note") or "")[:300]
    except Exception as exc:  # noqa: BLE001 — 单任务失败不中断整批
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if sb is not None:
            try:
                sb.kill()
            except Exception:  # noqa: BLE001
                pass
    return result


def classify(r: dict, threshold: float) -> str:
    """按 task_done + correctness 判定(只看这两维)。

    completable = task_done==1 且 correctness>=threshold(真完成且够正确)。
    否则按 precheck flags 归因为什么完不成。
    """
    score = r.get("score")
    flags = r.get("precheck_flags", [])
    if r.get("error") and score is None:
        return "run_error"
    if r.get("task_done") == 1 and float(r.get("correctness", 0)) >= threshold:
        return "completable"
    # 未完成 / 正确性不足 → 归因
    if "no_files" in flags or "only_tests" in flags:
        return "likely_uncompletable:missing_files"
    if "git_history_rubric" in flags:
        return "likely_uncompletable:no_git_history"
    return "likely_uncompletable:low_score"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck-only", action="store_true", help="只跑零成本预检")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-turns", type=int, default=30, help="hermes ReAct 上限")
    ap.add_argument("--slot-timeout", type=int, default=1200, help="单沙箱 hermes 超时(秒)")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    args = ap.parse_args()

    tasks = load_swe_tasks()
    print(f"SWE 任务: {len(tasks)} 个", flush=True)

    # ── 预检(始终跑,零成本)AFS rglob 慢,并发跑;已有结果则复用 ──
    PRECHECK_OUT.parent.mkdir(parents=True, exist_ok=True)

    def _pre(t):
        rubric_text, n_rubric = load_rubric(t["swe_id"])
        pre = precheck_one(t["swe_id"], rubric_text)
        return {"swe_id": t["swe_id"], "n_rubric": n_rubric, **pre}

    pre_rows = []
    if PRECHECK_OUT.is_file() and not args.precheck_only:
        with open(PRECHECK_OUT, encoding="utf-8") as f:
            pre_rows = [json.loads(line) for line in f if line.strip()]
        print(f"复用已有预检 {len(pre_rows)} 条 ({PRECHECK_OUT.name})", flush=True)
    if len(pre_rows) != len(tasks):
        pre_rows = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_pre, tasks), 1):
                pre_rows.append(r)
                if i % 500 == 0 or i == len(tasks):
                    print(f"  precheck {i}/{len(tasks)} ({i / (time.time() - t0):.0f}/s)", flush=True)
        with open(PRECHECK_OUT, "w", encoding="utf-8") as f:
            for r in pre_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fc = Counter()
    for r in pre_rows:
        for fl in (r["precheck_flags"] or ["ok"]):
            fc[fl] += 1
    print(f"\n=== 预检({PRECHECK_OUT.name})===", flush=True)
    for k, v in fc.most_common():
        print(f"  {k}: {v}", flush=True)

    if args.precheck_only:
        print("\n--precheck-only,不实跑沙箱", flush=True)
        return 0

    # ── 实跑(沙箱+luna)断点续跑 ──
    key = load_key()
    done = {}
    if OUT.is_file():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[d["swe_id"]] = d  # 同 id 后写覆盖先写(重试结果覆盖旧的)
                except Exception:  # noqa: BLE001
                    continue
    # 只把"已成功打分"(有 score)的算作 done;judge_failed/error(score 缺失)重试
    scored = {k for k, v in done.items() if v.get("score") is not None}
    remaining = [t for t in tasks if t["swe_id"] not in scored]
    if args.limit:
        remaining = remaining[: args.limit]
    print(f"\n实跑: {len(remaining)} 待跑 ({len(done)} 已跑), workers={args.workers}", flush=True)
    if not remaining:
        print("无待跑任务", flush=True)
    else:
        fout = open(OUT, "a", encoding="utf-8")
        ok = fail = 0
        t0 = time.time()

        def _work(t):
            return run_one(t, key, args.max_turns, args.slot_timeout)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_work, remaining):
                if r.get("error") and r.get("score") is None:
                    fail += 1
                else:
                    ok += 1
                    done[r["swe_id"]] = r
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n = ok + fail
                if n % 20 == 0 or n == len(remaining):
                    el = time.time() - t0
                    print(f"  {n}/{len(remaining)} ok={ok} run_err={fail} rate={n / el:.2f}/s", flush=True)
        fout.close()

    # ── 汇总分类 ──
    all_rows = list(done.values())
    cls = Counter(classify(r, args.score_threshold) for r in all_rows)
    print(f"\n=== 分类汇总(阈值 {args.score_threshold},共 {len(all_rows)} 已判)===", flush=True)
    for k, v in cls.most_common():
        print(f"  {k}: {v}", flush=True)
    uncompletable = [r["swe_id"] for r in all_rows if classify(r, args.score_threshold).startswith("likely_uncompletable")]
    print(f"\nlikely_uncompletable 共 {len(uncompletable)} 个", flush=True)
    print(f"清单见 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
