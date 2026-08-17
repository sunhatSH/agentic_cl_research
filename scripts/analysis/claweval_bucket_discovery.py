#!/usr/bin/env python3
"""让大模型读 ClawEval 全集任务，自由归纳该划分多少主桶 / 子桶。

背景（@孙豪 2026-07-04）：
  - 现有 7 主桶 / 25 子桶（SUB_BUCKETS）是拍的；想让 LLM 从 ClawEval 数据里
    **自由归纳**该分多少桶（不套用现有分类）。
  - 归纳结果若 ≠ 现有 7/25，据此重定桶（buffer / 训练 / 评测跟着改）。

数据源：ClawEval 官方全集 300 task.yaml
  /mnt/afs_agents/qinshilong/claw-eval/tasks/*/task.yaml
  每个自带 category/difficulty/tags/prompt——但 **category 不喂给 LLM**
  （否则 LLM 照抄，不是真归纳；category 仅在末尾做事后对照）。
  split：T 161 General + M 101 Multimodal + C 38 Multi-turn。默认排除多模态
  （本项目只用纯文本，见 CLAUDE.md），可用 --include-multimodal 纳入。

方法：两阶段自由归纳
  阶段一（逐任务提议）：LLM 读每个任务的 prompt/persona → 自由提议
    {main_bucket, sub_bucket, reasoning}（不给固定选项，自由命名）。
  阶段二（聚合归纳）：把阶段一所有提议喂给 LLM → 合并近义 → 输出最终
    {n_main, n_sub, taxonomy:[{main, subs:[...], definition, task_count}]}。

走 tokenhub（本地开发机可达商汤内网；沙箱侧也直连 tokenhub）。key 取 TOKENHUB_API_KEY。
可用 DISCOVERY_API_BASE / DISCOVERY_API_KEY / DISCOVERY_MODEL 覆盖端点/密钥/模型。

用法：
  source scripts/env/load_training_env.sh
  # 全量纯文本（默认，排除多模态）：
  python scripts/analysis/claweval_bucket_discovery.py --write
  # 小批验证：
  python scripts/analysis/claweval_bucket_discovery.py --limit 30
  # 纳入多模态：
  python scripts/analysis/claweval_bucket_discovery.py --include-multimodal --write
  # 换归纳模型：
  python scripts/analysis/claweval_bucket_discovery.py --model openai/gpt-5.5

产物（--write，落 runs/_analysis/bucket_discovery/）：
  proposals.jsonl        阶段一逐任务提议
  taxonomy.json          阶段二归纳出的桶体系（该分几主桶几子桶）
  compare_vs_builtin.json  归纳结果 vs 数据自带 category 的对照
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLAWEVAL_TASKS = "/mnt/afs_agents/qinshilong/claw-eval/tasks"
OUT_DIR = ROOT / "runs" / "_analysis" / "bucket_discovery"

DISCOVERY_BASE = os.environ.get("DISCOVERY_API_BASE", "https://tokenhub.sensetime.com/v1")

# Multimodal categories (M-series) — excluded by default (project uses text-only 195).
_MULTIMODAL_CATS = {
    "video_qa", "video_search", "video_edit", "video_ocr", "video_webpage",
    "video_image", "video_chart", "doc_extraction", "doc_search",
    "multimodal", "multimodal_webpage", "webpage_generation", "web_dev",
}


def _load_key() -> str:
    # 归纳在开发机本地跑(不经沙箱),用 tokenhub(商汤内网可达);沙箱侧也走 tokenhub。
    # 可用 DISCOVERY_API_KEY 覆盖 key 变量名/值。
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
        sys.exit("ERROR: TOKENHUB_API_KEY not set (see .env / source scripts/env/load_training_env.sh)")
    return key


def _chat(messages: list[dict], model: str, key: str, max_tokens: int = 1024, retries: int = 3) -> str:
    import time

    import httpx

    last = None
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{DISCOVERY_BASE}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {key}"},
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 -- transient gateway timeout / 502
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # 2s,4s backoff
    raise last


def _parse_json(text: str):
    """Tolerant JSON extraction (strip markdown fences / prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                text = text[: i + 1]
                break
    return json.loads(text)


def load_tasks(include_multimodal: bool) -> list[dict]:
    """Load ClawEval 300 task.yaml. Returns list of task dicts (built-in category
    kept ONLY for post-hoc compare -- NOT shown to the LLM)."""
    files = sorted(glob.glob(f"{CLAWEVAL_TASKS}/*/task.yaml"))
    if not files:
        sys.exit(f"ERROR: no task.yaml under {CLAWEVAL_TASKS}")
    tasks = []
    for f in files:
        try:
            d = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        cat = d.get("category", "")
        if not include_multimodal and cat in _MULTIMODAL_CATS:
            continue
        prompt = d.get("prompt", {})
        text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
        persona = ""
        ua = d.get("user_agent", {})
        if isinstance(ua, dict):
            persona = ua.get("persona", "") or ""
        tasks.append({
            "task_id": d.get("task_id", Path(f).parent.name),
            "task_name": d.get("task_name", ""),
            "prompt": text,
            "persona": persona,
            "_builtin_category": cat,     # 事后对照用，不喂 LLM
            "_difficulty": d.get("difficulty", ""),
        })
    return tasks


# --------------------------------------------------------------------------- #
# 阶段一：逐任务自由提议（不给固定桶选项）                                       #
# --------------------------------------------------------------------------- #

_STAGE1_SYSTEM = textwrap.dedent("""\
    你是一个持续学习 agent 训练项目的任务分析专家。你要为一批 agent 任务设计
    **能力分桶**体系。现在处于"自由归纳"阶段：**不要**套用任何已有分类，完全
    根据任务本身，为它提议一个合适的【主桶】（粗粒度能力域）和【子桶】（细粒度
    能力）名称。

    要求：
    - 主桶 = 这条任务考察的核心能力域（如"多步编排""系统操作""结构化业务计算"等），
      你自己命名，简短英文小写下划线。
    - 子桶 = 主桶下更细的能力（如主桶 system_ops 下的子桶 file_edit）。
    - 只输出 JSON：{"main_bucket": "...", "sub_bucket": "...", "reasoning": "一句话"}
    - 不要 markdown、不要多余文字。
""")

_STAGE1_USER = textwrap.dedent("""\
    任务信息：
    task_name: {task_name}
    用户请求: {prompt}
    {persona_block}
    请为这条任务提议主桶和子桶（自由命名，不套用任何已有分类）。只回 JSON。
""")


def stage1_propose(tasks: list[dict], model: str, key: str) -> list[dict]:
    from tqdm import tqdm

    proposals = []
    for t in tqdm(tasks, desc="stage1 propose"):
        persona_block = f"用户画像: {t['persona'][:600]}" if t["persona"] else ""
        user = _STAGE1_USER.format(
            task_name=t["task_name"],
            prompt=(t["prompt"] or "")[:1500],
            persona_block=persona_block,
        )
        try:
            out = _chat(
                [{"role": "system", "content": _STAGE1_SYSTEM},
                 {"role": "user", "content": user}],
                model, key, max_tokens=400,
            )
            v = _parse_json(out)
        except Exception as exc:
            print(f"\n[stage1] FAIL {t['task_id']}: {exc}", file=sys.stderr)
            continue
        proposals.append({
            "task_id": t["task_id"],
            "main_bucket": v.get("main_bucket", ""),
            "sub_bucket": v.get("sub_bucket", ""),
            "reasoning": v.get("reasoning", ""),
            "_builtin_category": t["_builtin_category"],
        })
    return proposals


# --------------------------------------------------------------------------- #
# 阶段二：聚合归纳（合并近义，输出最终该分几桶）                                 #
# --------------------------------------------------------------------------- #

_STAGE2_SYSTEM = textwrap.dedent("""\
    你是能力分桶体系的归纳专家。下面是对一批 agent 任务逐条提议的（主桶, 子桶）
    标签，命名不统一、有近义重复。请把它们**归纳/合并**成一套干净的分桶体系：
    - 合并语义相近的主桶、子桶（如 system_ops / sysops / system_operations 归一）。
    - 给出最终该分【几个主桶】、【几个子桶】。
    - 每个主桶给一句定义，列出它包含的子桶，以及归到该桶的任务数（按输入统计）。
    - 只输出 JSON：
      {"n_main": <int>, "n_sub": <int>,
       "taxonomy": [{"main": "...", "definition": "...",
                     "subs": ["...", ...], "task_count": <int>}, ...]}
    - 不要 markdown、不要多余文字。
""")


def stage2_aggregate(proposals: list[dict], model: str, key: str) -> dict:
    # 压缩 payload：把 193 条逐条提议去重成"唯一(主桶,子桶)对 + 出现次数"，
    # 避免一次性发几百条重复文本撑爆 tokenhub 网关(曾致 502 Bad Gateway)。
    pair_counts = Counter((p["main_bucket"], p["sub_bucket"]) for p in proposals)
    pairs = [{"main": m, "sub": s, "count": c} for (m, s), c in pair_counts.most_common()]
    payload = json.dumps(pairs, ensure_ascii=False)
    user = (
        f"共 {len(proposals)} 条任务，去重后 {len(pairs)} 个唯一 (主桶,子桶,出现次数) 提议：\n"
        f"{payload}\n\n请归纳成最终分桶体系(task_count 用 count 求和)，只回 JSON。"
    )
    out = _chat(
        [{"role": "system", "content": _STAGE2_SYSTEM},
         {"role": "user", "content": user}],
        model, key, max_tokens=4096,
    )
    return _parse_json(out)


def _load_proposals(path: Path) -> list[dict]:
    """Read a previously-saved proposals.jsonl (skip stage1 re-run)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def compare_vs_builtin(proposals: list[dict], taxonomy: dict) -> dict:
    """LLM 归纳 vs 数据自带 category 的粗对照（供 @孙豪 判断）。"""
    builtin = Counter(p["_builtin_category"] for p in proposals if p["_builtin_category"])
    llm_main = Counter(p["main_bucket"] for p in proposals if p["main_bucket"])
    return {
        "builtin_category_count": len(builtin),
        "builtin_categories": dict(builtin.most_common()),
        "llm_main_bucket_count_raw": len(llm_main),  # 归纳前的原始主桶名数
        "llm_main_buckets_raw": dict(llm_main.most_common()),
        "llm_taxonomy_n_main": taxonomy.get("n_main"),
        "llm_taxonomy_n_sub": taxonomy.get("n_sub"),
        "note": "builtin 是数据自带 category(未喂 LLM)；llm_* 是自由归纳结果。对照看归纳是否合理。",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务(0=全量)")
    ap.add_argument("--include-multimodal", action="store_true", help="纳入多模态 M-series(默认排除)")
    ap.add_argument("--model", default=os.environ.get("DISCOVERY_MODEL", "claude-opus-4-8-thinking"),
                    help="归纳模型(tokenhub id，默认 claude-opus-4-8-thinking)")
    ap.add_argument("--write", action="store_true", help="产物落 runs/_analysis/bucket_discovery/")
    ap.add_argument("--from-proposals", action="store_true",
                    help="跳过阶段一，从已存 proposals.jsonl 直接做阶段二归纳(stage2 失败重跑用)")
    args = ap.parse_args()

    key = _load_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 阶段一：从已存 proposals 恢复(--from-proposals)或重跑；跑完【立即落盘】，
    # 这样 stage2 若失败(曾 502)不用重跑 stage1。
    prop_path = OUT_DIR / "proposals.jsonl"
    if args.from_proposals:
        proposals = _load_proposals(prop_path)
        print(f"[discovery] 从 {prop_path} 恢复 {len(proposals)} 条提议(跳过阶段一)", flush=True)
    else:
        tasks = load_tasks(args.include_multimodal)
        if args.limit > 0:
            tasks = tasks[: args.limit]
        print(f"[discovery] 载入 {len(tasks)} 个任务 "
              f"(multimodal={'纳入' if args.include_multimodal else '排除'}, model={args.model})", flush=True)
        proposals = stage1_propose(tasks, args.model, key)
        print(f"[discovery] 阶段一: {len(proposals)}/{len(tasks)} 条提议成功", flush=True)
        if not proposals:
            sys.exit("ERROR: 阶段一无成功提议")
        # 立即落盘 stage1 结果(不等 stage2)
        prop_path.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals) + "\n", encoding="utf-8")
        print(f"[discovery] 阶段一已落盘 -> {prop_path}", flush=True)

    taxonomy = stage2_aggregate(proposals, args.model, key)
    n_main, n_sub = taxonomy.get("n_main"), taxonomy.get("n_sub")
    print(f"[discovery] 阶段二归纳: 主桶 {n_main} 个 / 子桶 {n_sub} 个", flush=True)
    for tb in taxonomy.get("taxonomy", []):
        print(f"    - {tb.get('main')} ({tb.get('task_count')}): subs={tb.get('subs')}", flush=True)

    compare = compare_vs_builtin(proposals, taxonomy)
    print(f"\n[discovery] 对照: 数据自带 {compare['builtin_category_count']} category "
          f"vs LLM 归纳 {n_main} 主桶/{n_sub} 子桶", flush=True)

    # taxonomy / compare 总是落盘(proposals 已在上面存过)
    (OUT_DIR / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "compare_vs_builtin.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[discovery] 产物 -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
