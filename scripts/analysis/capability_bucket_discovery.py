#!/usr/bin/env python3
"""按【能力维度】把 ClawEval 任务划分成一套【单层】能力桶（无子桶）。

背景（@孙豪 2026-07-04）：
  - 之前 claweval_bucket_discovery.py 做的是"主桶+子桶"两层、且 LLM 自由归纳出的
    其实是【话题/领域】划分（loan_prepayment / sla_audit ...），不是能力划分。
  - 现有桶体系（见 configs/base.yaml bucket_names = 9-bucket: workflow/ops/qa/finance/
    office/communication/safety/coding/research）是【能力维度】划分，但之前版本的
    9 桶划分"不够细也不够好"。
  - 目标：让 GPT 从任务本身归纳一套【能力维度】的【单层】桶——按"完成任务所需的
    核心能力"分，不按业务领域/话题/具体任务分。数量由 GPT 自定，不锁 9。不要子桶。

排除多模态：默认排除 M 系及 OCR/图像类（本项目纯文本）。

两阶段：
  阶段一（逐任务）：LLM 读任务 → 提炼"完成它所依赖的【核心能力】"（自由描述，不分桶）。
  阶段二（归纳）：把所有能力描述喂给 LLM → 归纳成一套【单层能力桶】，每桶给能力定义。

产物（runs/_analysis/capability_buckets/）：
  ability_notes.jsonl    阶段一逐任务能力提炼
  buckets.json           阶段二能力桶体系 {n_buckets, buckets:[{name,definition,task_count}]}

用法：
  python scripts/analysis/capability_bucket_discovery.py            # 全量纯文本
  python scripts/analysis/capability_bucket_discovery.py --limit 5  # 冒烟
  python scripts/analysis/capability_bucket_discovery.py --from-notes  # 跳过阶段一，只重跑阶段二
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLAWEVAL_TASKS = "/mnt/afs_agents/qinshilong/claw-eval/tasks"
OUT_DIR = ROOT / "runs" / "_analysis" / "capability_buckets"
API_BASE = os.environ.get("DISCOVERY_API_BASE", "https://tokenhub.sensetime.com/v1")

# 多模态：category 层 + 语义层（OCR/图像/视频）双重排除
_MULTIMODAL_CATS = {
    "video_qa", "video_search", "video_edit", "video_ocr", "video_webpage",
    "video_image", "video_chart", "doc_extraction", "doc_search",
    "multimodal", "multimodal_webpage", "webpage_generation", "web_dev",
}
# 语义多模态 task_id（OCR 扫描件 / 图像生成）——纯文本 split 里混入的，显式剔除
_MULTIMODAL_TASK_IDS = {
    "T074_paper_review_injection", "T076_officeqa_defense_spending",
    "T077_officeqa_highest_dept_spending", "T078_officeqa_max_yield_spread",
    "T079_officeqa_zipf_exponent", "T080_officeqa_bond_yield_change",
    "T081_officeqa_cagr_trust_fund", "T082_officeqa_qoq_esf_change",
    "T083_officeqa_mad_excise_tax", "T084_officeqa_geometric_mean_silver",
    "T085_officeqa_army_expenditures", "C04zh_image_processing",
}


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


def _chat(messages: list[dict], model: str, key: str, max_tokens: int = 1024, retries: int = 3) -> str:
    import time

    import httpx

    last = None
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {key}"},
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 -- transient gateway timeout / 502
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


def load_tasks(limit: int) -> list[dict]:
    files = sorted(glob.glob(f"{CLAWEVAL_TASKS}/*/task.yaml"))
    if not files:
        sys.exit(f"ERROR: no task.yaml under {CLAWEVAL_TASKS}")
    tasks = []
    for f in files:
        try:
            d = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tid = d.get("task_id", Path(f).parent.name)
        cat = d.get("category", "")
        # 排除多模态：category 层 + task_id 层 + M 系前缀
        if cat in _MULTIMODAL_CATS or tid in _MULTIMODAL_TASK_IDS or tid.startswith("M"):
            continue
        prompt = d.get("prompt", {})
        text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
        persona = ""
        ua = d.get("user_agent", {})
        if isinstance(ua, dict):
            persona = ua.get("persona", "") or ""
        tasks.append({
            "task_id": tid,
            "task_name": d.get("task_name", ""),
            "prompt": text,
            "persona": persona,
        })
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


# --------------------------------------------------------------------------- #
# 阶段一：逐任务提炼"完成它所需的核心能力"（不分桶、不提话题）                     #
# --------------------------------------------------------------------------- #

_STAGE1_SYSTEM = textwrap.dedent("""\
    你在分析 agent 任务需要的【核心能力】。对给定的一个任务，只回答一件事：
    一个 agent 要【成功完成】这个任务，主要依赖【什么能力】？

    【能力】= agent 做事的本领，例如：多步骤规划与编排、工具/系统操作（读写文件/
    执行命令/调用接口）、多轮交互与澄清追问、结构化规则计算、信息检索、跨来源信息
    综合、书面表达与沟通、审慎与安全判断……

    【严禁】用业务领域/话题作答（如"金融""保险""医疗""房贷""邮件"都是话题，不是能力）。
    也不要给这个任务起分类名，只描述它考验的能力。

    只输出 JSON: {"capabilities": ["能力短语1", "能力短语2"], "primary": "最核心的那一个能力"}
    能力短语用中文，2-6 字，聚焦"本领"而非"领域"。不要 markdown。
""")

_STAGE1_USER = textwrap.dedent("""\
    task_id: {task_id}
    任务: {prompt}
    {persona_block}
    这个任务主要考验 agent 的什么能力？只回 JSON。
""")


def stage1_notes(tasks: list[dict], model: str, key: str) -> list[dict]:
    from tqdm import tqdm

    notes = []
    for t in tqdm(tasks, desc="stage1 能力提炼"):
        persona_block = f"用户画像: {t['persona'][:300]}" if t.get("persona") else ""
        user = _STAGE1_USER.format(
            task_id=t["task_id"], prompt=(t["prompt"] or "")[:1500], persona_block=persona_block,
        )
        try:
            out = _chat([{"role": "system", "content": _STAGE1_SYSTEM},
                         {"role": "user", "content": user}], model, key, max_tokens=300)
            v = _parse_json(out)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[stage1] FAIL {t['task_id']}: {exc}", file=sys.stderr)
            continue
        notes.append({
            "task_id": t["task_id"],
            "capabilities": v.get("capabilities", []),
            "primary": v.get("primary", ""),
        })
    return notes


# --------------------------------------------------------------------------- #
# 阶段二：把能力描述归纳成一套【单层能力桶】                                       #
# --------------------------------------------------------------------------- #

_STAGE2_SYSTEM = textwrap.dedent("""\
    你是持续学习 agent 训练体系的能力分桶架构师。下面是对一批任务逐条提炼的"完成它
    所需核心能力"。请把它们归纳成一套【单层能力桶】——每个桶代表一种 agent 能力维度。

    ## 硬约束
    1. 【按能力划分，不按话题】桶必须是"能力维度"（如：多步规划编排、工具与系统操作、
       多轮交互澄清、结构化计算、信息检索、信息综合、表达沟通、安全审慎判断……）。
       【严禁】出现业务领域/话题桶（金融、保险、邮件、运维……都不允许作为桶名）。
    2. 【单层】只有一层桶，【没有子桶】。
    3. 【互斥】桶之间能力边界清晰不重叠；一个任务应能明确归入某一个能力桶。
    4. 桶数量你自己定合理值（覆盖这批任务的能力谱，通常 6-10 个），要比笼统的分类更贴合，
       但不要碎到一能力一桶。
    5. 每个桶给一句能力定义 + 估算归入的任务数（按输入统计，求和≈任务总数）。

    ## 只输出 JSON（不要 markdown、不要多余文字）
    {"n_buckets": <int>,
     "buckets": [{"name": "英文桶名(能力,如 workflow_orchestration)",
                  "cn": "中文能力名", "definition": "这个能力桶的定义",
                  "task_count": <int>}, ...]}
""")


def stage2_buckets(notes: list[dict], model: str, key: str) -> dict:
    # 喂逐任务能力(primary + capabilities), 让模型看能力谱归纳
    items = [{"primary": n.get("primary", ""), "caps": n.get("capabilities", [])} for n in notes]
    payload = json.dumps(items, ensure_ascii=False)
    user = (
        f"共 {len(notes)} 个任务的能力提炼：\n{payload}\n\n"
        "请归纳成一套【单层能力桶】(按能力不按话题)，只回 JSON。"
    )
    out = _chat([{"role": "system", "content": _STAGE2_SYSTEM},
                 {"role": "user", "content": user}], model, key, max_tokens=4096)
    return _parse_json(out)


def _load_notes(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--from-notes", action="store_true", help="跳过阶段一，从 ability_notes.jsonl 重跑阶段二")
    ap.add_argument("--model", default=os.environ.get("DISCOVERY_MODEL", "claude-opus-4-8"))
    args = ap.parse_args()

    key = _load_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes_path = OUT_DIR / "ability_notes.jsonl"

    if args.from_notes:
        notes = _load_notes(notes_path)
        print(f"[cap] 从 {notes_path} 恢复 {len(notes)} 条能力提炼", flush=True)
    else:
        tasks = load_tasks(args.limit)
        print(f"[cap] 载入 {len(tasks)} 个纯文本任务(已排除多模态), model={args.model}", flush=True)
        notes = stage1_notes(tasks, args.model, key)
        print(f"[cap] 阶段一: {len(notes)}/{len(tasks)} 条能力提炼成功", flush=True)
        if not notes:
            sys.exit("ERROR: 阶段一无结果")
        notes_path.write_text("\n".join(json.dumps(n, ensure_ascii=False) for n in notes) + "\n",
                              encoding="utf-8")
        print(f"[cap] 阶段一已落盘 -> {notes_path}", flush=True)

    buckets = stage2_buckets(notes, args.model, key)
    n = len(buckets.get("buckets", []))
    buckets["n_buckets"] = n
    print(f"\n[cap] 能力桶: {n} 个", flush=True)
    total = 0
    for b in buckets.get("buckets", []):
        tc = b.get("task_count", 0)
        total += tc
        print(f"  {b.get('name'):32s} {b.get('cn', ''):12s} ({tc:3}) {b.get('definition', '')[:50]}")
    print(f"[cap] task_count 求和: {total} (输入 {len(notes)})")

    (OUT_DIR / "buckets.json").write_text(json.dumps(buckets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cap] 落盘 -> {OUT_DIR / 'buckets.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
