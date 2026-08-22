#!/usr/bin/env python3
"""Prompt 写好了 → 人打标几个样本 → 模型照样本批量打标全量

Usage:
  # 1. 生成人工模板给合作者填
  python scripts/data/label_buckets.py --generate-template > buckets_to_label.csv
  # 合作者填完 (bucket, bucket_sub, reasoning 三列) → 另存为 buckets_labeled.csv

  # 2. 合作者填好后，用已打标样本作 few-shot，模型打全量
  python scripts/data/label_buckets.py --few-shot buckets_labeled.csv --write

参考: 脚本调用 tokenhub endpoint 和 TOKENHUB_API_KEY。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKSPECS_DIR = ROOT / "datasources" / "taskspecs"

CANONICAL_BUCKETS = [
    "workflow",
    "ops",
    "qa",
    "finance",
    "office",
    "communication",
    "safety",
    "coding",
    "research",
]

BUCKET_DEFINITIONS = {
    "workflow":      "多步骤工作流编排（拆解目标、串联动作、条件分支与流程协调）",
    "ops":           "系统操作（文件读写、命令行/终端执行、系统运维、资源增删改查）",
    "qa":            "问答检索（查事实、答问题、阅读理解、记忆检索；即查即答，不产出长报告）",
    "finance":       "财务金融（贷款/税务/估值/ROI 计算、采购等按金融业务规则算账）",
    "office":        "办公文档（办公问答、表格/报表处理、办公数据分析）",
    "communication": "沟通表达（邮件撰写/分类、内容创作、润色改写、翻译）",
    "safety":        "安全合规（拒绝不安全请求、漏洞/威胁评估、合规审查、敏感操作把关）",
    "coding":        "代码（编写、审查、调试、修复代码，代码正确性推理）",
    "research":      "研究综合（查多源信息并综合成报告/简报/摘要；区别于 qa 的即查即答）",
}


# ------------------------------------------------------------------
# 1. 读 taskspecs
# ------------------------------------------------------------------

def _load_all_taskspecs() -> list[dict]:
    tasks = []
    for d in sorted(TASKSPECS_DIR.iterdir()):
        if not d.is_dir():
            continue
        f = d / "taskspec.yaml"
        if f.is_file():
            ts = yaml.safe_load(f.read_text(encoding="utf-8"))
            ts["_path"] = str(f)
            tasks.append(ts)
    return tasks


# ------------------------------------------------------------------
# 2. 生成人工模板
# ------------------------------------------------------------------

def cmd_generate_template() -> None:
    tasks = _load_all_taskspecs()
    w = csv.writer(sys.stdout)
    w.writerow(["task_id", "task_family", "seed_query_snippet", "bucket", "bucket_sub", "reasoning"])
    for ts in tasks:
        w.writerow([
            ts["task_id"],
            ts["task_family"],
            (ts.get("seed_query", "") or "")[:120],
            "",  # bucket — 合作者填
            "",  # bucket_sub — 合作者填
            "",  # reasoning — 合作者填
        ])
    print(f"\n# 共 {len(tasks)} 条。合作者请填 bucket / bucket_sub / reasoning 三列。", file=sys.stderr)
    print(f"# 主桶可选: {CANONICAL_BUCKETS}", file=sys.stderr)
    print("# 桶定义:", file=sys.stderr)
    for b, d in BUCKET_DEFINITIONS.items():
        print(f"#   {b:<14} {d}", file=sys.stderr)
    print("# 合作者至少填 3 条即可，留空的条目由模型补标。", file=sys.stderr)
    print("# 填完另存为 buckets_labeled.csv，然后跑: python scripts/data/label_buckets.py --few-shot buckets_labeled.csv --write", file=sys.stderr)


# ------------------------------------------------------------------
# 3. 模型批量打标 (few-shot)
# ------------------------------------------------------------------

_FEW_SHOT_SYSTEM = textwrap.dedent("""\
You are a task classifier for a continual-learning agent training pipeline.
Your job is to classify each task into one of 9 capability buckets.

## Bucket definitions
{bucket_defs}

## Rules
- Pick exactly ONE primary bucket and ONE sub-bucket (different from primary).
- The primary bucket is where the task's trajectory should be stored in the
  replay buffer.  The sub-bucket is a secondary classification for reference.
- Base your decision on the task's seed_query, hidden_goal, task_family, and
  the few-shot examples below.
- Output ONLY a JSON object: {{"task_id": "...", "bucket": "...", "bucket_sub": "...", "reasoning": "..."}}
- No prose, no markdown fences.
""")

_FEW_SHOT_USER_TMPL = textwrap.dedent("""\
## Few-shot labeled examples
{examples}

## Unlabeled task to classify
task_id: {task_id}
task_family: {task_family}
seed_query: {seed_query}
hidden_goal: {hidden_goal}

Classify this task.  Reply with ONLY the JSON object.
""")


def _build_bucket_defs_text() -> str:
    return "\n".join(f"- {b}: {d}" for b, d in BUCKET_DEFINITIONS.items())


def _build_example_text(examples: list[dict]) -> str:
    lines = []
    for i, e in enumerate(examples, 1):
        lines.append(
            f"  [{i}] task_id={e['task_id']}, bucket={e['bucket']}, "
            f"bucket_sub={e['bucket_sub']}, reasoning={e['reasoning']}"
        )
    return "\n".join(lines)


def cmd_few_shot(few_shot_path: str, write: bool, model: str, dry_run: bool) -> None:
    # read few-shot examples
    examples: list[dict] = []
    labeled_ids: set[str] = set()
    with open(few_shot_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("task_id") or "").strip()
            bkt = (row.get("bucket") or "").strip()
            if sid and bkt:
                examples.append({
                    "task_id": sid,
                    "bucket": bkt,
                    "bucket_sub": (row.get("bucket_sub") or "").strip(),
                    "reasoning": (row.get("reasoning") or "").strip(),
                })
                labeled_ids.add(sid)

    if len(examples) < 1:
        sys.exit("ERROR: --few-shot 文件里至少要有 1 条已打标的样本")
    print(f"[label] 已打标样本: {len(examples)} 条 (ids: {sorted(labeled_ids)})", flush=True)

    # read all tasks, separate labeled vs unlabeled
    all_tasks = _load_all_taskspecs()
    unlabeled = [ts for ts in all_tasks if ts["task_id"] not in labeled_ids]
    if not unlabeled:
        print("[label] 所有任务均已有人工打标，无需模型补标。")
        return

    print(f"[label] 待模型补标: {len(unlabeled)} 条", flush=True)

    # tokenhub API
    api_key = os.environ.get("TOKENHUB_API_KEY", "")
    if not api_key:
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("TOKENHUB_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        sys.exit("ERROR: TOKENHUB_API_KEY not set (source scripts/env/load_training_env.sh)")
    base_url = "https://tokenhub.sensetime.com/v1"

    bucket_defs = _build_bucket_defs_text()
    example_text = _build_example_text(examples)

    import httpx
    from tqdm import tqdm

    success = 0
    for ts in tqdm(unlabeled, desc="labeling"):
        user = _FEW_SHOT_USER_TMPL.format(
            examples=example_text,
            task_id=ts["task_id"],
            task_family=ts.get("task_family", "?"),
            seed_query=(ts.get("seed_query", "") or "")[:2000],
            hidden_goal=(ts.get("hidden_goal", "") or "")[:2000],
        )
        messages = [
            {"role": "system", "content": _FEW_SHOT_SYSTEM.format(bucket_defs=bucket_defs)},
            {"role": "user", "content": user},
        ]
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120.0,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            verdict = json.loads(content)
        except Exception as exc:
            print(f"\n[label] FAIL {ts['task_id']}: {exc}", file=sys.stderr)
            continue

        bucket = verdict.get("bucket", "")
        sub = verdict.get("bucket_sub", "")
        reasoning = verdict.get("reasoning", "")
        if bucket not in CANONICAL_BUCKETS:
            print(f"\n[label] WARN {ts['task_id']}: bucket={bucket!r} 不在 9 桶名单，跳过", file=sys.stderr)
            continue

        ts["bucket"] = bucket
        ts["bucket_sub"] = sub if sub in CANONICAL_BUCKETS else ""
        ts["bucket_reasoning"] = reasoning
        success += 1

        if write:
            path = ts.pop("_path")
            yaml_str = yaml.safe_dump(ts, allow_unicode=True, sort_keys=False, width=1000)
            Path(path).write_text(yaml_str, encoding="utf-8")
            ts["_path"] = path  # restore for next iteration

        if not dry_run:
            tqdm.write(f"  {ts['task_id']} -> {bucket} (sub: {sub})")

    # also write back the few-shot examples as yaml fields
    if write:
        for ts in all_tasks:
            if ts["task_id"] in labeled_ids:
                for e in examples:
                    if e["task_id"] == ts["task_id"]:
                        ts["bucket"] = e["bucket"]
                        ts["bucket_sub"] = e["bucket_sub"]
                        ts["bucket_reasoning"] = e["reasoning"]
                        path = ts.pop("_path")
                        yaml_str = yaml.safe_dump(ts, allow_unicode=True, sort_keys=False, width=1000)
                        Path(path).write_text(yaml_str, encoding="utf-8")
                        ts["_path"] = path
                        break

    print(f"\n[label] done: 人工 {len(examples)} + 模型 {success} = {len(examples)+success} 条已打标")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("---")[0])
    ap.add_argument("--generate-template", action="store_true",
                    help="生成人工打标模板(CSV)到 stdout")
    ap.add_argument("--few-shot", metavar="PATH",
                    help="已打标的 CSV，用作 few-shot 样例")
    ap.add_argument("--write", action="store_true",
                    help="将打标结果写回 taskspec.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="调模型但不写 yaml，仅打印结果")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="分类用的模型 (需便宜、稳定；默认 gpt-5-mini)")
    args = ap.parse_args()

    if args.generate_template:
        cmd_generate_template()
    elif args.few_shot:
        cmd_few_shot(args.few_shot, args.write, args.model, args.dry_run)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
