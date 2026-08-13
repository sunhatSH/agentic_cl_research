#!/usr/bin/env python3
"""生成 ClawEval 评测 manifest：195 纯文本任务，按官方 category 映射到 9 能力桶。

与训练打标【同一套】桶映射(runs/_analysis/capability_buckets/buckets.json 的
official_categories)，保证训练桶与评测桶口径一致。

排除多模态(category + task_id + M 前缀)。多轮(C 系 user_agent)按首轮 prompt
用 LLM 归桶（不归桶则为空，后续人工补）。

Input  : ClawEval tasks/*/task.yaml
Output : eval/claweval_manifest.json  —— JSON list, 每条:
           {task_id, split, modality:"text", bucket, category, prompt}

用法: python scripts/data/build_eval_manifest.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLAWEVAL_TASKS = "/mnt/afs_agents/qinshilong/claw-eval/tasks"
BUCKETS_JSON = ROOT / "runs" / "_analysis" / "capability_buckets" / "buckets.json"
OUT = ROOT / "eval" / "claweval_manifest.json"

_MULTIMODAL_CATS = {
    "video_qa", "video_search", "video_edit", "video_ocr", "video_webpage",
    "video_image", "video_chart", "doc_extraction", "doc_search",
    "multimodal", "multimodal_webpage", "webpage_generation", "web_dev",
}
_MULTIMODAL_TASK_IDS = {
    "T074_paper_review_injection", "T076_officeqa_defense_spending",
    "T077_officeqa_highest_dept_spending", "T078_officeqa_max_yield_spread",
    "T079_officeqa_zipf_exponent", "T080_officeqa_bond_yield_change",
    "T081_officeqa_cagr_trust_fund", "T082_officeqa_qoq_esf_change",
    "T083_officeqa_mad_excise_tax", "T084_officeqa_geometric_mean_silver",
    "T085_officeqa_army_expenditures", "C04zh_image_processing",
}


def _load_category_to_bucket() -> dict[str, str]:
    """从 buckets.json 反建 官方 category -> 桶名 的映射。"""
    d = json.loads(BUCKETS_JSON.read_text(encoding="utf-8"))
    cat2bucket = {}
    for b in d["buckets"]:
        for cat in b.get("official_categories", []):
            cat2bucket[cat] = b["name"]
    return cat2bucket


def _classify_user_agent(prompt: str, valid_buckets: set[str]) -> str:
    """LLM 按首轮 prompt 内容归桶（user_agent 多轮任务）。

    复用 data_pipeline/classify.py 的 classify_query。
    """
    from data_pipeline.classify import classify_query, make_default_client

    client = make_default_client()
    verdict = classify_query(prompt, client)
    bucket = verdict.get("bucket", "")
    if bucket in valid_buckets:
        print(f"  [classify] user_agent → {bucket}: {prompt[:60]}...", flush=True)
        return bucket
    print(f"  [classify] user_agent → unknown (model={verdict.get('model')}, "
          f"rationale={verdict.get('rationale','')[:80]})", flush=True)
    return ""


def main() -> int:
    cat2bucket = _load_category_to_bucket()
    valid_buckets = set(cat2bucket.values())
    print(f"[manifest] category→bucket 映射 {len(cat2bucket)} 条")

    records = []
    skipped_mm = 0
    unmapped = []
    for f in sorted(glob.glob(f"{CLAWEVAL_TASKS}/*/task.yaml")):
        try:
            d = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tid = d.get("task_id", Path(f).parent.name)
        cat = d.get("category", "")
        # 排除多模态
        if cat in _MULTIMODAL_CATS or tid in _MULTIMODAL_TASK_IDS or tid.startswith("M"):
            skipped_mm += 1
            continue
        split = "Multi-turn" if tid.startswith("C") else "General"
        prompt = d.get("prompt", {})
        text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)

        # 1) 静态映射（38 个 ClawEval category → 9 桶）
        bucket = cat2bucket.get(cat)
        # 2) user_agent 不在静态表 → LLM 按首轮 prompt 归桶
        if bucket is None and cat == "user_agent":
            bucket = _classify_user_agent(text, valid_buckets)
        # 3) 仍然空的 → 记录并留空
        if not bucket:
            unmapped.append((tid, cat))
            bucket = ""

        records.append({
            "task_id": tid,
            "split": split,
            "modality": "text",
            "bucket": bucket,
            "category": cat,
            "prompt": text,
        })

    print(f"[manifest] 收录 {len(records)} 条 | 排除多模态 {skipped_mm} | "
          f"未映射 {len(unmapped)} 条")
    if unmapped:
        print(f"[manifest] 未映射(需人工补): {[t for t, _ in unmapped]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] 落盘 -> {OUT}")

    from collections import Counter
    c = Counter(r["bucket"] or "(未映射)" for r in records)
    for b, n in c.most_common():
        print(f"    {b:16s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
