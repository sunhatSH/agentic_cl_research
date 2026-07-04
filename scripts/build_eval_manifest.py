#!/usr/bin/env python3
"""生成 ClawEval 评测 manifest：195 纯文本任务，按官方 category 映射到 9 能力桶。

与训练打标【同一套】桶映射(runs/_analysis/capability_buckets/buckets.json 的
official_categories)，保证训练桶与评测桶口径一致。

排除多模态(category + task_id + M 前缀)。多轮(C 系)按官方 category 归桶。

Input  : ClawEval tasks/*/task.yaml
Output : eval/claweval_manifest.json  —— JSON list, 每条:
           {task_id, split, modality:"text", bucket, category, prompt}

用法: python scripts/build_eval_manifest.py
"""

from __future__ import annotations

import glob
import json
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
# 语义多模态(OCR 扫描件/图像生成)—— 与训练打标一致地剔除
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


def main() -> int:
    cat2bucket = _load_category_to_bucket()
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
        # split: C=Multi-turn, T=General
        split = "Multi-turn" if tid.startswith("C") else "General"
        # user_agent(多轮) category 不在映射表 -> 需按首轮归桶(暂标 category, 由 LLM 补)
        bucket = cat2bucket.get(cat)
        if bucket is None:
            unmapped.append((tid, cat))
            bucket = ""  # 留空, 下面单独处理
        prompt = d.get("prompt", {})
        text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
        records.append({
            "task_id": tid,
            "split": split,
            "modality": "text",
            "bucket": bucket,
            "category": cat,
            "prompt": text,
        })

    print(f"[manifest] 收录 {len(records)} 条 | 排除多模态 {skipped_mm} | "
          f"category 未映射(多轮 user_agent) {len(unmapped)} 条")
    if unmapped:
        print(f"[manifest] 未映射(需按首轮归桶): {[t for t, _ in unmapped]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] 落盘 -> {OUT}")

    # 桶分布
    from collections import Counter
    c = Counter(r["bucket"] or "(未映射)" for r in records)
    for b, n in c.most_common():
        print(f"    {b:16s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
