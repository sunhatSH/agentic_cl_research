#!/usr/bin/env python3
"""Phase 汇总：扫一个 phase 下所有实验的评测，产出 _phase_summary.json 供手动选参。

见 doc/eval/训练与评测总思路_产物结构.md。设计决定（2026-07-04 @孙豪）：
  - 训练起点：所有实验均从 π₀（初始 27B），保 ablation 可比 → 评测统一跟 baseline 比。
  - 每 phase 跑完，自动汇总本 phase 所有实验的每桶得分 + 跟 baseline 的 delta；
    **不自动选 top-2** —— selected 留空，@孙豪 看了每桶表现后手动填。
  - 不仅汇总：为每实验每桶保留 per_bucket/*.jsonl 的下钻路径索引。

用法：
    python scripts/analysis/phase_summary.py --phase-dir runs/phase2
    python scripts/analysis/phase_summary.py --phase-dir runs/phase2 --baseline-dir runs/baseline

产出：runs/phaseN/_phase_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_scores(exp_dir: Path) -> dict | None:
    """Read one experiment's eval/scores.json (总账), or None if not evaluated yet."""
    f = exp_dir / "eval" / "scores.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _bucket_detail_index(exp_dir: Path) -> dict[str, str]:
    """Map bucket -> relative path of its per_bucket/*.jsonl (细则下钻入口)."""
    pbd = exp_dir / "eval" / "per_bucket"
    if not pbd.is_dir():
        return {}
    return {p.stem: str(p) for p in sorted(pbd.glob("*.jsonl"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase-dir", required=True, help="e.g. runs/phase2")
    ap.add_argument(
        "--baseline-dir",
        default="runs/baseline",
        help="baseline 实验目录（无 CL 的 27B），用于跨实验统一 delta 参照",
    )
    args = ap.parse_args()

    phase_dir = Path(args.phase_dir)
    if not phase_dir.is_dir():
        print(f"[phase_summary] not a dir: {phase_dir}")
        return 2

    # baseline 每桶 reward（统一参照；缺失则 delta 留空）
    base_scores = _load_scores(Path(args.baseline_dir))
    base_bucket = {}
    if base_scores:
        base_bucket = {b: row.get("reward") for b, row in base_scores.get("per_bucket", {}).items()}

    experiments: dict[str, dict] = {}
    # 每实验一个子目录（跳过 _* 汇总文件、.gitkeep）
    for exp_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        scores = _load_scores(exp_dir)
        if scores is None:
            experiments[exp_dir.name] = {"status": "not_evaluated"}
            continue
        per_bucket = scores.get("per_bucket", {})
        # 每桶：本实验 reward + baseline reward + delta（都看，不只 cl_score）
        bucket_view = {}
        for bucket, row in per_bucket.items():
            b_ref = base_bucket.get(bucket)
            bucket_view[bucket] = {
                "reward": row.get("reward"),
                "pass_rate": row.get("pass_rate"),
                "n_tasks": row.get("n_tasks"),
                "baseline_reward": b_ref,
                "delta_vs_baseline": (row.get("reward") - b_ref) if b_ref is not None else None,
                "detail": str(exp_dir / "eval" / "per_bucket" / f"{bucket}.jsonl"),  # 细则下钻
            }
        experiments[exp_dir.name] = {
            "status": "evaluated",
            "summary": scores.get("summary", {}),  # cl_score / forgetting / new_task_perf
            "per_bucket": bucket_view,
            "detail_index": _bucket_detail_index(exp_dir),
            "config_snapshot": str(exp_dir / "config.snapshot.yaml"),
        }

    out = {
        "phase": phase_dir.name,
        "baseline_dir": args.baseline_dir,
        "delta_reference": "baseline (无 CL 的 27B, 同从 π₀ 训); 所有实验统一跟它比",
        "note": "训练起点均从 π₀ 保 ablation 可比; '合起来不优'由 Phase 4 组合验证兜底",
        "experiments": experiments,
        # ↓ @孙豪 手动填：看完每桶表现后选本 phase 的 top-2 传下一 phase
        "selected": None,
        "selected_rationale": "",
    }
    out_path = phase_dir / "_phase_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_eval = sum(1 for e in experiments.values() if e.get("status") == "evaluated")
    print(f"[phase_summary] {phase_dir.name}: {n_eval}/{len(experiments)} evaluated -> {out_path}")
    print("[phase_summary] selected=null 待手动填 (看 per_bucket delta_vs_baseline 选 top-2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
