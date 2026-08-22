#!/usr/bin/env python3
"""Calibrate reward-judge candidates against ClawEval human rubrics.

Runs each candidate judge over a human-labeled set, computes judge↔human
agreement (eval/judge_agreement), and recommends the smallest judge that clears
the agreement bar. This is how the FINAL reward judge is chosen
(trainer/model_reward.py); see doc/sandbox/Sandbox_Agent架构.md §7b.

Inputs
------
--labeled  JSONL, one human-graded trajectory per line:
    {task, trajectory, rubric, bucket, human:{completion,safety,robustness}, pass?}
    (Built from ClawEval's human rubrics -- BLOCKED on the eval manifest; until
     then run on a small hand-labeled pilot set with the SAME schema.)

--judges   JSON config of candidates, e.g.:
    [
      {"name": "qwen2.5-14b", "size_b": 14, "base_url": "http://127.0.0.1:8101/v1",
       "model": "reward-judge", "api_key": "sk-local"},
      {"name": "qwen2.5-32b", "size_b": 32, "base_url": "http://127.0.0.1:8100/v1",
       "model": "reward-judge"}
    ]
  Launch each with scripts/serve/serve_reward_model.sh on its own port/GPUs first.

Output: a per-judge agreement report + the recommended judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # 6 库包在 src/

from eval.judge_agreement import evaluate_judge, rank_judges  # noqa: E402
from trainer.model_reward import OpenAIJudgeClient  # noqa: E402


def load_labeled(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_judge_over(samples: list[dict[str, Any]], client: Any) -> list[dict[str, float]]:
    preds = []
    for s in samples:
        try:
            v = client.score(
                task=s.get("task", ""),
                trajectory=s.get("trajectory", ""),
                rubric=s.get("rubric", ""),
                data_source=s.get("data_source", "agentic_cl"),
            )
        except Exception as exc:  # noqa: BLE001 -- record and continue
            print(f"  [warn] judge failed on a sample: {exc}", file=sys.stderr)
            v = {"completion": 0.0, "safety": 0.0, "robustness": 0.0}
        preds.append(dict(v))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", required=True, help="JSONL of human-graded trajectories")
    ap.add_argument("--judges", required=True, help="JSON config of candidate judges")
    ap.add_argument("--pass-threshold", type=float, default=0.5)
    ap.add_argument("--min-kappa", type=float, default=0.6)
    ap.add_argument("--out", default=None, help="optional path to dump full reports JSON")
    args = ap.parse_args()

    samples = load_labeled(Path(args.labeled))
    judges = json.loads(Path(args.judges).read_text(encoding="utf-8"))
    print(f"[calibrate] {len(samples)} labeled samples, {len(judges)} candidate judges")

    reports: dict[str, Any] = {}
    sizes: dict[str, float] = {}
    for cfg in judges:
        name = cfg["name"]
        sizes[name] = float(cfg.get("size_b", float("inf")))
        client = OpenAIJudgeClient(
            base_url=cfg["base_url"], model=cfg["model"], api_key=cfg.get("api_key", "sk-local")
        )
        print(f"[calibrate] scoring with {name} ({cfg['base_url']})...")
        preds = run_judge_over(samples, client)
        reports[name] = evaluate_judge(samples, preds, pass_threshold=args.pass_threshold)

    ranking = rank_judges(reports, sizes_b=sizes, min_kappa=args.min_kappa)

    print("\n=== judge agreement (vs human rubric) ===")
    for row in ranking["ranking"]:
        name = row["judge"]
        rep = reports[name]
        print(
            f"  {name:16s} kappa={rep['pass'].get('kappa', 0):.3f} "
            f"acc={rep['pass']['accuracy']:.3f} f1={rep['pass']['f1']:.3f} "
            f"score_mae={rep['score_mae']:.3f}"
        )
    rec = ranking["recommended"]
    flag = "OK" if ranking["cleared_threshold"] else f"NONE cleared kappa>={args.min_kappa}"
    print(f"\n[calibrate] recommended judge: {rec}  ({flag})")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"reports": reports, "ranking": ranking}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[calibrate] full reports -> {args.out}")


if __name__ == "__main__":
    main()
