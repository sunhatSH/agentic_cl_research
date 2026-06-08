"""Evaluation entry: run a checkpoint over the 195-task text subset of ClawEval.

Outputs:
    - per-task pass/fail with safety / completion / robustness sub-scores
    - aggregated New Task Perf, Old Task Forgetting, CL Score
    - dump JSON report to eval/results/<run_id>/

Usage:
    python -m eval.run_eval --ckpt <path> --baseline-ckpt <path>
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, default=None,
                        help="Reference ckpt for forgetting computation.")
    parser.add_argument("--output-dir", type=str, default="eval/results")
    parser.add_argument("--num-runs", type=int, default=3, help="Pass^N standard.")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
