"""Warm-start the 7-bucket replay buffer from collected rollout JSONL.

Reads cold-collection output (data/mock/rollouts/{actor}/rollouts_*.jsonl, each
line = one multi-turn session with `trajectories`) and ingests every trajectory
into the 7-bucket BucketReplayBuffer, then dumps a single sqlite snapshot for the
trainer to preload (warm-start, anti-forgetting cold start).

Classification (孙豪 2026-06-13: "得分类、填进去、后续再淘汰"):
  bucket = trajectory's own <task_domain> tag (parse_domain)
        -> else bucket_hint(messages) keyword vote (convert_dataset)
        -> else FALLBACK_BUCKET (do NOT drop; eviction handles it later)

Cold data has no reward/advantage, so buffer priority falls back to its default
(the buffer computes priority from available signals; missing reward is fine).

Usage:
    python scripts/warmup_buffer.py \
        --in-dir data/mock/rollouts \
        --out data/mock/buffer_dumps/warmup.sqlite
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replay_buffer.bucket import BucketReplayBuffer
from scripts.convert_dataset import bucket_hint
from trainer.domain_tagging import DEFAULT_BUCKETS, parse_domain

VALID_BUCKETS = tuple(DEFAULT_BUCKETS)  # Workflow/SysOps/Dialogue/Finance/Communication/Knowledge/OfficeQA
FALLBACK_BUCKET = "Knowledge"  # least-specific catch-all; evicted later if low value


def classify(traj: dict, session_msgs: list) -> str:
    """bucket: own <task_domain> tag -> keyword hint -> fallback (never drop)."""
    # 1) trajectory's own emitted tag
    b = traj.get("bucket")
    if b in VALID_BUCKETS:
        return b
    # 2) keyword vote over the trajectory's messages (and session seed)
    msgs = traj.get("messages") or []
    hint = bucket_hint(msgs) or bucket_hint(session_msgs)
    if hint in VALID_BUCKETS:
        return hint
    # 3) fallback (cold-start: keep, evict later)
    return FALLBACK_BUCKET


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-start 7-bucket buffer from rollout JSONL.")
    ap.add_argument("--in-dir", default="data/mock/rollouts", help="dir with {actor}/rollouts_*.jsonl")
    ap.add_argument("--out", default="data/mock/buffer_dumps/warmup.sqlite")
    ap.add_argument("--total-capacity", type=int, default=25000)
    ap.add_argument("--skip-empty", action="store_true", default=True,
                    help="skip trajectories whose assistant content is all empty (dead-vllm garbage)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.in_dir) / "*" / "rollouts_*.jsonl")))
    if not files:
        print(f"[warmup] no rollout files under {args.in_dir}", flush=True)
        sys.exit(1)
    print(f"[warmup] {len(files)} files: {[Path(f).name for f in files]}", flush=True)

    buffer = BucketReplayBuffer(total_capacity=args.total_capacity)

    sessions = trajs = added = empty = 0
    per_bucket: dict[str, int] = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sessions += 1
            seed_msgs = [{"role": "user", "content": row.get("seed_query", "")}]
            for t in row.get("trajectories", []):
                trajs += 1
                msgs = t.get("messages") or []
                # skip dead-vllm empty garbage (assistant all blank)
                if args.skip_empty:
                    asst = [m.get("content") for m in msgs if m.get("role") == "assistant"]
                    if not any((c or "").strip() for c in asst):
                        empty += 1
                        continue
                bucket = classify(t, seed_msgs)
                payload = {
                    "messages": msgs,
                    "response_token_ids": t.get("response_token_ids", []),
                    "response_mask": t.get("response_mask", []),
                }
                # unique tid per trajectory: cold rollouts reuse ids like "q0-s0"
                # across sessions; without a unique id store.put overwrites them.
                meta = {
                    "trajectory_id": f"warm-{trajs}-{t.get('trajectory_id','')}",
                    "reward": None,  # cold data: unscored
                    "original_logprobs": [],
                    "success_rate": None,
                    "num_turns": t.get("num_turns"),
                    "warmup": True,
                }
                buffer.add_trajectory(payload, bucket, meta)
                added += 1
                per_bucket[bucket] = per_bucket.get(bucket, 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer.dump(out)

    print(f"[warmup] sessions={sessions} trajs={trajs} added={added} "
          f"empty_skipped={empty}", flush=True)
    print(f"[warmup] per-bucket: {per_bucket}", flush=True)
    print(f"[warmup] buffer.stats: { {k: v for k, v in buffer.stats().items() if k in ('total_size','per_bucket')} }", flush=True)
    print(f"[warmup] dumped -> {out}", flush=True)


if __name__ == "__main__":
    main()
