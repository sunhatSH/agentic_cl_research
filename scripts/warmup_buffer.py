"""Warm-start the 9-bucket replay buffer from collected rollout JSONL.

Reads cold-collection output (data/mock/rollouts/{actor}/rollouts_*.jsonl, each
line = one multi-turn session with `trajectories`) and ingests every trajectory
into the 9-bucket BucketReplayBuffer, then dumps a single sqlite snapshot for the
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
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replay_buffer.bucket import BucketReplayBuffer
from scripts.convert_dataset import bucket_hint
from trainer.domain_tagging import DEFAULT_BUCKETS

VALID_BUCKETS = tuple(DEFAULT_BUCKETS)  # 9 buckets from configs/base.yaml via trainer/domain_tagging
FALLBACK_BUCKET = "qa"  # least-specific catch-all; evicted later if low value

# Cold-start data-source tags (Phase 0, doc/Plan_冷启动数据来源消融.md).
SRC_27B = "pi0_27b"  # on-policy: the training policy itself (Qwen3.6-27B)
SRC_GPT5 = "gpt5"  # off-policy: the stronger model (openai/gpt-5)


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


def _source_tag(traj: dict, session: dict, file_path: str) -> str:
    """Resolve the cold-start data-source tag for one trajectory.

    Priority: trajectory `policy` field -> session `policy` field ->
    infer from the `{actor}` sub-dir in the file path (backward-compat with
    rollouts collected before the tag existed).
    """
    tag = traj.get("policy") or session.get("policy")
    if tag in (SRC_27B, SRC_GPT5):
        return tag
    # infer from path: .../rollouts/local/... -> 27B ; .../remote/... -> gpt5
    parts = Path(file_path).parts
    if "remote" in parts:
        return SRC_GPT5
    if "local" in parts:
        return SRC_27B
    return SRC_27B  # conservative default: treat unknown as on-policy


def _mix_by_ratio(
    grouped: dict[str, dict[str, list]], ratio_27b: float, rng: random.Random
) -> tuple[dict[str, list], dict[str, dict[str, int]]]:
    """Per-bucket sample of the two sources to hit the 27B:gpt5 ratio.

    grouped[bucket][src] = list of (payload, meta) candidates. For each bucket we
    keep as many trajectories as possible while matching ratio_27b, limited by
    whichever source is scarcer (no upsampling / duplication). Returns the mixed
    per-bucket lists plus a manifest of realized per-source counts.
    """
    mixed: dict[str, list] = {}
    manifest: dict[str, dict[str, int]] = {}
    for bucket, by_src in grouped.items():
        c27 = list(by_src.get(SRC_27B, []))
        cg5 = list(by_src.get(SRC_GPT5, []))
        rng.shuffle(c27)
        rng.shuffle(cg5)
        n27_avail, ng5_avail = len(c27), len(cg5)
        # Largest total N s.t. round(N*ratio) <= n27_avail and round(N*(1-ratio)) <= ng5_avail.
        n_total = 0
        for cand in range(n27_avail + ng5_avail, 0, -1):
            want27 = round(cand * ratio_27b)
            wantg5 = cand - want27
            if want27 <= n27_avail and wantg5 <= ng5_avail:
                n_total = cand
                break
        take27 = round(n_total * ratio_27b)
        takeg5 = n_total - take27
        mixed[bucket] = c27[:take27] + cg5[:takeg5]
        rng.shuffle(mixed[bucket])
        manifest[bucket] = {SRC_27B: take27, SRC_GPT5: takeg5}
    return mixed, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-start 9-bucket buffer from rollout JSONL.")
    ap.add_argument("--in-dir", default="data/mock/rollouts", help="dir with {actor}/rollouts_*.jsonl")
    ap.add_argument("--out", default="data/mock/buffer_dumps/warmup.sqlite")
    ap.add_argument("--total-capacity", type=int, default=25000)
    ap.add_argument(
        "--ratio-27b",
        type=float,
        default=None,
        help="Phase 0 cold-start source mix: fraction of 27B (pi0_27b) trajectories "
        "per bucket in [0,1]; the rest are gpt5. E.g. 1.0=P0-A (all 27B), 0.0=P0-B "
        "(all gpt5), 0.5=P0-C, 0.7=P0-D, 0.3=P0-E. Omit = ingest ALL (legacy behavior).",
    )
    ap.add_argument("--mix-seed", type=int, default=0, help="seed for reproducible per-bucket source sampling")
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="skip trajectories whose assistant content is all empty (dead-vllm garbage)",
    )
    args = ap.parse_args()

    if args.ratio_27b is not None and not (0.0 <= args.ratio_27b <= 1.0):
        print(f"[warmup] --ratio-27b must be in [0,1], got {args.ratio_27b}", flush=True)
        sys.exit(2)

    files = sorted(glob.glob(str(Path(args.in_dir) / "*" / "rollouts_*.jsonl")))
    if not files:
        print(f"[warmup] no rollout files under {args.in_dir}", flush=True)
        sys.exit(1)
    print(f"[warmup] {len(files)} files: {[Path(f).name for f in files]}", flush=True)

    buffer = BucketReplayBuffer(total_capacity=args.total_capacity)

    sessions = trajs = added = empty = 0
    per_bucket: dict[str, int] = {}
    # grouped[bucket][source] = [(bucket, payload, meta), ...] -- collected first so
    # ratio mixing (if requested) can sample per-bucket per-source (Phase 0).
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
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
                source = _source_tag(t, row, f)
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
                    "policy": source,  # cold-start data-source tag (Phase 0)
                }
                grouped[bucket][source].append((payload, meta))

    # Ingest: either ALL (legacy) or ratio-mixed per bucket (Phase 0).
    manifest: dict[str, dict[str, int]] = {}
    if args.ratio_27b is None:
        for bucket, by_src in grouped.items():
            for src, cands in by_src.items():
                for payload, meta in cands:
                    buffer.add_trajectory(payload, bucket, meta)
                    added += 1
                    per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
                    manifest.setdefault(bucket, {})[src] = manifest.get(bucket, {}).get(src, 0) + 1
    else:
        rng = random.Random(args.mix_seed)
        mixed, manifest = _mix_by_ratio(grouped, args.ratio_27b, rng)
        for bucket, cands in mixed.items():
            for payload, meta in cands:
                buffer.add_trajectory(payload, bucket, meta)
                added += 1
                per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
        print(f"[warmup] ratio-27b={args.ratio_27b} (mix-seed={args.mix_seed}) applied per bucket", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer.dump(out)

    # Manifest sidecar: realized per-bucket per-source counts (Phase 0 acceptance).
    manifest_path = out.with_suffix(".manifest.json")
    manifest_doc = {
        "ratio_27b": args.ratio_27b,
        "mix_seed": args.mix_seed,
        "total_capacity": args.total_capacity,
        "added": added,
        "per_bucket_source": manifest,
        "per_bucket_total": per_bucket,
    }
    manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[warmup] sessions={sessions} trajs={trajs} added={added} " f"empty_skipped={empty}", flush=True)
    print(f"[warmup] per-bucket: {per_bucket}", flush=True)
    print(f"[warmup] per-bucket-source: {manifest}", flush=True)
    print(
        f"[warmup] buffer.stats: { {k: v for k, v in buffer.stats().items() if k in ('total_size','per_bucket')} }",
        flush=True,
    )
    print(f"[warmup] dumped -> {out}  (manifest -> {manifest_path})", flush=True)


if __name__ == "__main__":
    main()
