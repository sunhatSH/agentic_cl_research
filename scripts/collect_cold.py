"""Cold-start trajectory collection (跑通优先, 不打分, 无三 agent).

Drives the SIMPLEST useful rollout loop to manufacture an initial replay buffer:

    raw _stage_prefix_pass.jsonl ─► seed q1 (each session's first user query)
        └─ per seed: SessionSandboxPool(backend=e2b).run_query(q1, react_agent)
              react_agent = make_react_agent_fn(HTTPGenerateFn(local 27B vllm))
        └─ ingest all M slot trajectories into the 7-bucket BucketReplayBuffer
    buffer.dump(out.sqlite)

This is deliberately a SINGLE turn per session: no observer / questioner / judge
(doc/UserSim_多轮Query在线生成.md is for online multi-turn WITH scoring). Cold
data only needs "seed -> actor rollout -> sandbox exec -> trajectory". Rewards
stay None; bucket comes from the actor's <task_domain> tag (trainer.domain_tagging).

The actor (Qwen3.6-27B) is served by a LOCAL vllm OpenAI-compatible server (see
scripts/collect_cold.sh) and reached via inference.HTTPGenerateFn -- a local
model, not a remote API.

Usage (normally launched by scripts/collect_cold.sh after vllm is up):
    python scripts/collect_cold.py \
        --raw /mnt/.../source/_stage_prefix_pass.jsonl \
        --limit 10000 --group-size 8 --backend e2b \
        --actor-base http://127.0.0.1:8000/v1 --actor-model cold-actor \
        --out logs/cold/buffer.sqlite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.generate import HTTPGenerateFn
from replay_buffer.bucket import BucketReplayBuffer
from rollout.collect import ingest_trajectories, make_react_agent_fn
from rollout.session_pool import SessionSandboxPool
from scripts.prepare_queries import iter_sessions, session_queries

# The 7 capability buckets (must match BucketReplayBuffer / configs/base.yaml).
VALID_BUCKETS = (
    "Workflow", "SysOps", "Dialogue", "Finance",
    "Communication", "Knowledge", "OfficeQA",
)


def iter_seeds(raw_path: str | Path, limit: int | None):
    """Yield (record_id, seed_query) = each session's FIRST user query (q1).

    Reuses prepare_queries.session_queries (drops <summary>/empty turns) and
    takes [0]. Sessions with no usable query are skipped.
    """
    n = 0
    for obj in iter_sessions(raw_path):
        record = obj.get("record", {}) or {}
        queries = session_queries(record)
        if not queries:
            continue
        yield obj.get("record_id"), queries[0]
        n += 1
        if limit is not None and n >= limit:
            return


def build_buffer() -> BucketReplayBuffer:
    """7-bucket buffer with library defaults (capacity etc.)."""
    return BucketReplayBuffer()


def main() -> None:
    ap = argparse.ArgumentParser(description="Cold-start trajectory collection (no scoring).")
    ap.add_argument("--raw", required=True, help="raw _stage_prefix_pass.jsonl path")
    ap.add_argument("--limit", type=int, default=10000, help="number of sessions/seeds to collect")
    ap.add_argument("-m", "--group-size", type=int, default=8, help="slots (rollouts) per query")
    ap.add_argument("--backend", default="e2b", choices=["e2b", "local"])
    ap.add_argument("--actor-base", default="http://127.0.0.1:8000/v1", help="local vllm OpenAI base")
    ap.add_argument("--actor-model", default="cold-actor", help="vllm --served-model-name")
    ap.add_argument("--max-turns", type=int, default=6, help="ReAct turn cap per rollout")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="logs/cold/buffer.sqlite", help="buffer dump path")
    ap.add_argument("--log-every", type=int, default=50, help="print stats every N sessions")
    ap.add_argument("--dump-every", type=int, default=500, help="checkpoint dump every N sessions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Actor = local 27B served by vllm, reached over the HTTP GenerateFn.
    generate_fn = HTTPGenerateFn(
        base_url=args.actor_base,
        model=args.actor_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    agent_fn = make_react_agent_fn(generate_fn, max_turns=args.max_turns)

    buffer = build_buffer()

    total = added = skipped = failed = 0
    t0 = time.time()
    print(
        f"[cold] start backend={args.backend} M={args.group_size} "
        f"actor={args.actor_base} ({args.actor_model}) limit={args.limit}",
        flush=True,
    )

    for record_id, seed in iter_seeds(args.raw, args.limit):
        total += 1
        # One session = spawn M slots, single-turn rollout of the seed, ingest all.
        pool = SessionSandboxPool(slots=args.group_size, backend=args.backend, seed=args.seed)
        try:
            pool.spawn()
            trajs = pool.run_query(seed, agent_fn)
            counts = ingest_trajectories(buffer, trajs, valid_buckets=VALID_BUCKETS)
            added += counts["added"]
            skipped += counts["skipped"]
        except Exception as e:  # noqa: BLE001 -- one bad session must not kill the batch
            failed += 1
            print(f"[cold] session {record_id} failed: {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                pool.destroy_all()
            except Exception:  # noqa: BLE001
                pass

        if total % args.log_every == 0:
            rate = total / max(1e-9, time.time() - t0)
            print(
                f"[cold] sessions={total} added={added} skipped={skipped} "
                f"failed={failed} ({rate:.2f} sess/s)",
                flush=True,
            )
        if args.dump_every and total % args.dump_every == 0:
            buffer.dump(out_path)
            print(f"[cold] checkpoint dump -> {out_path} (added={added})", flush=True)

    buffer.dump(out_path)
    elapsed = time.time() - t0
    print(
        f"[cold] DONE sessions={total} trajectories_added={added} "
        f"skipped={skipped} failed={failed} in {elapsed:.0f}s -> {out_path}",
        flush=True,
    )
    print(f"[cold] buffer.stats(): {buffer.stats()}", flush=True)


if __name__ == "__main__":
    main()
