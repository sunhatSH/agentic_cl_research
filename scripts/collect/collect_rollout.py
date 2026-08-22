"""Multi-turn user-sim rollout collection entry (observer + questioner, NO reward).

Two actor backends, data kept SEPARATE (2026-06-13 request):
  --actor local   -> local vllm serving Qwen3.6-27B
  --actor remote  -> remote model via tokenhub, e.g. gpt-5

Each session: 1 sandbox runs 1 query == 1 rollout (no GRPO/winner), multi-turn
driven by observer (openai/gpt-5-mini) + questioner (anthropic/claude-sonnet-5,
with multi-model rotation). N sessions run in parallel = N sandboxes on N
different seed queries.

observer / questioner are REQUIRED remote agents -- resolved from env by
agents.base (OBSERVER_API_BASE/MODEL/KEY, USERSIM_API_BASE/MODEL/KEY). If the
remote endpoint is unreachable this script aborts (the stage cannot proceed
without them); the launcher (scripts/collect/collect_rollout.sh) pre-checks connectivity.

Output: one JSONL per session-batch under <out-dir>, plus a manifest. Local and
remote actors write to DIFFERENT out-dirs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from agents.base import resolve_observer_client, resolve_questioner_client
from agents.observer import Observer
from agents.personas import sample_persona
from agents.questioner import Questioner
from datasources.cleaning import clean_messages, clean_query
from inference.generate import HTTPGenerateFn
from rollout.collect import make_react_agent_fn
from rollout.session_pool import SessionSandboxPool
from rollout.usersim_collect import run_usersim_session


def iter_seeds(queries_path, limit):
    """Yield (record_id, seed_query) from a queries JSONL file.

    Input format: one JSON line per session, ``{"record_id": "...", "queries": ["q1", ...]}``.
    Takes the FIRST query per session as the seed.
    """
    n = 0
    with open(queries_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            queries = obj.get("queries") or []
            if not queries:
                continue
            yield obj.get("record_id"), queries[0]
            n += 1
            if limit is not None and n >= limit:
                return


def _actor_policy_tag(actor: str) -> str:
    """Map the --actor route to a data-source tag for trajectory provenance.

    local  -> pi0_27b  (on-policy: the training policy itself)
    remote -> gpt5     (off-policy: stronger model, used for cold-start buffer)
    """
    return "pi0_27b" if actor == "local" else "gpt5"


def _traj_to_dict(t, no_clean=False, policy=None):
    messages = t.messages
    if not no_clean:
        result = clean_messages(messages)
        if result.dropped:
            return None
        messages = result.messages
    return {
        "trajectory_id": t.trajectory_id,
        "messages": messages,
        "bucket": t.bucket,
        "response_token_ids": t.response_token_ids,
        "response_mask": t.meta.get("response_mask", []),
        "num_turns": t.meta.get("num_turns"),
        "prompt_tokens": t.meta.get("prompt_tokens", 0),
        "completion_tokens": t.meta.get("completion_tokens", 0),
        # Data-source tag for trajectory provenance. warmup_buffer may use this.
        # mix 27B/gpt-5 trajectories by ratio; kept in buffer meta for attribution.
        "policy": policy,
    }


def run_one_session(args, generate_fn, observer, questioner, record_id, seed, idx):
    """Run a single multi-turn session in its own sandbox. Returns a dict row."""
    # Clean seed query before rollout.
    if not args.no_clean:
        cleaned = clean_query(seed)
        if cleaned is None:
            return None
        seed = cleaned
    persona = sample_persona(random.Random(args.seed + idx))
    pool = SessionSandboxPool(slots=1, backend=args.backend, seed=args.seed + idx)
    agent_fn = make_react_agent_fn(generate_fn, max_turns=args.max_turns)
    res = run_usersim_session(
        pool,
        seed,
        agent_fn,
        persona=persona,
        observer=observer,
        questioner=questioner,
        k_max=args.k_max,
        seed=args.seed + idx,
    )
    trajs = []
    policy = _actor_policy_tag(args.actor)
    for t in res.trajectories:
        d = _traj_to_dict(t, no_clean=args.no_clean, policy=policy)
        if d is not None:
            trajs.append(d)
    if not trajs:
        return None
    return {
        "record_id": record_id,
        "seed_query": seed,
        "persona": res.persona_name,
        "num_turns": res.num_turns,
        "ended_by": res.ended_by,
        "generated_queries": res.generated_queries,
        "trajectories": trajs,
        "reports": [r.__dict__ for r in res.reports],
        "policy": policy,  # data-source tag
    }


def main():
    ap = argparse.ArgumentParser(description="Multi-turn user-sim rollout collection (no reward).")
    ap.add_argument(
        "--queries", required=True, help="queries JSONL path (output of prepare_queries / data-filter)"
    )
    ap.add_argument("--actor", required=True, choices=["local", "remote"])
    ap.add_argument("--actor-base", required=True, help="actor OpenAI base (local vllm or tokenhub)")
    ap.add_argument("--actor-model", required=True)
    ap.add_argument("--actor-key", default="sk-local")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--backend", default="e2b", choices=["e2b", "aliyun", "local"])
    ap.add_argument("--concurrency", type=int, default=8, help="parallel sessions (sandboxes)")
    ap.add_argument("--k-max", type=int, default=3, help="follow-up turns upper bound")
    ap.add_argument("--max-turns", type=int, default=6, help="ReAct turn cap per rollout")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="actor sampling temperature. Default 0.4 (NOT 1.0): this is SINGLE-PATH "
        "cold-start collection (slots=1, no GRPO group) whose goal is high-quality "
        "anti-forgetting ANCHOR trajectories -- agent tool-use tasks want to be done "
        "RIGHT, not sampled diversely. High temp (train-rollout's 1.0) only adds failed "
        "trajectories to the buffer. Applies to both actors (27B on-policy + gpt-5.5 "
        "off-policy). See doc/模型选型.md '温度策略'. GRPO multi-path collection "
        "(collect_cold / sandbox_grpo) keeps a higher temp for winner-selection variance.",
    )
    ap.add_argument("--out-dir", required=True, help="output dir (local/remote MUST differ)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-nodes", type=int, default=1, help="total shards (e.g. 8 nodes)")
    ap.add_argument("--node-rank", type=int, default=0, help="this shard's rank in [0,num_nodes)")
    ap.add_argument("--no-clean", action="store_true", help="skip text cleaning (ZW strip + garble filter)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_tag = f"_r{args.node_rank}" if args.num_nodes > 1 else ""
    out_file = out_dir / f"rollouts_{args.actor}{rank_tag}.jsonl"

    # observer + questioner are REQUIRED -- fail loudly if env not configured.
    try:
        observer = Observer(client=resolve_observer_client())
        questioner = Questioner(client=resolve_questioner_client())
    except Exception as e:  # noqa: BLE001
        print(f"[rollout] FATAL: observer/questioner remote not configured: {e}", flush=True)
        print("[rollout] this stage REQUIRES remote observer+questioner; aborting.", flush=True)
        sys.exit(5)

    generate_fn = HTTPGenerateFn(
        base_url=args.actor_base,
        model=args.actor_model,
        api_key=args.actor_key,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Fail-fast: actor endpoint must be reachable AND produce non-empty output.
    # Without this, a dead vllm endpoint silently yields empty trajectories while
    # `done` keeps climbing (the 8-node empty-run incident, 2026-06-13).
    try:
        probe = generate_fn([{"role": "user", "content": "ping"}])
        if not (probe.text or "").strip():
            print(
                f"[rollout] FATAL: actor {args.actor_base} returned EMPTY output "
                f"(endpoint up but model not serving?). ABORT.",
                flush=True,
            )
            sys.exit(6)
    except Exception as e:  # noqa: BLE001
        print(
            f"[rollout] FATAL: actor {args.actor_base} unreachable: {type(e).__name__}: {e}. ABORT.",
            flush=True,
        )
        sys.exit(6)
    print(f"[rollout] actor self-check OK ({args.actor_model})", flush=True)

    print(
        f"[rollout] actor={args.actor} ({args.actor_model} @ {args.actor_base}) "
        f"backend={args.backend} conc={args.concurrency} -> {out_file}",
        flush=True,
    )

    seeds = list(iter_seeds(args.queries, args.limit))
    if args.num_nodes > 1:
        seeds = [s for i, s in enumerate(seeds) if i % args.num_nodes == args.node_rank]
        print(f"[rollout] shard {args.node_rank}/{args.num_nodes}: {len(seeds)} seeds", flush=True)
    total = len(seeds)
    done = failed = 0
    t0 = time.time()
    with open(out_file, "w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_one_session, args, generate_fn, observer, questioner, rid, seed, i): rid
            for i, (rid, seed) in enumerate(seeds)
        }
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                row = fut.result()
                if row is None:
                    failed += 1
                else:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[rollout] session {rid} failed: {type(e).__name__}: {e}", flush=True)
            if (done + failed) % args.log_every == 0:
                rate = (done + failed) / max(1e-9, time.time() - t0)
                print(
                    f"[rollout] {done+failed}/{total} done={done} failed={failed} " f"({rate:.2f}/s)",
                    flush=True,
                )

    print(
        f"[rollout] DONE actor={args.actor} sessions={total} done={done} failed={failed} "
        f"in {time.time()-t0:.0f}s -> {out_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
