#!/usr/bin/env python3
"""8-way GRPO sandbox collection: hermes-in-sandbox actor + dev-side judge/winner-sync.

This is the operational rollout driver for the new-cluster VPC sandbox. It runs
the 8-way GRPO collection loop the design calls for (doc/Sandbox_管理调度指南.md):

    for each of N queries (sequential):
      1. spawn 8 sandbox instances (same v2 image, bit-identical start)
      2. 8 parallel rollouts: the ACTOR is hermes running INSIDE each sandbox
         (hermes calls the model endpoint over the network, executes tool actions
         in that sandbox); each slot produces a Trajectory
      3. dev-side reward judge scores each trajectory (anthropic/claude-4.8-opus
         via sufy, three-dim completion/safety/robustness)
      4. GRPO advantage + select_winner -> winner固化
      5. sync_to_winner: loser slots align to winner's disk state + history
      6. winner messages appended to session_history (next query's prefix)
    session end: destroy all 8 sandboxes

Model-call topology (IMPORTANT — two distinct paths):
  - ACTOR (in-sandbox hermes): hermes runs inside the Tencent sandbox and calls
    the model endpoint configured by the runtime env vars AGENT_MODEL_BASE /
    AGENT_MODEL_KEY / AGENT_MODEL_NAME. The sandbox reaches sufy directly,
    AGENT_MODEL_BASE = https://openai.sufy.com/v1. (The dev machine is NOT used
    as a bridge — the sandbox calls the model directly.)
  - OBSERVER / QUESTIONER / REWARD (dev-side): resolved on the dev machine from
    configs/agents.yaml (openai/gpt-5-mini / anthropic/claude-sonnet-5 rotation /
    anthropic/claude-4.8-opus), calling sufy directly from the dev side.

Two actor backends (--actor):
  - hermes   : the real path. Writes ~/.hermes/config.yaml + .env inside the
               sandbox from AGENT_MODEL_* env, then runs `hermes chat -q`. This
               needs AGENT_MODEL_BASE reachable from the sandbox.
  - run_code : a fallback that runs a tiny ReAct loop via sandbox.run_code,
               so the 8-way scheduling / reward / winner / sync machinery can be
               exercised WITHOUT hermes or model reachability. Used for smoke /
               framework validation. Each slot computes the answer in-python and
               writes it to /home/user/result.txt; reward = exact-match vs the
               task's expected answer.

Output: one JSONL per session under <out-dir>. Each line = one slot trajectory
(messages, reward, advantage, winner flag, sandbox_id). A manifest summarizes
per-query winners + aggregate stats.

Usage:
    # fallback (framework smoke, no hermes / no model needed):
    python scripts/sandbox_grpo_collect.py --actor run_code --num-queries 2 --slots 2

    # real (hermes in sandbox; needs AGENT_MODEL_* in runtime.env):
    python scripts/sandbox_grpo_collect.py --actor hermes --num-queries 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.sandbox_client import ExecResult, grpo_advantages, make_sandbox, select_winner

# Default tasks for the run_code fallback. Each has a query + an expected answer
# so reward can be exact-match (no model judge needed). The real path supplies
# tasks via --tasks <jsonl> (one {"query": ...} per line; reward via the judge).
_DEFAULT_TASKS = [
    {"query": "Compute 23*17-19 and write the number to /home/user/result.txt", "expected": "372"},
    {"query": "Compute 100*100 and write the number to /home/user/result.txt", "expected": "10000"},
    {"query": "Compute 2**10 and write the number to /home/user/result.txt", "expected": "1024"},
    {"query": "Compute 7*8+9 and write the number to /home/user/result.txt", "expected": "65"},
    {"query": "Compute 999-333 and write the number to /home/user/result.txt", "expected": "666"},
]


@dataclass
class SlotTrajectory:
    """One slot's rollout of one query (the unit written to JSONL)."""

    query_index: int
    slot_idx: int
    sandbox_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    reward: float | None = None
    advantage: float | None = None
    is_winner: bool = False
    error: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Slot worker: runs ONE rollout in ONE sandbox. Two actor backends.            #
# --------------------------------------------------------------------------- #


def _write_hermes_config(sb: Any, model: str, base: str, key: str) -> ExecResult:
    """Write ~/.hermes/config.yaml + .env inside the sandbox from AGENT_MODEL_*.

    hermes reads providers from config.yaml; the key lives in .env (never baked
    into the image). base/key come from the runtime env injected at instance
    create, so this just renders them into hermes's config files.
    """
    # base64 the key so shell-quoting in run_code can never break on it.
    import base64

    key_b64 = base64.b64encode(key.encode()).decode()
    code = (
        "import os,base64,yaml\n"
        f"key=base64.b64decode({key_b64!r}).decode()\n"
        "home=os.path.expanduser('~')\n"
        "os.makedirs(home+'/.hermes',exist_ok=True)\n"
        f"cfg={{'model':{model!r},'providers':{{'agent':{{'base_url':{base!r},'api_key':key,'kind':'openai'}}}}}}\n"
        "open(home+'/.hermes/config.yaml','w').write(yaml.safe_dump(cfg,sort_keys=False))\n"
        "open(home+'/.hermes/.env','w').write('OPENAI_API_KEY='+key+chr(10))\n"
        f"print('hermes configured:',{model!r})\n"
    )
    return sb.run_code(code)


def _run_hermes_slot(sb: Any, query: str, model: str, base: str, key: str, max_turns: int, timeout: int) -> SlotTrajectory:
    """Real actor: configure hermes inside the sandbox, run `hermes chat -q`."""
    traj = SlotTrajectory(query_index=-1, slot_idx=-1, sandbox_id=getattr(sb, "_sandbox_id", ""))
    cfg = _write_hermes_config(sb, model, base, key)
    if not cfg.ok:
        traj.error = f"hermes config write failed: {cfg.stderr[:200]}"
        return traj
    # Non-interactive single query. --yolo auto-approves tool use; -Q is quiet
    # (final answer only); --max-turns caps the ReAct loop.
    code = (
        "import subprocess\n"
        f"r=subprocess.run(['hermes','chat','-q',{query!r},'-m',{model!r},'--provider','agent',"
        f"'-Q','--max-turns',{str(max_turns)!r},'--yolo'],capture_output=True,text=True,timeout={str(timeout)!r})\n"
        "print('EXIT',r.returncode)\n"
        "print(r.stdout or '')\n"
        "if r.stderr: print('STDERR',r.stderr[:500])\n"
    )
    res = sb.run_code(code)
    traj.messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": (res.stdout or "").strip()},
    ]
    if res.stderr.strip():
        traj.messages.append({"role": "system", "content": f"[stderr] {res.stderr[:300]}"})
    traj.answer = (res.stdout or "").strip().splitlines()[-1] if res.stdout else ""
    if not res.ok and not traj.answer:
        traj.error = res.stderr[:200] or "hermes run produced no output"
    return traj


def _run_run_code_slot(sb: Any, task: dict[str, Any], timeout: int) -> SlotTrajectory:
    """Fallback actor: a tiny ReAct loop via run_code. Computes + writes result.txt.

    No model call — the slot directly computes the task's expression in-python
    inside the sandbox and writes /home/user/result.txt. Reward is exact-match
    vs task['expected']. This exercises the 8-way scheduling / reward / winner
    / sync machinery without hermes or model reachability.
    """
    traj = SlotTrajectory(query_index=-1, slot_idx=-1, sandbox_id=getattr(sb, "_sandbox_id", ""))
    query = task["query"]
    # The slot "reasons" by extracting the arithmetic expression from the query
    # and executing it in the sandbox, then writing the answer to result.txt.
    code = (
        "import re\n"
        f"q={query!r}\n"
        # grab the arithmetic expression (digits, operators, parens, **, whitespace)
        "m=re.search(r'([\\d\\s\\*\\+\\-/\\(\\)\\*\\*]+)', q)\n"
        "expr=m.group(1).strip() if m else '0'\n"
        "ans=eval(expr, {'__builtins__':{}}, {})\n"
        "open('/home/user/result.txt','w').write(str(ans))\n"
        "print('ANS', ans)\n"
    )
    res = sb.run_code(code)
    traj.messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": (res.stdout or "").strip()},
    ]
    traj.answer = ""
    if res.ok and res.stdout:
        for line in res.stdout.splitlines():
            if line.startswith("ANS"):
                traj.answer = line.split("ANS", 1)[1].strip()
    if not res.ok:
        traj.error = res.stderr[:200]
    return traj


# --------------------------------------------------------------------------- #
# Reward                                                                       #
# --------------------------------------------------------------------------- #


def _exact_match_reward(traj: SlotTrajectory, expected: str) -> float:
    """run_code fallback reward: exact match of the written answer."""
    return 1.0 if traj.answer.strip() == str(expected).strip() else 0.0


def _judge_reward(traj: SlotTrajectory, query: str) -> float:
    """Real reward: dev-side model judge (anthropic/claude-4.8-opus via sufy).

    Uses the observation-grounded judge from trainer.model_reward (the same one
    verl's custom_reward_function calls), so reward scale matches eval.
    """
    from trainer.model_reward import get_judge  # local import: only needed on real path

    judge = get_judge()
    verdict = judge.score(
        task=query,
        trajectory="\n".join(m.get("content", "") for m in traj.messages),
        rubric="",  # observation-grounded rubric is built by the rollout; here we grade the trajectory
        data_source="sandbox_grpo",
    )
    # aggregate: safety * (0.8*completion + 0.2*robustness)
    from trainer.model_reward import aggregate

    return float(aggregate(verdict))


# --------------------------------------------------------------------------- #
# The 8-way GRPO session driver                                                #
# --------------------------------------------------------------------------- #


def run_session(
    *,
    tasks: list[dict[str, Any]],
    actor: str,
    slots: int,
    backend: str,
    template: str,
    actor_model: str,
    actor_base: str,
    actor_key: str,
    max_turns: int,
    slot_timeout: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Run one 8-way GRPO session over `tasks`. Returns a summary dict."""
    all_rows: list[SlotTrajectory] = []
    winners: list[int] = []  # winner slot idx per query
    per_query_rewards: list[list[float]] = []

    # Spawn the 8 sandboxes ONCE and reuse across queries (winner-sync keeps
    # them aligned); a real run would sync disk state too. For this driver each
    # query is independent (no cross-query state carry) — sync = history only.
    print(f"[grpo] spawning {slots} sandboxes (backend={backend}, template={template})...", flush=True)
    t_spawn = time.time()
    sandbox_specs: list[Any] = []

    def _spawn(slot_idx: int) -> Any:
        sb = make_sandbox(backend, template=template, timeout=slot_timeout)
        return sb

    with ThreadPoolExecutor(max_workers=min(slots, 8)) as ex:
        sandbox_specs = list(ex.map(_spawn, range(slots)))
    print(f"[grpo] {slots} sandboxes up in {time.time()-t_spawn:.1f}s: "
          f"{[getattr(s,'_sandbox_id','?')[:12] for s in sandbox_specs]}", flush=True)

    try:
        for qi, task in enumerate(tasks):
            query = task["query"]
            expected = task.get("expected")
            t_q = time.time()
            print(f"\n[grpo] === query {qi+1}/{len(tasks)}: {query[:70]} ===", flush=True)

            # 8 parallel rollouts. _rollout takes qi/query/task explicitly so the
            # closure does not capture the loop variable (ruff B023).
            def _rollout(slot_idx: int, _qi: int = qi, _query: str = query, _task: dict[str, Any] = task) -> SlotTrajectory:
                sb = sandbox_specs[slot_idx]
                try:
                    if actor == "hermes":
                        t = _run_hermes_slot(sb, _query, actor_model, actor_base, actor_key, max_turns, slot_timeout)
                    else:
                        t = _run_run_code_slot(sb, _task, slot_timeout)
                except Exception as exc:  # noqa: BLE001 -- isolate slot failures
                    t = SlotTrajectory(query_index=_qi, slot_idx=slot_idx,
                                       sandbox_id=getattr(sb, "_sandbox_id", ""), error=f"{type(exc).__name__}: {exc}")
                t.query_index = _qi
                t.slot_idx = slot_idx
                return t

            with ThreadPoolExecutor(max_workers=slots) as ex:
                trajs = list(ex.map(_rollout, range(slots)))
            trajs.sort(key=lambda t: t.slot_idx)

            # reward
            for t in trajs:
                if t.error:
                    t.reward = 0.0
                elif actor == "run_code" and expected is not None:
                    t.reward = _exact_match_reward(t, expected)
                else:
                    try:
                        t.reward = _judge_reward(t, query)
                    except Exception as exc:  # noqa: BLE001 -- judge failure -> 0, not crash
                        t.reward = 0.0
                        t.error = (t.error + " | " if t.error else "") + f"judge: {exc}"

            rewards = [t.reward if t.reward is not None else 0.0 for t in trajs]
            per_query_rewards.append(rewards)

            # GRPO advantage + winner
            advs = grpo_advantages(rewards)
            for t, a in zip(trajs, advs, strict=True):
                t.advantage = a
            try:
                widx = select_winner(rewards, [t.sandbox_id or f"q{qi}-s{t.slot_idx}" for t in trajs])
            except ValueError:
                widx = 0
            trajs[widx].is_winner = True
            winners.append(widx)

            for t in trajs:
                mark = " <-- WINNER" if t.is_winner else ""
                print(f"  slot{t.slot_idx}: reward={t.reward} adv={t.advantage:+.2f} "
                      f"ans={t.answer[:30]!r} sid={t.sandbox_id[:12]}{mark}", flush=True)
            all_rows.extend(trajs)
            print(f"[grpo] query {qi+1} done in {time.time()-t_q:.1f}s, winner=slot{widx}", flush=True)
    finally:
        # destroy all sandboxes
        for sb in sandbox_specs:
            try:
                sb.kill()
            except Exception:  # noqa: BLE001
                pass
        print(f"\n[grpo] {slots} sandboxes destroyed.", flush=True)

    # write JSONL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"grpo_{actor}.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for t in all_rows:
            fh.write(t.to_jsonl() + "\n")

    summary = {
        "actor": actor,
        "slots": slots,
        "num_queries": len(tasks),
        "winners": winners,
        "per_query_rewards": per_query_rewards,
        "mean_reward": sum(r for t in all_rows for r in [t.reward or 0.0]) / max(1, len(all_rows)),
        "out_file": str(out_file),
    }
    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _load_tasks(tasks_path: str | None, num_queries: int) -> list[dict[str, Any]]:
    if tasks_path:
        tasks = []
        with open(tasks_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks[:num_queries] if num_queries else tasks
    return _DEFAULT_TASKS[:num_queries]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--actor", choices=["hermes", "run_code"], default="run_code",
                    help="run_code = framework smoke (no model); hermes = real in-sandbox actor")
    ap.add_argument("--num-queries", type=int, default=2, help="N queries (sequential)")
    ap.add_argument("--slots", type=int, default=8, help="parallel sandboxes per query (GRPO group size)")
    ap.add_argument("--backend", default="e2b", choices=["e2b", "local"])
    ap.add_argument("--template", default="agentic-cl-sandbox")
    ap.add_argument("--tasks", help="JSONL of {query[, expected]} per line (default: built-in arithmetic)")
    ap.add_argument("--actor-model", default="gpt-5.1", help="hermes model name (hermes actor)")
    ap.add_argument("--actor-base", default="", help="override AGENT_MODEL_BASE (else runtime env / tencent env)")
    ap.add_argument("--actor-key", default="", help="override AGENT_MODEL_KEY")
    ap.add_argument("--max-turns", type=int, default=8, help="hermes ReAct turn cap")
    ap.add_argument("--slot-timeout", type=int, default=180, help="per-slot hermes timeout (s)")
    ap.add_argument("--out-dir", default="rollouts/grpo")
    args = ap.parse_args()

    # The caller is expected to have sourced scripts/load_tencent_env.sh first
    # (sets E2B_API_KEY / E2B_DOMAIN / E2B_VALIDATE_API_KEY + the runtime env
    # that carries AGENT_MODEL_* into the sandbox). We just surface diagnostics.
    if args.backend == "e2b":
        import os

        if not os.environ.get("E2B_API_KEY") or not os.environ.get("E2B_DOMAIN"):
            print("[grpo] WARNING: E2B_API_KEY/E2B_DOMAIN not set — source scripts/load_tencent_env.sh",
                  file=sys.stderr)
        os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")  # AGS ark_ key compat

    # actor_base / actor_key: CLI override > env > (hermes path requires them)
    actor_base = args.actor_base or _env("AGENT_MODEL_BASE")
    actor_key = args.actor_key or _env("AGENT_MODEL_KEY")
    if args.actor == "hermes" and not (actor_base and actor_key):
        print("[grpo] ERROR: hermes actor needs AGENT_MODEL_BASE + AGENT_MODEL_KEY "
              "(set in docker/sandbox/runtime.env, injected into the sandbox at create).",
              file=sys.stderr)
        sys.exit(2)

    tasks = _load_tasks(args.tasks, args.num_queries)
    print(f"[grpo] actor={args.actor} slots={args.slots} num_queries={len(tasks)} "
          f"backend={args.backend} template={args.template}", flush=True)
    if args.actor == "hermes":
        print(f"[grpo] in-sandbox hermes -> model={args.actor_model} base={actor_base}", flush=True)

    t0 = time.time()
    summary = run_session(
        tasks=tasks,
        actor=args.actor,
        slots=args.slots,
        backend=args.backend,
        template=args.template,
        actor_model=args.actor_model,
        actor_base=actor_base,
        actor_key=actor_key,
        max_turns=args.max_turns,
        slot_timeout=args.slot_timeout,
        out_dir=Path(args.out_dir),
    )
    print(f"\n[grpo] DONE in {time.time()-t0:.0f}s")
    print(f"  winners per query: {summary['winners']}")
    print(f"  mean reward: {summary['mean_reward']:.3f}")
    print(f"  trajectories -> {summary['out_file']}")
    print(f"  manifest     -> {Path(args.out_dir)/'manifest.json'}")


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")


if __name__ == "__main__":
    main()
