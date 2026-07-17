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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.actor import CliStdoutActor, make_actor, register_actor
from rollout.sandbox_client import ExecResult, grpo_advantages, make_sandbox, select_winner



@dataclass
class SlotTrajectory:
    """One slot's rollout of one query (the unit written to JSONL)."""

    query_index: int
    slot_idx: int
    sandbox_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""       # hermes _cached_system_prompt (base identity + date + model)
    tools: list[dict[str, Any]] = field(default_factory=list)  # hermes tool definitions (OpenAI format)
    api_calls: int = 0            # number of API calls made this turn
    partial: bool = False         # partial completion flag
    children: list[dict[str, Any]] = field(default_factory=list)  # sub-agent (delegate_task) trajectories
    answer: str = ""
    reward: float | None = None
    advantage: float | None = None
    is_winner: bool = False
    error: str = ""
    bucket: str = ""
    persona_name: str = ""       # session persona (multi-turn collect)
    num_turns: int = 0           # actual turns run (multi-turn collect)
    ended_by: str = ""           # k_budget | end_session | agent_error (multi-turn)
    observer_reports: list[dict[str, Any]] = field(default_factory=list)  # per-turn debug
    qc_hard: bool = False        # failed hard QC (tool hallucination/truncation/loop/...) -> dropped
    qc_codes: list[str] = field(default_factory=list)  # QC failure-mode codes tripped

    def to_jsonl(self) -> str:
        # Standard OpenAI intermediate format (agent_data_tools compatible).
        msgs = list(self.messages)
        # Inject system_prompt as the first system message if present and not already in messages.
        if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        # Clean up tool messages and normalize assistant fields.
        for m in msgs:
            if m.get("role") == "tool":
                if "success" not in m:
                    m["success"] = True
                m.pop("tool_name", None)
            if m.get("role") == "assistant":
                m.pop("finish_reason", None)
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tc.pop("call_id", None)
                    tc.pop("response_item_id", None)

        obj = {
            "status": "completed" if not self.error else "error",
            "total_steps": sum(1 for m in msgs if m.get("role") in ("assistant", "tool")),
            "enable_thinking": True,
            "messages": msgs,
            "tools": self.tools,
            "metadata": {
                "bucket": self.bucket,
                "persona_name": self.persona_name,
                "num_turns": self.num_turns,
                "ended_by": self.ended_by,
                "api_calls": self.api_calls,
                "partial": self.partial,
                "sandbox_id": self.sandbox_id,
                "query_index": self.query_index,
                "reward": self.reward,
                "error": self.error,
                "observer_reports": self.observer_reports,
            },
        }
        raw = json.dumps(obj, ensure_ascii=False)
        try:
            raw.encode("utf-8")
        except UnicodeEncodeError:
            raw = (
                raw.encode("utf-8", errors="backslashreplace")
                .decode("utf-8", errors="backslashreplace")
            )
        return raw


# --------------------------------------------------------------------------- #
# Slot worker: runs ONE rollout in ONE sandbox. Two actor backends.            #
# --------------------------------------------------------------------------- #


def _write_hermes_config(sb: Any, model: str, base: str) -> ExecResult:
    """Write ~/.hermes/config.yaml + .env + AGENTS.md inside the sandbox.

    The model key is already injected into the sandbox as AGENT_MODEL_KEY
    (from runtime.env via E2BSandbox envs=).  We read it INSIDE the sandbox
    so the key never travels through the dev-machine process — it stays
    inside the sandbox where it belongs.

    AGENTS.md bans the ``clarify`` tool (non-interactive sandbox has no human
    to answer) and sets ``terminal.timeout`` to 60 s so hangs don't eat the
    whole slot budget.
    """
    code = (
        "import os, yaml, subprocess, stat\n"
        f"model = os.environ.get('AGENT_MODEL_NAME', {model!r})\n"
        f"base  = os.environ.get('AGENT_MODEL_BASE', {base!r})\n"
        "key   = os.environ['AGENT_MODEL_KEY']\n"  # MUST be injected, else fail loud
        "home  = os.path.expanduser('~')\n"
        "os.makedirs(home + '/.hermes', exist_ok=True)\n"
        # hermes config
        "cfg = {'model': model, 'providers': {'agent': {'base_url': base, 'api_key': key}}}\n"
        "open(home + '/.hermes/config.yaml', 'w').write(yaml.safe_dump(cfg, sort_keys=False))\n"
        "open(home + '/.hermes/.env', 'w').write('OPENAI_API_KEY=' + key + chr(10))\n"
        # Serper search CLI — hermes doesn't natively support Serper, so we
        # build a tiny script from a list of lines (avoids shell-escaping hell).
        "sk = os.environ.get('SERPER_API_KEY', '').strip()\n"
        "if sk:\n"
        "    os.makedirs(home + '/.local/bin', exist_ok=True)\n"
        "    lines = ['#!/usr/bin/env python3',\n"
        "        'import json,os,sys,urllib.request as u',\n"
        "        'sk=os.environ[\"SERPER_API_KEY\"]',\n"
        "        'q=sys.argv[1]if len(sys.argv)>1 else sys.stdin.read().strip()',\n"
        "        \"r=u.Request('https://google.serper.dev/search',\"\n"
        "        \"    data=json.dumps({'q':q,'num':10}).encode(),\"\n"
        "        \"    headers={'X-API-KEY':sk,'Content-Type':'application/json'})\",\n"
        "        'd=json.loads(u.urlopen(r,timeout=15).read())',\n"
        "        'for i,o in enumerate(d.get(\"organic\",[])[:10],1):',\n"
        "        \"    t=o['title'];l=o['link'];s=o.get('snippet','')[:200]\",\n"
        "        \"    print(f'{i}. {t}\\\\n   {l}\\\\n   {s}\\\\n')\",\n"
        "    ]\n"
        "    open(home + '/.local/bin/serper-search', 'w').write(chr(10).join(lines) + chr(10))\n"
        "    os.chmod(home + '/.local/bin/serper-search', 0o755)\n"
        # AGENTS.md: ban clarify + teach serper-search
        "open(home + '/AGENTS.md', 'w').write("
        "'## Rules\\n'"
        "'- NEVER call the clarify tool. You are running in a non-interactive '"
        "'sandbox with no human feedback path. If you need clarifications, make '"
        "'reasonable assumptions and proceed.\\n'"
        "'- **Web search**: use `serper-search` CLI. e.g. `serper-search \"query\"`.\\n')\n"
        # reduce internal command timeout so clarify/failures don't eat slot budget
        "subprocess.run(['hermes', 'config', 'set', 'terminal.timeout', '60'], "
        "capture_output=True)\n"
        f"print('hermes configured: model=' + repr(model))\n"
    )
    return sb.run_code(code)


_SESSION_ID_RE = re.compile(r"session_id:\s*(\S+)")


def _hermes_chat(
    sb: Any, query: str, model: str, hermes_max_turns: int, timeout: int,
    resume_sid: str | None = None,
) -> tuple[str, str, bool, str | None]:
    """Run one `hermes chat -q <query>` in the sandbox.

    Returns (stdout, stderr, ok, session_id). The sandbox filesystem is
    persistent across calls, so successive queries on the SAME sandbox see files
    written by earlier turns. Passing ``resume_sid`` (from a prior turn's stderr)
    resumes hermes' OWN conversation memory via ``--resume`` — so the actor keeps
    its reasoning context across turns, not just the sandbox file state.
    """
    import shlex

    cmd = (
        f"hermes chat -q {shlex.quote(query)} -m {shlex.quote(model)} "
        f"--provider agent -Q --max-turns {hermes_max_turns} --yolo"
    )
    if resume_sid:
        cmd += f" --resume {shlex.quote(resume_sid)}"
    try:
        out = sb._sb.commands.run(cmd, timeout=timeout)  # type: ignore[union-attr]
        stdout, stderr = (out.stdout or "").strip(), (out.stderr or "").strip()
        m = _SESSION_ID_RE.search(stderr)
        sid = m.group(1) if m else None
        return stdout, stderr, out.exit_code == 0, sid
    except Exception as exc:  # noqa: BLE001 — isolate slot failures
        return "", f"{type(exc).__name__}: {exc}", False, None


# Register the default CLI actor now that _hermes_chat exists. The collection loop
# resolves the actor by name via make_actor(); swapping to the structured actor (P2)
# touches no loop code. chat_fn is injected to avoid a circular import.
register_actor("hermes_cli", lambda **kw: CliStdoutActor(chat_fn=_hermes_chat))


def _run_hermes_slot(sb: Any, query: str, model: str, base: str, hermes_max_turns: int, timeout: int) -> SlotTrajectory:
    """Real actor: configure hermes inside the sandbox, then run `hermes chat` directly.

    Uses ``commands.run`` (NOT a Python wrapper subprocess) so hermes stdout/stderr
    are captured cleanly without wrapper noise.
    """
    traj = SlotTrajectory(query_index=-1, slot_idx=-1, sandbox_id=getattr(sb, "_sandbox_id", ""))
    cfg = _write_hermes_config(sb, model, base)
    if not cfg.ok:
        traj.error = f"hermes config write failed: {cfg.stderr[:200]}"
        return traj

    stdout, stderr, ok, _sid = _hermes_chat(sb, query, model, hermes_max_turns, timeout)
    if not stdout and not ok and stderr.startswith(("TimeoutException", "Exception", "RuntimeError")):
        traj.error = stderr[:200]
        return traj

    traj.messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": stdout},
    ]
    if stderr:
        traj.messages.append({"role": "system", "content": f"[stderr] {stderr[:500]}"})
    # Last non-empty line is typically the hermes session-id line; the real answer
    # is above it.  Store both for downstream consumers.
    traj.answer = [l for l in stdout.splitlines() if l.strip()][-1] if stdout.strip() else ""
    if not ok and not traj.answer:
        traj.error = stderr[:200] or "hermes run produced no output"
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
# Workspace upload                                                              #
# --------------------------------------------------------------------------- #

# Root directory of taskspec data. Each subdir s_<id>/files/ mirrors the
# sandbox workspace layout and is uploaded before hermes runs.
_TASKSPECS_DIR = Path(__file__).resolve().parent.parent / "data" / "taskspecs"

# Files skipped during upload (OS junk / lock files).
_SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db"})


def _upload_workspace(sb: Any, record_id: str, workspace_dir: str | None = None) -> int:
    """Upload workspace files into the sandbox from ``<workspace_dir>/<record_id>/files/``.

    When ``workspace_dir`` is None, falls back to ``data/taskspecs/``.
    Preserves the directory structure (relative paths).  Junk files
    (.DS_Store, Thumbs.db, ~$* lock files) are skipped.  Returns the
    number of files uploaded (0 if the taskspec has no files/ dir).
    """
    base = Path(workspace_dir) if workspace_dir else _TASKSPECS_DIR
    src = base / record_id / "files"
    if not src.is_dir():
        return 0

    entries: list[dict[str, Any]] = []
    for fpath in sorted(src.rglob("*")):
        if not fpath.is_file():
            continue
        name = fpath.name
        if name in _SKIP_NAMES or name.startswith("~$"):
            continue
        rel = str(fpath.relative_to(src))
        entries.append({"path": rel, "data": fpath.read_bytes()})

    if entries:
        # write_files accepts list[WriteEntry] where WriteEntry = {"path": str, "data": bytes}.
        sb._sb.files.write_files(entries)  # type: ignore[union-attr]
    return len(entries)


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


def _run_one_collect_query(
    task: dict[str, Any],
    qi: int,
    *,
    actor: str,
    actor_model: str,
    actor_base: str,
    max_turns: int,
    hermes_max_turns: int,
    slot_timeout: int,
    backend: str,
    template: str,
    observer: Any = None,
    questioner: Any = None,
    rng_seed: int = 0,
    workspace_dir: str | None = None,
    observer_log: Path | None = None,
    actor_impl: str = "hermes_structured",
) -> SlotTrajectory:
    """Full lifecycle for ONE query in collect mode (multi-turn, no reward/winner).

    Each query gets its OWN persistent sandbox. Flow per session:
        spawn → upload workspace → persona = sample_persona(rng)
        for turn in 1..K  (K = randint(2, max_turns)):
            actor (hermes) runs the current query in the sandbox
            observer.observe(sandbox diff) → report        [state-only, no judge]
            questioner.next_query(persona, report, history) → follow-up | end
        destroy sandbox
    The whole multi-turn conversation is ONE trajectory (messages = all turns).

    When observer/questioner are None → single-turn (seed query only), for
    smoke tests. reward/winner are NOT computed here (that's the training stage).
    """
    import random as _random

    from agents.personas import PERSONAS, sample_persona

    actor_obj = make_actor(actor_impl)  # facade: hermes_cli (default) | hermes_structured (P2)
    from agents.questioner import PatienceTracker

    query = task["query"]
    record_id = task.get("record_id", "")
    bucket = task.get("bucket", "")
    rng = _random.Random(rng_seed)

    sb = make_sandbox(backend, template=template, timeout=10800)  # 3h total; per-turn ctrl via commands.run timeout
    sid = getattr(sb, "_sandbox_id", "")
    t = SlotTrajectory(query_index=qi, slot_idx=0, sandbox_id=sid, bucket=bucket)
    all_messages: list[dict[str, Any]] = []

    try:
        if record_id:
            n = _upload_workspace(sb, record_id, workspace_dir)
            if n:
                print(f"  q{qi}: uploaded {n} ws files ({record_id})", flush=True)

        # hermes must be configured once; sandbox FS persists across turns.
        cfg = _write_hermes_config(sb, actor_model, actor_base)
        if not cfg.ok:
            t.error = f"hermes config write failed: {cfg.stderr[:200]}"
            return t

        multiturn = observer is not None and questioner is not None
        # Pre-assigned persona (from queries.jsonl) > random
        pinned_name = task.get("persona_name", "random")
        if multiturn and pinned_name not in ("", "random"):
            target = next((p for p in PERSONAS if p.name == pinned_name), None)
            persona = target if target is not None else sample_persona(rng)
        else:
            persona = sample_persona(rng) if multiturn else None
        if persona is not None:
            t.persona_name = persona.name

        patience_tracker = PatienceTracker(persona, rng) if (multiturn and persona) else None

        # Baseline AFTER workspace upload + hermes config write, so seed files
        # and hermes runtime files are in the baseline and don't show up as
        # "added" on turn 1. Only actor's own changes during the session count.
        baseline = observer.snapshot(sb) if multiturn else None
        turn = 0
        ended_by = "k_budget"
        cur_query: str | None = query
        session_sid: str | None = sid  # use sandbox_id as stable session key across turns

        while cur_query is not None:
            turn += 1
            aturn = actor_obj.run_turn(
                sb, cur_query,
                model=actor_model, base=actor_base,
                max_turns=hermes_max_turns, timeout=slot_timeout,
                resume_sid=session_sid,
                session_id=session_sid,
            )
            if aturn.session_id:
                session_sid = aturn.session_id
            ok = aturn.ok
            # Derive stdout/stderr for the legacy downstream checks. For hermes_cli
            # the parent messages are [user, assistant(stdout), (system[stderr])].
            _asst = next((m for m in aturn.messages if m.get("role") == "assistant"), None)
            stdout = (_asst or {}).get("content", "") if _asst else ""
            stderr = aturn.error or ""
            all_messages.extend(aturn.messages)
            # Accumulate system_prompt / tools from the FIRST turn only (same across turns).
            if turn == 1:
                t.system_prompt = aturn.system_prompt or ""
                t.tools = aturn.tools or []
                t.api_calls = aturn.api_calls
                t.partial = aturn.partial
            # Accumulate any sub-agent (delegate_task) child trajectories this turn.
            for _ch in aturn.children:
                t.children.append(
                    {"turn": turn, "task_index": _ch.task_index,
                     "goal": _ch.goal, "messages": _ch.messages,
                     "system_prompt": _ch.system_prompt,
                     "base_system_prompt": _ch.base_system_prompt,
                     "tools": _ch.tools}
                )

            # Hard failure on first turn: session-ending ONLY when the actor
            # truly produced nothing (no structured messages at all). Hermes
            # hitting max_iterations without a final summary produces 60+
            # messages of useful tool_calls but ok=False — that's a partial
            # completion, not a hard failure worth discarding.
            if turn == 1 and not ok and len(all_messages) <= 1:
                t.error = stderr[:200] or "hermes produced no output"
                ended_by = "agent_error"
                break

            if not multiturn:
                ended_by = "k_budget"
                break

            # Determine if this turn failed (no output, or error)
            turn_failed = (not ok and not stdout)

            # observer: diff-driven objective report (state only, no judge).
            report = None
            try:
                post = observer.snapshot(sb)
                report = observer.observe(sb, actor_trajectory=all_messages, baseline=baseline, post=post)
                baseline = post
            except Exception as exc:  # noqa: BLE001
                turn_failed = True
                all_messages.append({"role": "system", "content": f"[observer_error] {exc}"})

            # Record observer report for debugging (smoke + real collection).
            _rep_row = None
            if report is not None:
                _rep_row = {
                    "turn": turn,
                    "query": cur_query,
                    "has_effect": getattr(report, "has_effect", None),
                    "final": getattr(report, "final", None),
                    "intermediate": getattr(report, "intermediate", None),
                    "file_tree": getattr(report, "file_tree", ""),
                    "state_diff": getattr(report, "state_diff", ""),
                    "discrepancies": getattr(report, "discrepancies", ""),
                    "has_red_flag": getattr(report, "has_red_flag", False),
                }
                t.observer_reports.append(_rep_row)
            else:
                _rep_row = {"turn": turn, "query": cur_query, "observer_error": True}
                t.observer_reports.append(_rep_row)

            # Persist observer report to a SEPARATE debug file, indexed by
            # session_id (sandbox_id) + turn, so the report can be located
            # by "which session, which turn was asked". Appended per turn so
            # it survives even if the session later crashes.
            if observer_log is not None:
                try:
                    line = {"session_id": sid, "record_id": record_id,
                            "persona": t.persona_name, "bucket": bucket, **_rep_row}
                    with open(observer_log, "a", encoding="utf-8") as _f:
                        _f.write(json.dumps(line, ensure_ascii=False) + "\n")
                except Exception:  # noqa: BLE001 -- logging must never break collection
                    pass

            if turn_failed and patience_tracker is not None:
                # Persona-driven redo: observer → questioner → natural retry query.
                if patience_tracker.on_failure():
                    try:
                        nxt = questioner.next_query(persona, report, all_messages) if report else None
                    except Exception:  # noqa: BLE001
                        nxt = None
                    if nxt is None:
                        ended_by = "patience_exhausted"
                        break
                    cur_query = nxt  # redo with persona-voiced retry
                    continue
                ended_by = "patience_exhausted"
                break

            # Only Questioner controls the loop: follow-up → continue; None → stop.
            # Satisfied persona stops; unsatisfied asks more. No budget / patience / randomness.
            nxt = questioner.next_query(persona, report, all_messages) if report else None
            if nxt is None:
                ended_by = "end_session" if not questioner.last_query_was_error else "questioner_error"
                break
            cur_query = nxt

        t.messages = all_messages
        t.num_turns = turn
        t.ended_by = ended_by
        # answer = last non-empty line of the last assistant message.
        last_asst = next((m["content"] for m in reversed(all_messages) if m["role"] == "assistant"), "")
        t.answer = [l for l in last_asst.splitlines() if l.strip()][-1] if last_asst.strip() else ""
    except Exception as exc:  # noqa: BLE001
        t.error = (t.error + " | " if t.error else "") + f"{type(exc).__name__}: {exc}"
        t.messages = all_messages
    finally:
        try:
            sb.kill()
        except Exception:  # noqa: BLE001
            pass

    t.query_index = qi
    t.slot_idx = 0
    t.sandbox_id = sid
    t.bucket = bucket
    # Failure-mode QC (P3): audit the structured trajectory + sub-agent children.
    # Only meaningful for structured actors (tool_calls present); on flattened
    # trajectories the tool checks are inert. Marks qc_hard for the write path to
    # drop (HARD = tool hallucination / truncation / loop / empty-turn / XML leak).
    if not t.error:
        try:
            from scripts.qc_trajectory import audit_trajectory

            _qc = audit_trajectory({"messages": t.messages, "children": t.children})
            t.qc_hard = _qc["hard"]
            t.qc_codes = _qc["codes"]
        except Exception:  # noqa: BLE001 — QC must never break collection
            pass
    return t


# --------------------------------------------------------------------------- #
# The 8-way GRPO session driver                                                #
# --------------------------------------------------------------------------- #


def run_session(
    *,
    tasks: list[dict[str, Any]],
    actor: str,
    mode: str = "grpo",
    slots: int = 8,
    backend: str = "e2b",
    template: str = "agentic-cl-sandbox",
    actor_model: str = "gpt-5.1",
    actor_base: str = "",
    max_turns: int = 8,
    hermes_max_turns: int = 50,
    slot_timeout: int = 180,
    out_dir: Path = Path("rollouts/grpo"),
    max_concurrent: int = 1,
) -> dict[str, Any]:
    """Run one session over `tasks`.

    When ``max_concurrent <= 1`` (default), runs queries sequentially with
    sandbox reuse (original GRPO path).  When ``max_concurrent > 1``, runs
    queries in parallel — each query gets its own sandbox lifecycle (spawn →
    upload workspace → hermes → destroy) — suitable for cold-start collection
    where each task has different workspace files.
    """
    # ── parallel path (cold-start collection) ──────────────────────────
    if max_concurrent > 1:
        if actor != "hermes":
            raise ValueError("parallel mode requires --actor hermes")

        # Dev-side agents for multi-turn (observer/questioner).
        from agents.observer import Observer
        from agents.questioner import Questioner
        _observer = Observer(use_llm=False)  # deterministic, no LLM cost
        _questioner = Questioner()

        total = len(tasks)
        done = 0
        t0 = time.time()
        all_rows: list[SlotTrajectory] = []

        def _run_one(task_idx: int) -> SlotTrajectory:
            nonlocal done
            t = _run_one_collect_query(
                tasks[task_idx], task_idx,
                actor=actor, actor_model=actor_model, actor_base=actor_base,
                max_turns=max_turns, hermes_max_turns=hermes_max_turns,
                slot_timeout=slot_timeout,
                backend=backend, template=template,
                observer=_observer, questioner=_questioner,
            )
            done += 1
            if done % max(1, total // 20) == 0:
                print(f"[collect] {done}/{total} queries done ({done/max(1e-9,time.time()-t0):.1f}/s)", flush=True)
            return t

        print(f"[collect] parallel mode: {total} queries, {max_concurrent} concurrent", flush=True)
        with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
            all_rows = list(ex.map(_run_one, range(total)))
        all_rows = [r for r in all_rows if r is not None]

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"grpo_{actor}.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for t in all_rows:
                fh.write(t.to_jsonl() + "\n")

        ok = sum(1 for t in all_rows if not t.error)
        err = sum(1 for t in all_rows if t.error)
        elapsed = time.time() - t0
        print(f"[collect] DONE {total} queries in {elapsed:.0f}s ({total/max(1,elapsed):.1f}/s) ok={ok} err={err}", flush=True)
        summary = {
            "actor": actor, "mode": mode, "max_concurrent": max_concurrent,
            "num_queries": total, "ok": ok, "errors": err,
            "out_file": str(out_file),
        }
        manifest = out_dir / "manifest.json"
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return summary

    # ── sequential path (GRPO / single-query debug) ────────────────────
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
        sb = make_sandbox(backend, template=template, timeout=10800)   # 3h lifetime; per-cmd ctrl
        return sb

    with ThreadPoolExecutor(max_workers=min(slots, 8)) as ex:
        sandbox_specs = list(ex.map(_spawn, range(slots)))
    print(f"[grpo] {slots} sandboxes up in {time.time()-t_spawn:.1f}s: "
          f"{[getattr(s,'_sandbox_id','?')[:12] for s in sandbox_specs]}", flush=True)

    try:
        for qi, task in enumerate(tasks):
            query = task["query"]
            record_id = task.get("record_id", "")
            bucket = task.get("bucket", "")
            expected = task.get("expected")
            t_q = time.time()
            print(f"\n[grpo] === query {qi+1}/{len(tasks)}: {query[:70]} ===", flush=True)

            # 8 parallel rollouts. _rollout takes qi/query/task explicitly so the
            # closure does not capture the loop variable (ruff B023).
            def _rollout(slot_idx: int, _qi: int = qi, _query: str = query, _task: dict[str, Any] = task, _rid: str = record_id, _bucket: str = bucket) -> SlotTrajectory:
                sb = sandbox_specs[slot_idx]
                try:
                    if actor == "hermes":
                        if _rid:
                            n = _upload_workspace(sb, _rid)
                            if n:
                                print(f"  slot{slot_idx}: uploaded {n} workspace files for {_rid}", flush=True)
                        t = _run_hermes_slot(sb, _query, actor_model, actor_base, hermes_max_turns, slot_timeout)
                    else:
                        t = _run_run_code_slot(sb, _task, slot_timeout)
                except Exception as exc:  # noqa: BLE001 -- isolate slot failures
                    t = SlotTrajectory(query_index=_qi, slot_idx=slot_idx,
                                       sandbox_id=getattr(sb, "_sandbox_id", ""), error=f"{type(exc).__name__}: {exc}")
                t.query_index = _qi
                t.slot_idx = slot_idx
                t.bucket = _bucket
                return t

            with ThreadPoolExecutor(max_workers=slots) as ex:
                trajs = list(ex.map(_rollout, range(slots)))
            trajs.sort(key=lambda t: t.slot_idx)

            # reward (skip in collect mode)
            if mode == "collect":
                for t in trajs:
                    t.reward = None
                rewards = [0.0] * len(trajs)
            else:
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

            # GRPO advantage + winner (skip in collect mode)
            if mode == "collect":
                for t in trajs:
                    t.advantage = None
                widx = -1  # no winner
            else:
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
                if mode == "collect":
                    print(f"  slot{t.slot_idx}: ans={t.answer[:40]!r} sid={t.sandbox_id[:12]}"
                          f"{' ERR='+t.error[:40] if t.error else ''}", flush=True)
                else:
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
        "mode": mode,
        "slots": slots,
        "num_queries": len(tasks),
        "winners": winners if mode != "collect" else [],
        "per_query_rewards": per_query_rewards,
        "mean_reward": sum(r for t in all_rows for r in [t.reward or 0.0]) / max(1, len(all_rows)) if mode != "collect" else None,
        "out_file": str(out_file),
    }
    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _load_queries(queries_path: str, num_queries: int | None) -> list[dict[str, Any]]:
    """Load tasks from a queries JSONL (output of taskspec_to_queries.py).

    Each line: {"record_id": "...", "queries": ["q1", ...]}. Takes queries[0] as
    the seed query; record_id is stashed in the task dict for traceability.
    """
    tasks: list[dict[str, Any]] = []
    with open(queries_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qs = obj.get("queries") or []
            if not qs:
                continue
            tasks.append({
                "query": qs[0],
                "record_id": obj.get("record_id", ""),
                "bucket": obj.get("bucket", ""),
                "persona_name": obj.get("persona_name", "random"),
            })
            if num_queries and len(tasks) >= num_queries:
                break
    return tasks


def _load_tasks(tasks_path: str | None, num_queries: int) -> list[dict[str, Any]]:
    """Load tasks from a JSONL file; raises if no path is given."""
    if not tasks_path:
        raise SystemExit("ERROR: --tasks <jsonl> or --queries <jsonl> is required (no built-in defaults).")
    tasks = []
    with open(tasks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks[:num_queries] if num_queries else tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--actor", choices=["hermes", "run_code"], default="run_code",
                    help="run_code = framework smoke (no model); hermes = real in-sandbox actor")
    ap.add_argument("--mode", choices=["grpo", "collect"], default="grpo",
                    help="grpo = full judge+winner; collect = pure trajectory collection (no judge/winner)")
    ap.add_argument("--num-queries", type=int, default=0, help="0=all queries; N=first N queries only")
    ap.add_argument("--slots", type=int, default=8, help="parallel sandboxes per query (GRPO group size)")
    ap.add_argument("--backend", default="e2b", choices=["e2b", "local"])
    ap.add_argument("--template", default="agentic-cl-sandbox")
    ap.add_argument("--tasks", help="JSONL of {query[, expected]} per line (required unless --queries given)")
    ap.add_argument("--queries", help="JSONL of {record_id, queries:[q1,...]} per line (collect mode; takes queries[0])")
    ap.add_argument("--actor-model", default="gpt-5.1", help="hermes model name (hermes actor)")
    ap.add_argument("--actor-base", default="", help="override AGENT_MODEL_BASE (else runtime env)")
    ap.add_argument("--max-turns", type=int, default=8, help="session turn cap (questioner rounds per session)")
    ap.add_argument("--hermes-max-turns", type=int, default=50,
                    help="hermes internal ReAct loop cap (per invocation)")
    ap.add_argument("--slot-timeout", type=int, default=180, help="per-slot hermes timeout (s)")
    ap.add_argument("--max-concurrent", type=int, default=1,
                    help="parallel queries (cold-start mode: each query = own sandbox lifecycle)")
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

    # actor_base / actor_model: CLI override > runtime.env injected into sandbox.
    # The model KEY is read INSIDE the sandbox (AGENT_MODEL_KEY from envVars),
    # never on the dev machine — so we don't validate it here.
    actor_base = args.actor_base or _env("AGENT_MODEL_BASE")
    if args.actor == "hermes" and not actor_base:
        print("[grpo] ERROR: hermes actor needs AGENT_MODEL_BASE "
              "(set in docker/sandbox/runtime.env, injected into the sandbox at create).",
              file=sys.stderr)
        sys.exit(2)

    tasks: list[dict[str, Any]]
    if args.queries:
        tasks = _load_queries(args.queries, args.num_queries or None)
        print(f"[grpo] loaded {len(tasks)} queries from {args.queries}", flush=True)
    else:
        tasks = _load_tasks(args.tasks, args.num_queries)
    print(f"[grpo] actor={args.actor} mode={args.mode} slots={args.slots} num_queries={len(tasks)} "
          f"backend={args.backend} template={args.template}", flush=True)
    if args.actor == "hermes":
        print(f"[grpo] in-sandbox hermes -> model={args.actor_model} base={actor_base}", flush=True)

    t0 = time.time()
    summary = run_session(
        tasks=tasks,
        actor=args.actor,
        mode=args.mode,
        slots=args.slots,
        backend=args.backend,
        template=args.template,
        actor_model=args.actor_model,
        actor_base=actor_base,
        max_turns=args.max_turns,
        hermes_max_turns=args.hermes_max_turns,
        slot_timeout=args.slot_timeout,
        out_dir=Path(args.out_dir),
        max_concurrent=args.max_concurrent,
    )
    print(f"\n[grpo] DONE in {time.time()-t0:.0f}s")
    if args.mode == "collect":
        print(f"  trajectories -> {summary['out_file']}")
    else:
        print(f"  winners per query: {summary['winners']}")
        print(f"  mean reward: {summary['mean_reward']:.3f}")
    print(f"  manifest     -> {Path(args.out_dir)/'manifest.json'}")


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")


if __name__ == "__main__":
    main()
