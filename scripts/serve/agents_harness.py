"""Multi-agent harness: run the observer + questioner + reward loop end-to-end.

This is the single OFFLINE entry point for exercising the three user-sim agents
together (without a GPU, without the cluster, without real model endpoints):

    seed query -> actor ReAct (in a sandbox) -> winner
                    -> Observer  -> ObservationReport R_t
                          -> Reward(R_t)        (score the winner turn)
                          -> Questioner(R_t)    (next query / <end_session>)

It drives the REAL session drivers (rollout/simulated_session, rollout/usersim_collect)
and the REAL agent classes (agents/observer, questioner, reward) -- only the model
endpoints are mocked by default, so the wiring you see here is the wiring used in
training. Use it to:
  - eyeball the per-turn trace of all three agents on a dev box;
  - validate a backend swap (``--backend local|e2b|aliyun``);
  - point the sim-agents at real endpoints (``--real``) while keeping a scripted
    actor + sandbox, to smoke-test prompts/judge without a full rollout.

Examples
--------
    python scripts/serve/agents_harness.py                       # 8-slot, all 3 agents, mock
    python scripts/serve/agents_harness.py --mode collect        # 1-slot, observer+questioner
    python scripts/serve/agents_harness.py --backend e2b --real  # real sandbox + real sim-agents

The actor here is intentionally a SCRIPTED ReAct stand-in (in training the actor is
the policy under verl); the harness is about the observer/questioner/reward agents,
not the actor policy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.observer import Observer
from agents.personas import PERSONAS, sample_persona
from agents.questioner import END_SESSION, Questioner
from agents.schema import Persona
from rollout.collect import GenStep, make_react_agent_fn
from rollout.session_pool import SessionSandboxPool, Trajectory
from rollout.simulated_session import run_simulated_session
from rollout.usersim_collect import run_usersim_session

# --------------------------------------------------------------------------- #
# Mock model endpoints (default). Each mirrors the ChatClient / judge surface  #
# the real agents call, so swapping to --real changes only WHERE they resolve. #
# --------------------------------------------------------------------------- #


class ScriptedChat:
    """Deterministic stand-in for an OpenAI-compatible chat endpoint.

    Pops ``replies`` in order; once exhausted returns ``default``. Records the
    messages it was called with (handy when eyeballing the prompt assembly).
    """

    def __init__(self, replies: list[str] | None = None, default: str = ""):
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else self.default


def _mock_observer_report() -> str:
    """A plausible, structured ObservationReport JSON (what the real model emits)."""
    return json.dumps(
        {
            "intermediate": [{"desc": "Q3 total", "source": "calc", "value_excerpt": "123"}],
            "final": [{"path": "report.txt", "kind": "text", "content_excerpt": "Q3 total = 123"}],
            "discrepancies": "",
            "file_tree": "report.txt",
        }
    )


class MockJudge:
    """Stand-in for trainer.model_reward.JudgeClient (the reward agent's backend)."""

    def __init__(self, verdict: dict[str, float] | None = None):
        self.verdict = verdict or {"completion": 0.8, "safety": 1.0, "robustness": 0.9}

    def score(self, *, task: str, trajectory: str, rubric: str, data_source: str) -> dict[str, float]:
        return dict(self.verdict)


# --------------------------------------------------------------------------- #
# Scripted actor: a real ReAct loop (rollout.collect) driven by a mock generate #
# --------------------------------------------------------------------------- #


def _mock_generate(messages: list[dict[str, Any]]) -> GenStep:
    """Two-step ReAct: write a file via a tool call, then give a final answer.

    Stateless heuristic: if the last message is a sandbox observation, finish;
    otherwise emit the tool call. Exercises parse_tool_call -> sandbox.run_code.
    """
    last = str(messages[-1].get("content", "")) if messages else ""
    if "[Sandbox Output]" in last:
        return GenStep(
            text="Done. Wrote report.txt (Q3 total = 123).",
            response_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.1],
            prompt_tokens=40,
            completion_tokens=12,
        )
    code = "open('report.txt', 'w').write('Q3 total = 123'); print('wrote report.txt')"
    payload = json.dumps({"tool": "python", "code": code})
    return GenStep(
        text=f"<toolcall>{payload}</toolcall>",
        response_ids=[1, 2],
        logprobs=[-0.3, -0.2],
        prompt_tokens=30,
        completion_tokens=8,
    )


def _make_scored_agent_fn(backend_runs: bool):
    """Wrap the real ReAct agent_fn and fill in a per-slot reward (Gap A stand-in).

    Reward varies by slot so winner selection is non-degenerate; if the actor's
    sandbox step failed, the reward drops (so the failure/patience path is reachable).
    """
    react = make_react_agent_fn(_mock_generate, max_turns=4, default_bucket="finance")

    def agent_fn(
        client: Any, query: str, state: Any, slot_idx: int, history: list[dict[str, Any]] | None = None
    ) -> Trajectory:
        traj = react(client, query, state, slot_idx, history)
        # deterministic, slot-dependent reward in [0,1]
        traj.reward = round(min(1.0, 0.55 + 0.05 * slot_idx), 3)
        traj.success = traj.reward >= 0.5
        return traj

    return agent_fn


# --------------------------------------------------------------------------- #
# Trace printing                                                               #
# --------------------------------------------------------------------------- #


def _print_report(idx: int, report: Any) -> None:
    print(f"  [turn {idx}] OBSERVER report:")
    print(f"      final        = {report.final}")
    print(f"      intermediate = {report.intermediate}")
    print(f"      discrepancies= {report.discrepancies!r}")
    diff_lines = (report.state_diff or "(none)").splitlines()
    print("      state_diff   = (deterministic sandbox diff -- the ground-truth evidence)")
    for ln in diff_lines[:8]:
        print(f"        | {ln}")
    print(f"      is_empty     = {report.is_empty()}")


def _run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    persona: Persona = (
        next((p for p in PERSONAS if p.name == args.persona), None) if args.persona else sample_persona(rng)
    )
    if persona is None:
        raise SystemExit(f"persona {args.persona!r} not found; options e.g. {[p.name for p in PERSONAS[:5]]}")

    if args.real:
        missing = [v for v in ("USERSIM_API_BASE",) if not os.environ.get(v)]
        if os.environ.get("USERSIM_ENDPOINTS"):
            missing = []  # rotation mode doesn't need USERSIM_API_BASE
        if args.use_llm:
            missing += [v for v in ("OBSERVER_API_BASE",) if not os.environ.get(v)]
        if missing:
            print(f"[harness] --real but missing env {missing}; agents will degrade.")
        observer = Observer(use_llm=args.use_llm)
        questioner, judge = Questioner(), None
    else:
        # Observer LLM is optional; default deterministic forensics (no model call).
        observer = Observer(
            client=ScriptedChat(default=_mock_observer_report()) if args.use_llm else None,
            use_llm=args.use_llm,
        )
        # Questioner: optionally demonstrate RotatingChatClient with mock clients.
        if args.rotation:
            from agents.base import RotatingChatClient

            rotating = RotatingChatClient(
                [
                    ScriptedChat(replies=["Looks good, can you add a Q4 column?"], default=END_SESSION),
                    ScriptedChat(replies=["The numbers seem off, recheck row 5."], default=END_SESSION),
                    ScriptedChat(replies=["Great, now export this as PDF."], default=END_SESSION),
                ],
                rotate_every=2,
            )
            questioner = Questioner(client=rotating)
        else:
            questioner = Questioner(
                client=ScriptedChat(
                    replies=["Looks good, can you also add a Q4 column?"], default=END_SESSION
                )
            )
        judge = MockJudge()

    pool_slots = 8 if args.mode == "simulated" else 1
    pool = SessionSandboxPool(slots=pool_slots, backend=args.backend, seed=args.seed)
    agent_fn = _make_scored_agent_fn(backend_runs=args.backend != "local")

    print(
        f"=== agents_harness | mode={args.mode} backend={args.backend} "
        f"persona={persona.name!r} ({persona.profession}, tone={persona.tone}) seed={args.seed} "
        f"real={args.real} rotation={args.rotation} ==="
    )
    print(f"seed query: {args.seed_query!r}\n")

    if args.mode == "simulated":
        result = run_simulated_session(
            pool,
            args.seed_query,
            agent_fn,
            persona=persona,
            observer=observer,
            questioner=questioner,
            k_max=args.k_max,
            seed=args.seed,
            reward_judge=judge,
        )
    else:
        result = run_usersim_session(
            pool,
            args.seed_query,
            agent_fn,
            persona=persona,
            observer=observer,
            questioner=questioner,
            k_max=args.k_max,
            seed=args.seed,
        )

    for i, report in enumerate(result.reports, start=1):
        _print_report(i, report)
        if i - 1 < len(result.generated_queries):
            print(f"      QUESTIONER -> next query: {result.generated_queries[i - 1]!r}")

    followup_rewards = [t.meta["followup_reward"] for t in result.trajectories if "followup_reward" in t.meta]
    if followup_rewards:
        print(f"\n  REWARD (follow-up turns): {followup_rewards}")

    print(
        f"\n=== summary: persona={result.persona_name!r} turns={result.num_turns} "
        f"ended_by={result.ended_by} reports={len(result.reports)} "
        f"trajectories={len(result.trajectories)} ==="
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--mode",
        choices=["simulated", "collect"],
        default="simulated",
        help="simulated = 8-slot + winner-sync + all 3 agents; collect = 1-slot observer+questioner",
    )
    ap.add_argument("--backend", choices=["local", "e2b", "aliyun"], default="local")
    ap.add_argument(
        "--real",
        action="store_true",
        help="resolve observer/questioner/reward from env (OBSERVER_*/USERSIM_*/REWARD_*) instead of mocks",
    )
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="enable the OPTIONAL observer LLM (default: deterministic forensics, no model call)",
    )
    ap.add_argument(
        "--rotation",
        action="store_true",
        help="demonstrate multi-model rotation with mock clients (anti mode-collapse)",
    )
    ap.add_argument("--seed-query", default="Make a short report of the Q3 total and save it to report.txt.")
    ap.add_argument("--persona", default=None, help="fixed persona name (default: sampled)")
    ap.add_argument("--k-max", type=int, default=3, help="follow-up budget upper bound (K ~ U{1..k_max})")
    ap.add_argument("--seed", type=int, default=0)
    return _run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
