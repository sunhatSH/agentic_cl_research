"""Rollout scheduler: run N sessions in parallel, each an 8-slot GRPO group (Gap C).

One training step = ``sessions_per_step`` (default 16) ``queries`` sessions run
in parallel; each session is a SessionSandboxPool (8 slots, sequential queries
with winner-sync between them). Total concurrent instances = sessions × slots
(default 16 × 8 = 128). Sessions are independent: no cross-session sync.

Two modes:
  static    – each session runs its seed queries through agent_fn (original).
  simulated – each session runs one seed query through the full Questioner +
              Observer loop (run_simulated_session), generating follow-ups online.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rollout.session_pool import AgentFn, SessionSandboxPool, Trajectory


@dataclass
class SessionSpec:
    """One ``queries`` session: ordered queries + the persona for its 8 slots."""

    session_id: str
    queries: list[str]
    persona: str | None = None
    fs_seed: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class RolloutScheduler:
    def __init__(
        self,
        agent_fn: AgentFn,
        *,
        sessions_per_step: int = 16,
        slots: int = 8,
        backend: str = "local",
        master_template: str = "agentic-cl-code-interpreter",
        pool_factory: Callable[..., SessionSandboxPool] | None = None,
        max_session_workers: int | None = None,
        seed: int = 0,
        # ── simulated-session mode ──
        simulated: bool = False,
        observer: Any = None,
        questioner: Any = None,
        k_max: int = 3,
        score_followups: bool = True,
    ) -> None:
        self.agent_fn = agent_fn
        self.sessions_per_step = sessions_per_step
        self.slots = slots
        self.backend = backend
        self.master_template = master_template
        self._pool_factory = pool_factory or self._default_pool
        self._max_session_workers = max_session_workers or sessions_per_step
        self.seed = seed
        self.simulated = simulated
        self.observer = observer
        self.questioner = questioner
        self.k_max = k_max
        self.score_followups = score_followups

    def _default_pool(self, spec: SessionSpec, seed: int) -> SessionSandboxPool:
        return SessionSandboxPool(
            master_template=self.master_template,
            slots=self.slots,
            backend=self.backend,
            persona=spec.persona,
            fs_seed=spec.fs_seed if spec.fs_seed is not None else spec.session_id,
            seed=seed,
        )

    def run_session(self, spec: SessionSpec, seed: int) -> list[Trajectory]:
        pool = self._pool_factory(spec, seed)
        if self.simulated and self.observer is not None and self.questioner is not None:
            trajs = self._run_simulated_session(pool, spec, seed)
        else:
            trajs = pool.run_session(spec.queries, self.agent_fn)
        for t in trajs:
            t.meta.setdefault("session_id", spec.session_id)
            t.trajectory_id = f"{spec.session_id}-{t.trajectory_id}"
        return trajs

    def _run_simulated_session(
        self, pool: SessionSandboxPool, spec: SessionSpec, seed: int
    ) -> list[Trajectory]:
        from rollout.simulated_session import run_simulated_session
        from agents.personas import sample_persona

        rng = __import__("random").Random(seed)
        persona = sample_persona(rng)

        seed_query = spec.queries[0] if spec.queries else ""
        result = run_simulated_session(
            pool=pool,
            seed_query=seed_query,
            agent_fn=self.agent_fn,
            persona=persona,
            observer=self.observer,
            questioner=self.questioner,
            k_max=self.k_max,
            seed=seed,
            score_followups=self.score_followups,
        )
        for t in result.trajectories:
            t.meta.setdefault("persona_name", persona.name)
            t.meta.setdefault("session_turns", result.num_turns)
            t.meta.setdefault("ended_by", result.ended_by)
        return result.trajectories

    def run_step(self, specs: Sequence[SessionSpec]) -> list[Trajectory]:
        """Run up to ``sessions_per_step`` sessions in parallel; collect all trajectories.

        Session-level failures are ISOLATED: if one session raises (bad sandbox,
        observer/snapshot error, sync error), we log it and keep its slot empty
        rather than aborting the whole step. Aborting here (the old ``ex.map``
        behaviour, which re-raises the first exception on iteration) took down
        ``generate_sequences`` -> ``fit()`` for ALL sessions because of one bad
        one. The single-turn contract's "exactly n*8 trajectories" invariant is
        still enforced downstream (cl_rollout_manager asserts the count), so a
        dropped session surfaces as a loud, localised error there -- not a
        mysterious mid-step crash.
        """
        batch = list(specs)[: self.sessions_per_step]
        results: list[list[Trajectory]] = [[] for _ in batch]

        def _run(args: tuple[int, SessionSpec]) -> tuple[int, list[Trajectory]]:
            i, spec = args
            try:
                return i, self.run_session(spec, self.seed + i)
            except Exception as exc:  # noqa: BLE001 -- isolate session failures
                import traceback

                print(
                    f"[rollout] session {spec.session_id} (idx {i}) crashed: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
                # Return a PLACEHOLDER trajectory, NOT [] -- the per-row single-turn
                # contract is exactly 1 trajectory per spec, and generate_sequences
                # asserts len(all_trajs) == len(prompts). Returning [] on a crashed
                # session (e.g. one e2b timeout) drops the row count -> the assert
                # fails -> the WHOLE training step crashes on one bad sandbox. Emit a
                # placeholder (empty messages, reward=None, error meta) so row count
                # is preserved; downstream handles empty messages (prompt falls back
                # to the query, empty response -> reward 0) and pick_winner treats
                # reward=None as scorer-error. This is the row-count half of the
                # failure isolation the exception-catch above only half-did.
                placeholder = Trajectory(
                    slot_idx=0,
                    trajectory_id=f"{spec.session_id}-session-crash",
                    messages=[],
                    reward=None,
                    meta={"error": f"{type(exc).__name__}: {exc}", "session_id": spec.session_id},
                )
                return i, [placeholder]

        with ThreadPoolExecutor(max_workers=self._max_session_workers) as ex:
            for i, trajs in ex.map(_run, enumerate(batch)):
                results[i] = trajs

        out: list[Trajectory] = []
        for trajs in results:
            out.extend(trajs)
        return out
