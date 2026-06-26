"""Rollout scheduler: run N sessions in parallel, each an 8-slot GRPO group (Gap C).

One training step = ``sessions_per_step`` (default 16) ``queries`` sessions run
in parallel; each session is a SessionSandboxPool (8 slots, sequential queries
with winner-sync between them). Total concurrent instances = sessions × slots
(default 16 × 8 = 128). Sessions are independent: no cross-session sync.

This layer owns the 16×8 + winner-sync ORCHESTRATION. Per-step generation is
delegated to ``agent_fn`` (in real training: a thin wrapper over verl's rollout
generate / agent_loop so token+logprob come from the framework natively — see
doc/sandbox/Sandbox_Agent架构.md §3, NOT an HTTP proxy).
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
    ) -> None:
        self.agent_fn = agent_fn
        self.sessions_per_step = sessions_per_step
        self.slots = slots
        self.backend = backend
        self.master_template = master_template
        self._pool_factory = pool_factory or self._default_pool
        self._max_session_workers = max_session_workers or sessions_per_step
        self.seed = seed

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
        trajs = pool.run_session(spec.queries, self.agent_fn)
        for t in trajs:
            t.meta.setdefault("session_id", spec.session_id)
            # namespace per session so ids are globally unique (avoid buffer overwrite)
            t.trajectory_id = f"{spec.session_id}-{t.trajectory_id}"
        return trajs

    def run_step(self, specs: Sequence[SessionSpec]) -> list[Trajectory]:
        """Run up to ``sessions_per_step`` sessions in parallel; collect all trajectories."""
        batch = list(specs)[: self.sessions_per_step]
        results: list[list[Trajectory]] = [[] for _ in batch]

        def _run(args: tuple[int, SessionSpec]) -> tuple[int, list[Trajectory]]:
            i, spec = args
            return i, self.run_session(spec, self.seed + i)

        with ThreadPoolExecutor(max_workers=self._max_session_workers) as ex:
            for i, trajs in ex.map(_run, enumerate(batch)):
                results[i] = trajs

        out: list[Trajectory] = []
        for trajs in results:
            out.extend(trajs)
        return out
