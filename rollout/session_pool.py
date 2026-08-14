"""Session-level sandbox orchestration (Gap C; doc/sandbox/Sandbox_管理调度指南.md).

A SessionSandboxPool runs ONE ``queries`` session:

    master template ──spawn 8 slots (identical start)──┐
       Query 1: 8 parallel rollouts → reward → winner ─┤ sync 8 slots → winner
       Query 2: 8 parallel rollouts → reward → winner ─┤ sync ...
       ...                                              │
       session end: destroy 8 slots (master untouched) ─┘

Design contracts (from the scheduling guide):
  - All 8 slots start bit-identical (same master + same persona/seed).
  - Within a query the 8 slots diverge (advantage variance source); we sync ONLY
    at query boundaries.
  - After each query: pick winner (with tie / all-fail / scorer-error fallback),
    copy winner disk state AND conversation history to all slots (§3 ① ②).
  - Session end: discard everything; the global master is never rewritten (§规则2).

Backend-agnostic + verl/Ray-free so it unit-tests with mocks. The real Tencent
winner-sync (pause→fork, §6 D1) is injected via ``sync_fn``; the default mock
copies the winner's in-memory state to every slot.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rollout.sandbox_client import make_sandbox, select_winner


@dataclass
class Trajectory:
    """One slot's rollout of one query (buffer-bound; fields mirror trajectory_adapter)."""

    slot_idx: int
    trajectory_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    reward: float | None = None
    success: bool | None = None
    response_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    bucket: str | None = None
    next_state: Any = None  # slot disk state AFTER this rollout (for winner sync)
    meta: dict[str, Any] = field(default_factory=dict)


# agent_fn(client, query, state, slot_idx, history) -> Trajectory
# ``history`` is the session 正史 = concatenation of prior winners' messages
# (§3 ①); the agent MUST prepend it so query_{k+1}'s prompt matches the winner's
# evolved environment, not each slot's own losing trajectory.
AgentFn = Callable[..., Trajectory]
# sync_fn(slots, winner_state) -> None  (mutate slot states to winner's)
SyncFn = Callable[[list["_Slot"], Any], None]


@dataclass
class _Slot:
    idx: int
    client: Any
    state: Any = None  # logical disk state (mock: arbitrary python object)
    alive: bool = True


def select_winner_with_fallback(
    rewards: Sequence[float | None],
    trajectory_ids: Sequence[str] | None,
    rng: random.Random,
) -> int | None:
    """Winner index with the guide's boundary rules (§3 ③).

    - any reward is None (scorer error)        -> None (caller keeps prev state)
    - all rewards equal (incl. all-zero/fail)  -> random slot (seeded)
    - otherwise                                -> argmax (ties by trajectory_id)
    """
    if not rewards:
        return None
    if any(r is None for r in rewards):
        return None
    vals = [float(r) for r in rewards]
    if max(vals) == min(vals):
        return rng.randrange(len(vals))
    ids = list(trajectory_ids) if trajectory_ids is not None else None
    return select_winner(vals, ids)


def _default_sync(slots: list[_Slot], winner_state: Any) -> None:
    """Copy winner state to every slot (deep copy = independent).

    This is the CORRECT single-turn behavior: after picking the winner, all 8
    slots are aligned to the winner's state so the next turn (if any) starts
    bit-identical. No slot is killed mid-session -- they all stay alive carrying
    the winner's state, and are destroyed together at session end
    (``destroy_all``). The real-backend contract (doc/ops/sandbox/
    Sandbox_管理调度指南.md §6) is the same in spirit: the winner is the
    session's only live state carrier; losers are derived from it, not kept as
    independent divergent states. Killing all 8 mid-session (the abandoned
    "D2 杀重建" path) drops process/in-memory state subsequent turns depend on
    and only happens at session end.
    """
    for s in slots:
        s.state = copy.deepcopy(winner_state)


class SessionSandboxPool:
    def __init__(
        self,
        master_template: str = "agentic-cl-sandbox",
        slots: int = 8,
        *,
        backend: str = "local",
        persona: str | None = None,
        fs_seed: str | None = None,
        sandbox_factory: Callable[..., Any] = make_sandbox,
        sync_fn: SyncFn | None = None,
        initial_state_factory: Callable[[], Any] | None = None,
        max_workers: int | None = None,
        seed: int = 0,
    ) -> None:
        self.master_template = master_template
        self.n_slots = slots
        self.backend = backend
        self.persona = persona
        self.fs_seed = fs_seed
        self._factory = sandbox_factory
        self._sync_fn = sync_fn or _default_sync
        self._initial_state = initial_state_factory or (lambda: {})
        self._max_workers = max_workers or slots
        self._rng = random.Random(seed)
        self._slots: list[_Slot] = []
        self.session_history: list[dict[str, Any]] = []  # winner trajectory messages, concatenated
        self.query_index = 0

    # --- lifecycle ---------------------------------------------------------
    def spawn(self) -> None:
        """Derive ``n_slots`` instances from the master (identical start)."""
        if self._slots:
            raise RuntimeError("pool already spawned; call destroy_all() first")
        init = self._initial_state()
        self._slots = [
            _Slot(idx=i, client=self._make_client(), state=copy.deepcopy(init)) for i in range(self.n_slots)
        ]

    def _make_client(self) -> Any:
        return self._factory(
            self.backend,
            template=self.master_template,
            persona=self.persona,
            fs_seed=self.fs_seed,
        )

    def destroy_all(self) -> None:
        """Kill every slot. The global master Tool is NEVER rewritten (§规则2)."""
        for s in self._slots:
            try:
                if s.alive and hasattr(s.client, "kill"):
                    s.client.kill()
            finally:
                s.alive = False
        self._slots = []

    # --- per-query ---------------------------------------------------------
    def run_query(self, query: str, agent_fn: AgentFn) -> list[Trajectory]:
        """Run the 8 slots in parallel on ``query`` from their current state."""
        if not self._slots:
            raise RuntimeError("pool not spawned")
        history = list(self.session_history)

        def _run(slot: _Slot) -> Trajectory:
            try:
                # feed the winner-derived session history so every slot's prompt
                # starts from the SAME 正史 (§3 ①), not its own past trajectory
                traj = agent_fn(slot.client, query, slot.state, slot.idx, history)
            except Exception as exc:  # noqa: BLE001 -- isolate slot failures
                traj = Trajectory(
                    slot_idx=slot.idx,
                    trajectory_id=f"q{self.query_index}-s{slot.idx}",
                    reward=None,
                    meta={"error": str(exc)},
                )
            traj.slot_idx = slot.idx
            if not traj.trajectory_id:
                traj.trajectory_id = f"q{self.query_index}-s{slot.idx}"
            traj.meta.setdefault("session_history_len", len(history))
            return traj

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            trajs = list(ex.map(_run, self._slots))
        trajs.sort(key=lambda t: t.slot_idx)
        # stash each slot's resulting state for a potential sync
        for slot, t in zip(self._slots, trajs, strict=True):
            if t.next_state is not None:
                slot.state = t.next_state
        return trajs

    def pick_winner(self, trajs: Sequence[Trajectory]) -> int | None:
        return select_winner_with_fallback(
            [t.reward for t in trajs],
            [t.trajectory_id for t in trajs],
            self._rng,
        )

    def sync_to_winner(self, winner_idx: int, trajs: Sequence[Trajectory]) -> None:
        """Align all 8 slots' disk state AND conversation history to the winner."""
        winner = trajs[winner_idx]
        self._sync_fn(
            self._slots, winner.next_state if winner.next_state is not None else winner.meta.get("state")
        )
        # conversation正史 = winner trajectory (§3 ①): losers drop out of history
        self.session_history.extend(winner.messages)

    def run_checkers(self, checkers: Sequence[dict], winner_idx: int) -> dict[str, bool]:
        """Run sandbox-type checkers on the winner BEFORE teardown (Gap A.4).

        Default mock evaluates file_exists / sandbox_assert against the winner's
        in-memory state dict. Real backend would exec in the winner instance.
        """
        results: dict[str, bool] = {}
        if not self._slots:
            return results
        winner_state = self._slots[winner_idx].state if winner_idx < len(self._slots) else {}
        files = winner_state.get("files", {}) if isinstance(winner_state, dict) else {}
        for i, chk in enumerate(checkers):
            ctype = chk.get("type")
            if ctype == "file_exists":
                results[str(i)] = chk.get("path") in files
            elif ctype == "sandbox_assert":
                results[str(i)] = bool(winner_state.get("asserts", {}).get(chk.get("id"), False))
        return results

    # --- full session ------------------------------------------------------
    def run_session(
        self,
        queries: Sequence[str],
        agent_fn: AgentFn,
    ) -> list[Trajectory]:
        """Run a whole session: spawn → per-query rollout+winner+sync → destroy.

        Returns ALL slot trajectories across all queries (route into the buffer).
        """
        self.spawn()
        all_trajs: list[Trajectory] = []
        try:
            for q in queries:
                trajs = self.run_query(q, agent_fn)
                all_trajs.extend(trajs)
                winner = self.pick_winner(trajs)
                self.query_index += 1
                if winner is None:
                    # scorer error / no winner: keep previous boundary state, no sync
                    continue
                self.sync_to_winner(winner, trajs)
        finally:
            self.destroy_all()
        return all_trajs
