"""Shared data structures for the user-sim three-agent pipeline.

Pure dataclasses, no verl / Ray / network dependency, so the whole agents
package unit-tests off-GPU. Mirrors the interface contract in
``doc/UserSim_多轮Query在线生成.md`` §7.2 (ObservationReport) and §7.5 (Persona).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObservationReport:
    """Objective, persona-free state report produced by the Observer (§7.2).

    One report, two consumers (§3.3 要点 3): the Questioner (state findings, to
    author the next query) and the Reward judge (state findings to score real
    effect, PLUS the pass-through trajectory to judge process/safety).
    """

    intermediate: list[dict] = field(default_factory=list)
    """Intermediate results: {desc, source(cmd/file), value_excerpt}."""

    final: list[dict] = field(default_factory=list)
    """Final deliverables: {path, kind, content_excerpt}."""

    actor_trajectory: str = ""
    """Raw actor trajectory text -- PASS-THROUGH only.

    Carried by the observer COMPONENT for the reward judge (process/safety), but
    NEVER fed to the observer MODEL (it is not put in the observer prompt -- no
    token waste). The observer's own findings stay state-only; this field is just
    the channel that delivers the trajectory to reward via the one R_t packet.
    """

    discrepancies: str = ""
    """Internal red flags in the produced state (empty/corrupt/contradictory); may be empty."""

    has_red_flag: bool = False
    """Structured verdict: does THIS report carry a genuine, unresolved red flag?

    Set authoritatively by the observer (deterministic checks always know; the LLM
    path sets it explicitly). Consumers (questioner banner / block-end, analyzer)
    key off THIS boolean, NOT keyword-matching ``discrepancies`` free text -- the
    observer routinely opens with a reassuring boilerplate sentence ("No empty
    deliverables detected.") and THEN states a real concern ("One discrepancy is
    present: ..."), which a substring filter would wrongly suppress.
    """

    file_tree: str = ""
    """Winner workspace file tree (depth-truncated; fallback evidence)."""

    state_diff: str = ""
    """Deterministic before/after sandbox diff for this turn (diff-driven evidence).

    Computed by the harness from the sandbox itself (files created/modified/removed
    + content), NOT by the model -- this is the ground truth the observer narrates
    and the reward judge grounds on. Empty when no baseline/sandbox was available.
    """

    has_effect: bool = True
    """Whether THIS turn actually changed the environment (diff non-empty).

    Gating signal (the "几层拦截"): the observer sets this False when the before/after
    FS + system diff is empty -- the turn produced nothing. Downstream uses it to
    SHORT-CIRCUIT expensive calls (skip the reward judge, take the failure path)
    without re-deriving the fact. Defaults True for legacy / no-diff reports.
    """

    def is_empty(self) -> bool:
        """True when the observer found no usable evidence of effect.

        Drives the failure / patience path (§3.6.5): an empty report after a
        winner rollout means the response failed / stopped / produced nothing.

        Evidence precedence (bug fix 2026-07-27): a concrete deliverable in
        ``final`` / ``intermediate`` is GROUND TRUTH and always counts as
        non-empty -- even when the FS/system diff is empty. Text-deliverable
        tasks (QA / reasoning / role-play, ~3 of the 9 buckets) legitimately
        write no files: the observer folds the assistant's reply into ``final``
        with ``has_effect`` possibly False (no env change). The OLD gate short-
        circuited on ``has_effect`` FIRST, so those reports were judged empty ->
        reward forced to 0 for ALL 8 slots -> zero GRPO advantage -> no learning
        signal on entire buckets (silent collapse). ``has_effect`` is only the
        fallback when there is no deliverable at all.
        """
        if self.intermediate or self.final:
            return False
        return not self.has_effect


@dataclass
class Persona:
    """One user persona (§7.5). 42 of these live in ``agents/personas.json``.

    A session draws ONE persona at random and keeps it fixed for the whole
    session (§3.5). The Questioner reads it; the Observer is persona-free and
    does NOT use this.
    """

    name: str
    profession: str
    preference: str
    profile: str
    observation_focus: str
    """整体|细节 × 形式|内容 -- which part of the report this persona stresses."""

    patience: float
    """Initial patience P0 (§3.6.5): willingness to give the agent chances."""

    patience_decay: float
    """Base decrement d0 (§3.6.5): how fast frustration escalates on failure."""

    tone: str = "neutral"
    """How the persona speaks (decoupled from patience): calm | neutral | hot."""
