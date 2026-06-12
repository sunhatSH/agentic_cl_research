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

    One report, two consumers (§3.3 要点 3): it is fed to BOTH the Questioner
    (to author the next query) and the Reward judge (to score on real effect).
    The report is driven by what the actor *claimed* it did (``actor_claims``)
    rather than a fixed snapshot template, so intermediate artifacts (easily
    overwritten by later steps) are captured before they vanish (§3.3 要点 2).
    """

    intermediate: list[dict] = field(default_factory=list)
    """Intermediate results: {desc, source(cmd/file), value_excerpt}."""

    final: list[dict] = field(default_factory=list)
    """Final deliverables: {path, kind, content_excerpt}."""

    actor_claims: str = ""
    """What the actor stated it did, in the winner trajectory (observation lead)."""

    discrepancies: str = ""
    """Claimed-vs-actual gaps (anti-hacking evidence for the judge); may be empty."""

    file_tree: str = ""
    """Winner workspace file tree (depth-truncated; fallback evidence)."""

    def is_empty(self) -> bool:
        """True when the observer found no usable evidence.

        Drives the failure / patience path (§3.6.5): an empty report after a
        winner rollout means the response failed / stopped / produced nothing.
        """
        return not (self.intermediate or self.final or self.actor_claims.strip())


@dataclass
class Persona:
    """One user persona (§7.5). 16 of these live in ``agents/personas.py``.

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
