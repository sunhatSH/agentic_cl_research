"""16 fixed user personas for the Questioner agent (doc §3.5 / §7.5 / O2).

The persona DATA lives in ``agents/personas.json`` (config decoupled from code);
this module only loads + validates it and exposes the same public surface as
before (``PERSONAS`` / ``sample_persona`` / ``DEFAULT_PATIENCE_DECAY``). Edit the
JSON to add/tune personas -- no code change needed. Full reference table:
``doc/UserSim_人设库.md``.

A session draws ONE persona at random and keeps it fixed (session-level), which
is one of the three anti-mode-collapse mechanisms (§3.5). Each persona carries:
  - profession / preference / profile / observation_focus  (voice + what it stresses)
  - patience (P0) + patience_decay (d0)                    (failure path, §3.6.5)

Two patience axes are deliberately decoupled (§3.6.5):
  P0 = initial tolerance (willingness to give chances),
  d0 = escalation speed (temper). Retries-to-give-up ≈ log2(P0/d0).

This is pure data + a seeded sampler, verl-free, unit-testable. The first three
seed the same workspaces as docker/sandbox/fs-seeds (finance / sysops / office);
the rest broaden coverage across the 7 capability buckets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from agents.schema import Persona

_CONFIG_PATH = Path(__file__).with_name("personas.json")

_PERSONA_FIELDS = (
    "name",
    "profession",
    "preference",
    "profile",
    "observation_focus",
    "patience",
)


def _load_personas(path: Path = _CONFIG_PATH) -> tuple[list[Persona], float]:
    """Load + validate the persona library from JSON.

    Returns (personas, default_patience_decay). A persona may omit
    ``patience_decay``; it then falls back to the file-level default (§3.6.5).
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    default_decay = float(raw.get("default_patience_decay", 0.1))
    personas: list[Persona] = []
    for i, entry in enumerate(raw["personas"]):
        missing = [k for k in _PERSONA_FIELDS if k not in entry]
        if missing:
            raise ValueError(f"persona #{i} ({entry.get('name', '?')}) missing fields: {missing}")
        personas.append(
            Persona(
                name=entry["name"],
                profession=entry["profession"],
                preference=entry["preference"],
                profile=entry["profile"],
                observation_focus=entry["observation_focus"],
                patience=float(entry["patience"]),
                patience_decay=float(entry.get("patience_decay", default_decay)),
            )
        )
    return personas, default_decay


PERSONAS, DEFAULT_PATIENCE_DECAY = _load_personas()

assert len(PERSONAS) == 16, "persona library must have exactly 16 entries (doc §3.5)"


def sample_persona(rng: random.Random) -> Persona:
    """Draw one persona for a session (seeded for reproducibility, §3.6.5 实现注记)."""
    return rng.choice(PERSONAS)
