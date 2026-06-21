"""Questioner agent (persona-driven) + patience mechanism -- doc §3.3 / §3.5 / §3.6.5 / §7.4.

The Questioner reads the objective ObservationReport (NOT the sandbox) plus the
session persona and history, and authors the next user query -- or ends the
session. The patience mechanism (§3.6.5) decides, on a FAILED winner turn,
whether to ask for a redo or give up, as a property of the simulated user.
"""

from __future__ import annotations

import random
from typing import Any

from agents.base import ChatClient, resolve_questioner_client
from agents.personas import DEFAULT_PATIENCE_DECAY
from agents.prompts import build_questioner_prompt
from agents.schema import ObservationReport, Persona

END_SESSION = "<end_session>"


class Questioner:
    """Persona user-agent that generates the next follow-up query (§7.4)."""

    def __init__(self, client: ChatClient | None = None, *, max_tokens: int = 256):
        self._client = client
        self._max_tokens = max_tokens

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = resolve_questioner_client()
        return self._client

    def next_query(
        self,
        persona: Persona,
        report: ObservationReport,
        session_history: list[dict[str, Any]],
    ) -> str | None:
        """Return the next query text, or None == <end_session> (§7.4)."""
        messages = build_questioner_prompt(
            persona=persona, report=report, session_history=session_history
        )
        try:
            raw = self.client.chat(messages, max_tokens=self._max_tokens)
        except Exception:  # noqa: BLE001 -- a failed generation ends the session
            return None
        text = (raw or "").strip()
        if not text or text == END_SESSION or END_SESSION in text:
            return None
        return text


# --------------------------------------------------------------------------- #
# Patience mechanism (§3.6.5) -- pure, seeded, unit-testable                   #
# --------------------------------------------------------------------------- #


class PatienceTracker:
    """Per-session patience over FAILED winner turns (§3.6.5).

    Patience decays with exponentially growing decrement:

        P_k = P_0 - d_0 * (2^k - 1)        (cumulative after k failures)

    Decision after the k-th failure: redo with probability clip(P_k, 0, 1),
    else end the session. P_k < 0 clips to 0 -> always stop. Successful turns
    do not consume patience; redo turns do not count toward the K follow-up
    budget (handled by the caller).
    """

    def __init__(self, persona: Persona, rng: random.Random):
        self.p0 = float(persona.patience)
        self.d0 = float(persona.patience_decay) if persona.patience_decay else DEFAULT_PATIENCE_DECAY
        self.rng = rng
        self.fail_count = 0

    def current_patience(self) -> float:
        """P_k after the failures seen so far (k = fail_count)."""
        k = self.fail_count
        return self.p0 - self.d0 * (2**k - 1)

    def on_failure(self) -> bool:
        """Register a failed turn; return True to REDO, False to end session.

        Returns the seeded probabilistic decision based on the post-decrement
        patience P_k. After P_k goes negative, redo probability is 0 (stop).
        """
        self.fail_count += 1
        pk = self.current_patience()
        redo_prob = max(0.0, min(1.0, pk))
        return self.rng.random() < redo_prob
