"""Questioner agent (persona-driven) + patience mechanism -- doc §3.3 / §3.5 / §3.6.5 / §7.4.

The Questioner reads the objective ObservationReport (NOT the sandbox) plus the
session persona and history, and authors the next user query -- or ends the
session. The patience mechanism (§3.6.5) decides, on a FAILED winner turn,
whether to ask for a redo or give up, as a property of the simulated user.
"""

from __future__ import annotations

import random
from typing import Any

from agents.base import ChatClient, TruncatedOutputError, resolve_questioner_client
from agents.personas import DEFAULT_PATIENCE_DECAY
from agents.prompts import build_questioner_prompt
from agents.schema import ObservationReport, Persona

END_SESSION = "<end_session>"


class Questioner:
    """Persona user-agent that generates the next follow-up query (§7.4)."""

    def __init__(self, client: ChatClient | None = None, *, max_tokens: int = 1024):
        # 1024 (was 512, orig 256): thinking models in the rotation pool
        # (deepseek-v4-pro / kimi-k2.6 / qwen3-max) spend a large share of the
        # budget on hidden reasoning before emitting the query. 512 still left
        # the content truncated for those two (observed TruncatedOutputError
        # flooding in the Jul-9/10 smoke runs), so the questioner kept failing
        # over to the non-thinking models and wasting calls. 1024 gives the
        # thinking models room to reason AND emit a short follow-up.
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
        """Return the next query text, or None == session ended.

        None can mean two things (distinguished by caller checking
        ``last_query_was_error``):
          - The model returned ``<end_session>`` (satisfied user).
          - An API/LLM error occurred (connection failure, timeout, etc.).

        Both currently result in session termination, but the caller should
        record the distinction for telemetry (P1 TODO from CLAUDE.md).
        """
        messages = build_questioner_prompt(persona=persona, report=report, session_history=session_history)
        self._last_query_was_error = False
        # Truncation guard (thinking models): max_tokens budget can be entirely
        # consumed by hidden reasoning, leaving content empty. Without this, an
        # empty reply is indistinguishable from "<end_session>" (satisfied user)
        # and the session would silently abort. Treat truncation as an error so
        # the patience/telemetry path (not the "satisfied" path) handles it.
        try:
            raw = self.client.chat(messages, max_tokens=self._max_tokens)
        except TruncatedOutputError:
            self._last_query_was_error = True
            return None
        except Exception:  # noqa: BLE001 -- a failed generation ends the session
            self._last_query_was_error = True
            return None
        text = (raw or "").strip()
        if not text or text == END_SESSION or END_SESSION in text:
            return None
        return text

    @property
    def last_query_was_error(self) -> bool:
        """True if the last ``next_query`` call ended due to an API/LLM error.

        When False and ``next_query`` returned None, the model chose to end
        the session (satisfied user). This distinction matters for telemetry
        and for the patience mechanism (§3.6.5): a genuinely satisfied user
        should not consume patience, whereas an API failure is ambiguous.
        """
        return getattr(self, "_last_query_was_error", False)


# --------------------------------------------------------------------------- #
# Patience mechanism (§3.6.5) -- pure, seeded, unit-testable                   #
# --------------------------------------------------------------------------- #


class PatienceTracker:
    """Per-session patience over FAILED winner turns (§3.6.5).

    Multiplicative decay with retention rate r (close to 1):
        P_k = P0 * r^k        (k = failed turns so far)

    Decision after the k-th failure:
      - P ≤ 0.1   → always stop (exhausted)
      - P ≥ 1     → always continue
      - otherwise → continue with probability P (coin flip)

    Successful turns do not consume patience.
    """

    def __init__(self, persona: Persona, rng: random.Random):
        self.p0 = float(persona.patience)
        self.r = float(persona.patience_decay) if persona.patience_decay else DEFAULT_PATIENCE_DECAY
        self.rng = rng
        self.fail_count = 0

    def current_patience(self) -> float:
        """P_k after k failures: P0 * r^k."""
        return self.p0 * self.r ** self.fail_count

    def on_failure(self) -> bool:
        """Register a failed turn; return True to REDO, False to end session.

        P ≤ 0.1 → exhausted (stop).  P ≥ 1 → always continue.
        Otherwise → continue with probability P (coin flip).
        """
        self.fail_count += 1
        pk = self.current_patience()
        if pk <= 0.1:
            return False
        if pk >= 1.0:
            return True
        return self.rng.random() < pk
