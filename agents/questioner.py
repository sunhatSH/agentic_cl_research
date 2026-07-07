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

    def __init__(self, client: ChatClient | None = None, *, max_tokens: int = 512):
        # 512 (was 256): thinking models in the rotation pool (sonnet-4-6 /
        # deepseek-v4-pro / qwen3.7-max / kimi-k2.6) can spend a large share of
        # the budget on hidden reasoning before emitting the query. 256 risked an
        # empty reply, which was indistinguishable from "<end_session>".
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

    Patience decays with exponentially growing decrement:

        P_k = P_0 - d_0 * (1.5^k - 1) / 0.5   (cumulative after k failures)

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
        return self.p0 - self.d0 * (1.5**k - 1) / 0.5

    def on_failure(self) -> bool:
        """Register a failed turn; return True to REDO, False to end session.

        Returns the seeded probabilistic decision based on the post-decrement
        patience P_k. After P_k goes negative, redo probability is 0 (stop).
        """
        self.fail_count += 1
        pk = self.current_patience()
        redo_prob = max(0.0, min(1.0, pk))
        return self.rng.random() < redo_prob
