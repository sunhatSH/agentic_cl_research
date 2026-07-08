"""Two-level failover chat client (2026-07-08).

Wraps a role's provider pool (from ``agents.config.ResolvedRole``) so that a
transient model outage (502/401/timeout/truncation) is handled transparently:

    outer loop : providers (vendors)   -- try next vendor only when one is fully down
    inner loop : models within a vendor -- try next same-vendor model on failure

The first endpoint that succeeds becomes the new default (``current_*`` indices),
so subsequent calls start from a known-good endpoint instead of re-hitting a dead
one every time. State lives in memory by default; pass ``state_path`` (or set env
``AGENT_ENDPOINT_STATE``) to persist it to a small JSON so concurrent short-lived
collection processes reuse the discovery instead of each re-probing a 502.

Anti mode-collapse rotation (questioner's ``rotate_every`` > 0) is orthogonal to
failover: every N calls it advances to the next *available* endpoint; if that one
is down, failover skips forward. ``rotate_every == 0`` disables rotation (the
client sticks to the current-good endpoint, which is what observer/reward want).

The public surface (``chat`` / ``chat_with_tools`` / ``model``) matches
``OpenAIChatClient`` so callers are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import httpx

from agents.base import OpenAIChatClient, TruncatedOutputError
from agents.config import ResolvedRole

logger = logging.getLogger(__name__)

# Errors that mean "this endpoint is unusable right now, try another".
# 4xx other than 401/408/429 are treated as caller errors (bad request) and
# re-raised, since retrying another model would not help.
_RETRYABLE_STATUS = {401, 408, 429, 500, 502, 503, 504}


class AllEndpointsFailed(RuntimeError):
    """Raised only when every model of every provider failed for one call."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (TruncatedOutputError, httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


class FailoverChatClient:
    """Two-level (provider x model) failover over a role's endpoint pool."""

    def __init__(self, role: ResolvedRole, *, state_path: str | None = None):
        # Flatten into (provider_idx, model_idx) addressable clients, but keep
        # provider boundaries so failover can express "this vendor is fully down".
        self._role = role.role
        self._providers = role.providers
        if not self._providers or not any(p.endpoints for p in self._providers):
            raise ValueError(f"FailoverChatClient[{role.role}] has no endpoints")
        self._clients: list[list[OpenAIChatClient]] = [
            [
                OpenAIChatClient(
                    base_url=ep.base_url,
                    model=ep.model,
                    api_key=ep.api_key,
                    temperature=ep.temperature,
                )
                for ep in p.endpoints
            ]
            for p in self._providers
        ]
        self._rotate_every = max(0, role.rotate_every)
        self._call_count = 0
        self._lock = threading.RLock()

        self._state_path = state_path or os.environ.get("AGENT_ENDPOINT_STATE", "").strip() or None
        # current default endpoint (provider_idx, model_idx)
        self._pi, self._mi = self._load_state()

    # -- state (memory + optional disk) ------------------------------------- #
    def _load_state(self) -> tuple[int, int]:
        if self._state_path and os.path.exists(self._state_path):
            try:
                with open(self._state_path, encoding="utf-8") as f:
                    st = json.load(f).get(self._role, {})
                pi, mi = int(st.get("provider_idx", 0)), int(st.get("model_idx", 0))
                if 0 <= pi < len(self._clients) and 0 <= mi < len(self._clients[pi]):
                    return pi, mi
            except Exception:  # noqa: BLE001 -- corrupt/partial state -> start fresh
                pass
        return 0, 0

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            data: dict = {}
            if os.path.exists(self._state_path):
                with open(self._state_path, encoding="utf-8") as f:
                    data = json.load(f) or {}
            data[self._role] = {"provider_idx": self._pi, "model_idx": self._mi}
            tmp = f"{self._state_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._state_path)  # atomic
        except Exception:  # noqa: BLE001 -- persistence is best-effort
            pass

    # -- endpoint ordering -------------------------------------------------- #
    @property
    def model(self) -> str:
        with self._lock:
            return self._clients[self._pi][self._mi].model

    @property
    def current_model(self) -> str:  # RotatingChatClient-compatible alias
        return self.model

    def _rotate_start(self) -> None:
        """Anti-collapse: advance the *default* model within the current provider
        every rotate_every calls (0 disables). Failover still applies on top."""
        if self._rotate_every <= 0:
            return
        self._call_count += 1
        if self._call_count >= self._rotate_every:
            self._call_count = 0
            prov = self._clients[self._pi]
            self._mi = (self._mi + 1) % len(prov)

    def _ordered_attempts(self) -> list[tuple[int, int]]:
        """Endpoints to try this call, starting from the current default,
        provider-major then model-minor, wrapping around once."""
        order: list[tuple[int, int]] = []
        nprov = len(self._clients)
        for dp in range(nprov):
            pi = (self._pi + dp) % nprov
            prov = self._clients[pi]
            # within the starting provider begin at current model; others at 0
            start_m = self._mi if pi == self._pi else 0
            for dm in range(len(prov)):
                mi = (start_m + dm) % len(prov)
                order.append((pi, mi))
        return order

    # -- core call with failover -------------------------------------------- #
    def _call(self, fn_name: str, *args, **kwargs):
        with self._lock:
            self._rotate_start()
            attempts = self._ordered_attempts()
        last_exc: Exception | None = None
        for pi, mi in attempts:
            client = self._clients[pi][mi]
            try:
                result = getattr(client, fn_name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if _is_retryable(exc):
                    last_exc = exc
                    logger.warning(
                        "FailoverChatClient[%s]: %s on %s (provider %s) -> next",
                        self._role, type(exc).__name__, client.model, self._providers[pi].name,
                    )
                    continue
                raise  # non-retryable (e.g. 400 bad request) -> surface immediately
            # success: promote this endpoint to the default and persist
            with self._lock:
                if (pi, mi) != (self._pi, self._mi):
                    self._pi, self._mi = pi, mi
                    logger.info(
                        "FailoverChatClient[%s]: default now %s (provider %s)",
                        self._role, client.model, self._providers[pi].name,
                    )
                    self._save_state()
            return result
        raise AllEndpointsFailed(
            f"FailoverChatClient[{self._role}]: all "
            f"{sum(len(p.endpoints) for p in self._providers)} endpoints failed; "
            f"last error: {last_exc!r}"
        )

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        return self._call("chat", messages, max_tokens=max_tokens)

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        return self._call("chat_with_tools", messages, tools=tools, max_tokens=max_tokens)
