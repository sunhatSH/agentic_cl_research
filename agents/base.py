"""Shared OpenAI-compatible backend for the user-sim agents.

The three agents (observer / questioner / reward) each call a model over an
OpenAI-compatible HTTP endpoint, with SEPARATE env config so they can point at
different models/endpoints -- mitigating the self-preference bias of one model
observing, asking, AND grading (doc §6 ⚠️ / §7.5):

    OBSERVER_API_BASE / OBSERVER_MODEL / TOKENHUB_API_KEY
    USERSIM_API_BASE  / USERSIM_MODEL  / TOKENHUB_API_KEY
    REWARD_API_BASE   / REWARD_MODEL   / TOKENHUB_API_KEY  (reused from model_reward)

This mirrors trainer/model_reward.OpenAIJudgeClient so the wire-up and parsing
are consistent. The client is injectable (``ChatClient`` Protocol) so agents
unit-test off-network with a mock.

Anti mode-collapse (2026-06-22): the Questioner now supports multi-model
rotation via ``USERSIM_ENDPOINTS`` -- a JSON array of endpoint objects, every
N calls it switches to the next model endpoint in the pool, cycling through
different providers so no single model's idiosyncrasies dominate the follow-up
queries.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

# finish_reason values that mean the reply was cut off mid-generation (hit the
# max_tokens ceiling). Thinking models (gpt-5.1, claude-opus-thinking, ...) can
# burn the whole token budget on hidden reasoning and emit little/ no visible
# content; if we swallow that as a "complete" answer, every downstream agent
# silently corrupts. Detect it here, centrally, for ALL model calls.
_TRUNCATION_REASONS = ("length", "length_tokens")


class TruncatedOutputError(RuntimeError):
    """Raised when a model reply was cut off (``finish_reason`` == length*).

    The visible ``content`` is partial and MUST NOT be consumed as a final
    answer. Callers react per their semantics: retry with a larger budget,
    fall back to a deterministic path, or flag a judge error -- but never treat
    the half-output as complete.
    """


class ChatClient(Protocol):
    """Minimal chat surface the agents depend on.

    ``chat`` / ``chat_with_tools`` raise ``TruncatedOutputError`` when the
    model hit its token ceiling (``finish_reason == length*``); callers that
    need a complete reply must catch it.
    """

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        """Return the assistant message text for the given chat messages."""
        ...


class OpenAIChatClient:
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "sk-local",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _raise_if_truncated(data, self.model)
        return data["choices"][0]["message"]["content"]

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Call /chat/completions with tool support (OpenAI function-calling).

        Returns the full message dict (may contain ``tool_calls`` or plain ``content``).
        When *tools* is None or empty, falls back to a plain ``chat`` call and
        wraps the result as ``{"content": ..., "role": "assistant"}``.
        """
        import httpx

        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _raise_if_truncated(data, self.model)
        return data["choices"][0]["message"]


def _raise_if_truncated(data: dict, model: str) -> None:
    """Raise ``TruncatedOutputError`` when the reply was cut off by max_tokens.

    OpenAI ``finish_reason`` taxonomy: ``stop`` (complete), ``length`` / ``length_tokens``
    (hit max_tokens mid-output), ``tool_calls`` (complete, requesting a tool).
    Thinking models can spend the whole budget on hidden reasoning and emit a
    fragment -- treat any ``length*`` reason as corruption, NOT as a final answer.
    Some gateways omit ``finish_reason``; then we fall back to a content check:
    an empty ``content`` AND no ``tool_calls`` is also suspicious -> raise.
    """
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return  # malformed upstream; let callers see whatever they parse
    finish = choice.get("finish_reason") or ""
    if str(finish).lower() in _TRUNCATION_REASONS:
        raise TruncatedOutputError(
            f"model {model!r} reply truncated (finish_reason={finish!r}); increase "
            "max_tokens or the output is partial. Do NOT consume as a final answer."
        )
    # Gateway may not report finish_reason (tokenhub gpt-5.1 sometimes omits it).
    # A truly empty reply (no content AND no tool_calls) almost always means the
    # thinking budget ate everything -- flag it rather than silently returning "".
    msg = choice.get("message") or {}
    if not finish and not (msg.get("content") or msg.get("tool_calls")):
        raise TruncatedOutputError(
            f"model {model!r} returned empty content with no finish_reason and no "
            "tool_calls; likely the thinking budget consumed all max_tokens."
        )


def _resolve(prefix: str, *, temperature: float) -> OpenAIChatClient:
    """Resolve client from env vars (legacy fallback).

    Prefer ``resolve_observer_client`` / ``resolve_questioner_client`` which
    read configs/agents.yaml first and only fall back to env vars.
    """
    base = os.environ.get(f"{prefix}_API_BASE")
    model = os.environ.get(f"{prefix}_MODEL")
    if not base or not model:
        raise RuntimeError(
            f"user-sim agent not configured: set {prefix}_API_BASE + {prefix}_MODEL. "
            "Agents intentionally use separate endpoints (anti self-preference, doc §6)."
        )
    return OpenAIChatClient(
        base_url=base,
        model=model,
        api_key=os.environ.get(f"{prefix}_API_KEY", "sk-local"),
        temperature=temperature,
    )


def validate_endpoints_distinct() -> list[str]:
    """Validate that Observer/Questioner/Reward use different model names.

    Anti self-preference (doc §6): the same model observing, asking, AND grading
    would bias the process. Returns a list of warnings (empty if all OK).

    Uses config-first resolution (configs/agents.yaml + env overrides) so it
    reflects the actual runtime configuration.
    """
    from agents.config import validate_model_distinctness

    return validate_model_distinctness()


def resolve_observer_client():
    """Observer is objective -> temperature 0 (deterministic evidence).

    Returns a FailoverChatClient over the observer provider pool (agents.yaml
    ``observer.providers``); falls back to OBSERVER_* env vars. rotate_every=0
    means it sticks to the current-good model and only switches on failure.
    """
    from agents.config import resolve_role

    try:
        role = resolve_role("observer")
        from agents.failover import FailoverChatClient

        return FailoverChatClient(role)
    except RuntimeError:
        pass  # fall through to env-only legacy path
    return _resolve("OBSERVER", temperature=0.0)


class RotatingChatClient:
    """Round-robin over multiple OpenAI-compatible endpoints every *rotate_every* calls.

    Anti mode-collapse: the Questioner rotates through different model providers
    so no single model's output style dominates follow-up queries (doc §3.5).

    Env convention (QUESTIONER rotation)::

        USERSIM_ENDPOINTS = '[{"base_url":"...","model":"...","api_key":"..."}, ...]'
        USERSIM_ROTATE_EVERY = 5   (default)

    ``USERSIM_ENDPOINTS`` is a JSON array of objects, each with keys
    ``base_url``, ``model``, and optionally ``api_key`` (defaults to
    ``"sk-local"``). This is cleaner and less error-prone than a delimited
    string, and supports values containing special characters.

    Falls back to the single ``USERSIM_API_BASE/MODEL/KEY`` if
    ``USERSIM_ENDPOINTS`` is not set.
    """

    def __init__(
        self,
        clients: list[OpenAIChatClient],
        rotate_every: int = 5,
    ) -> None:
        if not clients:
            raise ValueError("RotatingChatClient requires at least one client")
        self._clients = clients
        self._rotate_every = max(1, rotate_every)
        self._call_count = 0
        self._current_idx = 0
        self._lock = threading.Lock()

    @property
    def current_model(self) -> str:
        """The model name of the currently active client."""
        return self._clients[self._current_idx].model

    def _advance(self) -> None:
        """Increment call count and rotate if threshold reached."""
        with self._lock:
            self._call_count += 1
            if self._call_count >= self._rotate_every:
                self._call_count = 0
                self._current_idx = (self._current_idx + 1) % len(self._clients)
                logger.info(
                    "RotatingChatClient: switching to model %s (endpoint %d/%d)",
                    self.current_model,
                    self._current_idx + 1,
                    len(self._clients),
                )

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        """Chat via the current active client, then rotate if threshold reached.

        ``_advance`` runs in a ``finally`` so a raising endpoint still advances
        the rotation counter -- otherwise a persistently-down endpoint would be
        retried on every call (the counter never moved past it), pinning the
        client to a dead server instead of rotating away from it.
        """
        with self._lock:
            idx = self._current_idx
        client = self._clients[idx]
        try:
            return client.chat(messages, max_tokens=max_tokens)
        finally:
            self._advance()

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Tool-use chat via the current active client, then rotate if threshold reached."""
        with self._lock:
            idx = self._current_idx
        client = self._clients[idx]
        result = client.chat_with_tools(messages, tools=tools, max_tokens=max_tokens)
        self._advance()
        return result


def _parse_endpoints(raw: str) -> list[dict[str, str]]:
    """Parse ``USERSIM_ENDPOINTS`` as a JSON array of endpoint objects.

    Each element must be a dict with keys ``base_url`` and ``model``;
    ``api_key`` is optional (defaults to ``"sk-local"``).

    Example::

        [{"base_url":"http://a/v1","model":"openai/gpt-5","api_key":"sk-xxx"},
         {"base_url":"http://b/v1","model":"anthropic/claude-sonnet-5"}]

    Returns list of ``{base_url, model, api_key}`` dicts.
    Raises ``ValueError`` on malformed or empty input.
    """
    import json

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"USERSIM_ENDPOINTS must be a JSON array of objects, got: {raw[:200]!r}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("USERSIM_ENDPOINTS must be a non-empty JSON array")
    entries: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"USERSIM_ENDPOINTS entry #{i + 1} must be an object, got {type(item).__name__}")
        base = str(item.get("base_url", "")).strip()
        model = str(item.get("model", "")).strip()
        if not base or not model:
            raise ValueError(f"USERSIM_ENDPOINTS entry #{i + 1} missing 'base_url' or 'model': {item!r}")
        entries.append(
            {
                "base_url": base,
                "model": model,
                "api_key": str(item.get("api_key", "")).strip() or "sk-local",
            }
        )
    return entries


def resolve_questioner_client():
    """Questioner needs diversity -> higher temperature (anti mode-collapse, §3.5).

    Returns a FailoverChatClient over the questioner provider pool
    (agents.yaml ``questioner.providers``). Failover handles a dead model
    (e.g. a 502'd rotation member); rotate_every>0 additionally rotates through
    *available* models for style diversity. Falls back to USERSIM_* env vars.
    """
    from agents.config import resolve_role

    try:
        role = resolve_role("questioner")
        from agents.failover import FailoverChatClient

        logger.info(
            "Questioner failover pool: %s (rotate_every=%d)",
            [ep.model for ep in role.flat_endpoints()],
            role.rotate_every,
        )
        return FailoverChatClient(role)
    except RuntimeError:
        pass  # fall through to env-only legacy path
    return _resolve("USERSIM", temperature=0.9)


def resolve_reward_client():
    """Reward/judge over the ChatClient interface, with failover.

    (The verl custom_reward path uses ``OpenAIJudgeClient`` in
    trainer/model_reward.py, which has its own scoring HTTP shape; this resolver
    serves ChatClient-based reward callers.) Falls back to REWARD_* env vars.
    """
    from agents.config import resolve_role

    try:
        role = resolve_role("reward")
        from agents.failover import FailoverChatClient

        # Reward judge emits long rubric-scored output -> enable truncation
        # escalation (retry same model at 2x budget before failing over).
        return FailoverChatClient(role, escalate_on_truncation=True)
    except RuntimeError:
        pass
    return _resolve("REWARD", temperature=0.0)
