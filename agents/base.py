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


class ChatClient(Protocol):
    """Minimal chat surface the agents depend on."""

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
        return resp.json()["choices"][0]["message"]["content"]

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
        return resp.json()["choices"][0]["message"]


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


def resolve_observer_client() -> OpenAIChatClient:
    """Observer is objective -> temperature 0 (deterministic evidence).

    Reads configs/agents.yaml first; falls back to OBSERVER_* env vars.
    """
    from agents.config import resolve_observer as _resolve_from_config

    try:
        ep = _resolve_from_config()
        return OpenAIChatClient(
            base_url=ep.base_url,
            model=ep.model,
            api_key=ep.api_key,
            temperature=ep.temperature,
        )
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
        """Chat via the current active client, then rotate if threshold reached."""
        with self._lock:
            idx = self._current_idx
        client = self._clients[idx]
        result = client.chat(messages, max_tokens=max_tokens)
        self._advance()
        return result

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

        [{"base_url":"http://a/v1","model":"gpt-4.1","api_key":"sk-xxx"},
         {"base_url":"http://b/v1","model":"claude-sonnet-4-6"}]

    Returns list of ``{base_url, model, api_key}`` dicts.
    Raises ``ValueError`` on malformed or empty input.
    """
    import json

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"USERSIM_ENDPOINTS must be a JSON array of objects, got: {raw[:200]!r}"
        ) from exc
    if not isinstance(data, list) or not data:
        raise ValueError("USERSIM_ENDPOINTS must be a non-empty JSON array")
    entries: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"USERSIM_ENDPOINTS entry #{i + 1} must be an object, got {type(item).__name__}")
        base = str(item.get("base_url", "")).strip()
        model = str(item.get("model", "")).strip()
        if not base or not model:
            raise ValueError(
                f"USERSIM_ENDPOINTS entry #{i + 1} missing 'base_url' or 'model': {item!r}"
            )
        entries.append({
            "base_url": base,
            "model": model,
            "api_key": str(item.get("api_key", "")).strip() or "sk-local",
        })
    return entries


def resolve_questioner_client() -> OpenAIChatClient | RotatingChatClient:
    """Questioner needs diversity -> higher temperature (anti mode-collapse, §3.5).

    Reads configs/agents.yaml first; falls back to USERSIM_* env vars.
    Rotation pool is built from the config file's ``questioner.rotation`` section
    (or USERSIM_ENDPOINTS env override).
    """
    from agents.config import resolve_questioner as _resolve_from_config

    try:
        q_cfg = _resolve_from_config()
        if q_cfg.rotation:
            clients = [
                OpenAIChatClient(
                    base_url=ep.base_url,
                    model=ep.model,
                    api_key=ep.api_key,
                    temperature=ep.temperature,
                )
                for ep in q_cfg.rotation
            ]
            logger.info(
                "Questioner rotation: %d models, rotate every %d calls: %s",
                len(clients),
                q_cfg.rotate_every,
                [c.model for c in clients],
            )
            return RotatingChatClient(clients, rotate_every=q_cfg.rotate_every)
        if q_cfg.fallback:
            return OpenAIChatClient(
                base_url=q_cfg.fallback.base_url,
                model=q_cfg.fallback.model,
                api_key=q_cfg.fallback.api_key,
                temperature=q_cfg.fallback.temperature,
            )
    except RuntimeError:
        pass  # fall through to env-only legacy path
    return _resolve("USERSIM", temperature=0.9)
