"""Shared OpenAI-compatible backend for the user-sim agents.

The three agents (observer / questioner / reward) each call a model over an
OpenAI-compatible HTTP endpoint, with SEPARATE env config so they can point at
different models/endpoints -- mitigating the self-preference bias of one model
observing, asking, AND grading (doc §6 ⚠️ / §7.5):

    OBSERVER_API_BASE / OBSERVER_MODEL / OBSERVER_API_KEY
    USERSIM_API_BASE  / USERSIM_MODEL  / USERSIM_API_KEY
    JUDGE_API_BASE    / JUDGE_MODEL    / JUDGE_API_KEY   (reused from model_reward)

This mirrors trainer/model_reward.OpenAIJudgeClient so the wire-up and parsing
are consistent. The client is injectable (``ChatClient`` Protocol) so agents
unit-test off-network with a mock.
"""

from __future__ import annotations

import os
from typing import Protocol


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


def _resolve(prefix: str, *, temperature: float) -> OpenAIChatClient:
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


def resolve_observer_client() -> OpenAIChatClient:
    """Observer is objective -> temperature 0 (deterministic evidence)."""
    return _resolve("OBSERVER", temperature=0.0)


def resolve_questioner_client() -> OpenAIChatClient:
    """Questioner needs diversity -> higher temperature (anti mode-collapse, §3.5)."""
    return _resolve("USERSIM", temperature=0.9)
