"""Concrete GenerateFn implementations for the inference boundary.

A ``GenerateFn`` maps a chat message list to a ``GenStep`` (generated text +
response token ids + per-token logprobs). The session/agent layer is written
against this boundary (rollout/collect.py), so swapping generation backend is a
one-line change.
"""

from __future__ import annotations

from typing import Any

from rollout.collect import GenStep


class VerlRolloutGenerateFn:
    """Single-step generate backed by verl's native rollout engine.

    The cluster wiring (doc/Sandbox_Agent架构.md §3.2): we own the 16×8 +
    winner-sync orchestration, but call verl's rollout generate for each step so
    token + logprob + response_mask are produced natively (no proxy). The exact
    AgentLoopOutput field plumbing is validated on the GPU cluster (Gap D /
    doc/Progress.md). This holds the rollout worker group + tokenizer and adapts
    one generate() call into a GenStep.
    """

    def __init__(self, rollout_wg: Any, tokenizer: Any, *, max_new_tokens: int = 1024):
        self.rollout_wg = rollout_wg
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, Any]]) -> GenStep:
        # Cluster path: build a prompt DataProto from messages, call
        # self.rollout_wg.generate_sequences(...), and read response_ids /
        # rollout_log_probs / response_mask from the returned AgentLoopOutput.
        # Left as the integration seam validated on GPU (verl rollout API), to
        # avoid hardcoding a generate signature that differs by engine/version.
        raise NotImplementedError(
            "VerlRolloutGenerateFn wires verl rollout generate on the GPU cluster "
            "(Gap D, doc/Progress.md). Use HTTPGenerateFn or a mock off-cluster."
        )


class HTTPGenerateFn:
    """OpenAI-compatible single-step generate (W1 fallback, doc §3.3).

    Used only when driving generation outside verl (e.g. local smoke against a
    served policy). Returns generated text; token ids/logprobs are best-effort
    (logprobs require the endpoint to return them). Prefer VerlRolloutGenerateFn
    on the cluster so logprobs are native and consistent with training.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "sk-local",
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def __call__(self, messages: list[dict[str, Any]]) -> GenStep:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "logprobs": True,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = choice["message"]["content"] or ""

        # Best-effort token ids + logprobs from the OpenAI logprobs payload.
        response_ids: list[int] = []
        logprobs: list[float] = []
        lp = choice.get("logprobs") or {}
        for tok in lp.get("content", []) or []:
            logprobs.append(float(tok.get("logprob", 0.0)))
            response_ids.append(0)  # ids not exposed by chat API; placeholder
        return GenStep(text=text, response_ids=response_ids, logprobs=logprobs)
