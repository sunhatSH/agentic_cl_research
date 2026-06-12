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
    """Single-step generate backed by verl's native rollout LLM server.

    The cluster wiring (doc/Sandbox_Agent架构.md §3.2): we own the 16×8 +
    winner-sync orchestration, but call verl's rollout LLM server for each step
    so token + logprob are produced natively (no proxy). The connection point is
    ``LLMServerClient.generate(request_id, *, prompt_ids, sampling_params)
    -> TokenOutput{token_ids, log_probs}`` (verl/workers/rollout/llm_server.py),
    the same per-turn call verl's own AgentLoopWorker uses.

    ``llm_client.generate`` is async; the ReAct loop / scheduler are synchronous,
    so we bridge with a private event loop per call (the scheduler runs sessions
    on a thread pool, so each thread gets its own loop). Validated end-to-end on
    the GPU cluster (no verl/LLM server off-cluster).
    """

    def __init__(self, llm_client: Any, tokenizer: Any, *, sampling_params: dict | None = None):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params or {"temperature": 1.0, "max_tokens": 1024}

    def __call__(self, messages: list[dict[str, Any]]) -> GenStep:
        import asyncio
        from uuid import uuid4

        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)

        async def _gen():
            return await self.llm_client.generate(
                uuid4().hex, prompt_ids=prompt_ids, sampling_params=self.sampling_params
            )

        # Each scheduler thread runs its own loop; reuse if one is set, else create.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():  # pragma: no cover - nested-loop guard
                raise RuntimeError("nested loop")
            out = loop.run_until_complete(_gen())
        except RuntimeError:
            out = asyncio.new_event_loop().run_until_complete(_gen())

        token_ids = list(getattr(out, "token_ids", []) or [])
        logprobs = list(getattr(out, "log_probs", None) or [])
        text = self.tokenizer.decode(token_ids) if token_ids else ""
        return GenStep(text=text, response_ids=token_ids, logprobs=logprobs)


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
