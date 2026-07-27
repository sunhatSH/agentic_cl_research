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

    The cluster wiring (doc/sandbox/Sandbox_Agent架构.md §3.2): we own the 16×8 +
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

        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # apply_chat_template 可能返回 BatchEncoding / dict（含 input_ids）而非纯 int list。
        # BatchEncoding 不是 dict 子类，isinstance(_, dict) 判 False，直接 list() 会拿到
        # key 字符串 ['input_ids','attention_mask'] → lightllm "prompt format error"。
        if hasattr(prompt_ids, "get") and "input_ids" in prompt_ids:
            prompt_ids = prompt_ids["input_ids"]
        elif isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)
        # 若仍是嵌套（batch 维 [[...]]），取第一条。
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])

        async def _gen():
            return await self.llm_client.generate(
                uuid4().hex, prompt_ids=prompt_ids, sampling_params=self.sampling_params
            )

        # The scheduler runs each session on its own thread, which has no default
        # event loop. Detect the illegal "called from inside a running loop" case;
        # otherwise create a dedicated loop and ALWAYS close it (the old code
        # leaked one loop per call across thousands of steps -> fd exhaustion).
        # Errors raised INSIDE _gen() must propagate -- the previous
        # ``except RuntimeError`` swallowed genuine generate failures and silently
        # retried on a fresh loop, masking the real cause and doubling load.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:  # pragma: no cover - nested-loop guard
            raise RuntimeError(
                "VerlRolloutGenerateFn called from inside a running event loop; "
                "run it on a worker thread (the scheduler already does this)."
            )
        loop = asyncio.new_event_loop()
        try:
            out = loop.run_until_complete(_gen())
        finally:
            loop.close()

        token_ids = list(getattr(out, "token_ids", []) or [])
        logprobs = list(getattr(out, "log_probs", None) or [])
        # Guard the GRPO importance ratio: a backend that returns token_ids but
        # ragged/absent logprobs would silently misalign (chains into the
        # batch-wide logprob drop in cl_rollout_manager). Drop logprobs for THIS
        # step and warn rather than emit a length-mismatched vector.
        if logprobs and len(logprobs) != len(token_ids):
            print(
                f"[generate] WARNING: logprob len {len(logprobs)} != token len "
                f"{len(token_ids)}; dropping logprobs for this step.",
                flush=True,
            )
            logprobs = []
        text = self.tokenizer.decode(token_ids) if token_ids else ""
        return GenStep(text=text, response_ids=token_ids, logprobs=logprobs)


class HTTPGenerateFn:
    """OpenAI-compatible single-step generate with full actor-side data capture.

    Designed for cold-start collection and off-cluster sampling. Captures ALL
    data the actor produces in one generation step -- text, token IDs, logprobs,
    and prompt/completion token counts from the ``usage`` field -- so downstream
    consumers (buffer, metrics, weighting) have the same information they would
    get from verl's native rollout.

    When a ``tokenizer`` is provided, token text from the logprobs payload is
    re-encoded to recover integer token IDs (vllm /chat/completions returns
    token text + bytes but NOT token IDs). Without a tokenizer, IDs are
    placeholder zeros but the count matches the real token count so that
    ``response_mask`` length and ``TokenWeighting`` seq_len are correct.

    Prefer VerlRolloutGenerateFn on the cluster so logprobs are native and
    consistent with training.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "sk-local",
        *,
        tokenizer: Any = None,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        # Accumulated actor-side stats across all calls (for logging/metrics).
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _recover_token_ids(self, logprob_tokens: list[dict]) -> list[int]:
        """Best-effort recover integer token IDs from the logprobs payload.

        vllm returns ``{token: "Hello", logprob: ..., bytes: [...]}`` -- no
        integer token ID. With a tokenizer we can encode each token's text back
        to its ID. Without one we emit zeros (the count is still correct for
        mask/weighting purposes; build_replay_rows re-tokenizes anyway).
        """
        if not logprob_tokens:
            return []
        if self.tokenizer is None:
            return [0] * len(logprob_tokens)
        ids: list[int] = []
        for tok in logprob_tokens:
            text = tok.get("token", "")
            if not text:
                ids.append(0)
                continue
            try:
                # encode with add_special_tokens=False to get the raw token ID
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                ids.append(encoded[0] if encoded else 0)
            except Exception:  # noqa: BLE001 -- degrade gracefully
                ids.append(0)
        return ids

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
        body = resp.json()
        choice = body["choices"][0]
        text = choice["message"]["content"] or ""

        # --- actor-side usage stats ---
        usage = body.get("usage") or {}
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)

        # --- token IDs + logprobs from the OpenAI logprobs payload ---
        logprobs: list[float] = []
        lp = choice.get("logprobs") or {}
        lp_tokens: list[dict] = lp.get("content") or []
        for tok in lp_tokens:
            logprobs.append(float(tok.get("logprob", 0.0)))
        response_ids = self._recover_token_ids(lp_tokens)

        # When logprobs is empty (some endpoints don't return it), fall back
        # to estimating token count from the generated text length.
        if not response_ids and text:
            # Use usage.completion_tokens when available for accuracy.
            n_tokens = usage.get("completion_tokens") or max(1, len(text) // 4)
            response_ids = [0] * n_tokens
            logprobs = [0.0] * n_tokens
        return GenStep(
            text=text,
            response_ids=response_ids,
            logprobs=logprobs,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
