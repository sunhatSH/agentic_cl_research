"""Inference (generation) side: the single-step generate boundary.

Decision (doc/sandbox/Sandbox_Agent架构.md §3.2): we orchestrate the session loop, but
every single generation step is produced by verl's NATIVE rollout generate
(token + logprob + response_mask come for free), NOT by an HTTP proxy. The
session/agent layer depends only on the ``GenerateFn`` boundary defined in
``rollout/collect.py``; this package provides the concrete implementations of
that boundary.

Two implementations:
  - ``VerlRolloutGenerateFn`` : wraps verl's rollout LLM server
    (``LLMServerClient.generate`` -> token ids + logprobs). Cluster path;
    validated end-to-end on GPU. Returns native fields.
  - ``HTTPGenerateFn``        : OpenAI-compatible endpoint fallback (W1) used only
    when driving generation outside verl; logprobs are best-effort.

Both return a ``rollout.collect.GenStep`` so the ReAct loop / session driver are
generation-backend agnostic.
"""

from inference.generate import HTTPGenerateFn, VerlRolloutGenerateFn

__all__ = ["VerlRolloutGenerateFn", "HTTPGenerateFn"]
