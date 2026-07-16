"""Custom rollout manager: route verl's rollout through our session scheduler.

verl 0.8.0 exposes an OFFICIAL injection point for replacing rollout
(``ray_trainer.py:931`): set
``actor_rollout_ref.rollout.agent.agent_loop_manager_class`` to an FQN that
``load_class_from_fqn(..., "AgentLoopManager")`` resolves; verl then uses it
instead of the default ``AgentLoopManager`` -- WITHOUT touching ``fit()``.

We subclass ``AgentLoopManager`` and override only ``generate_sequences``:

    verl fit() ── generate_sequences(prompts: DataProto) ──►  CLSchedulerAgentLoopManager
                                                                │
                    RolloutScheduler (16 sessions × 8 slots)    │  winner-sync per query
                      each per-slot step → llm_client.generate  │  (our existing code)
                                                                ▼
                    list[Trajectory] ── trajectories_to_dataproto ──► DataProto (verl contract)

The per-step generation reuses verl's NATIVE rollout LLM server
(``llm_client.generate`` -> token_ids + log_probs) via
``inference.VerlRolloutGenerateFn`` -- no HTTP proxy. The 16×8 + winner-sync
orchestration is our existing, unit-tested ``RolloutScheduler`` /
``SessionSandboxPool``; this module is just the adapter on both ends.

verl is imported lazily (in ``create``/assembly) so the module imports off-cluster;
``trajectories_to_dataproto`` and the prompt-extraction helper are pure and
unit-tested with fakes. End-to-end (real verl DataProto + LLM server + sandbox)
is validated on the GPU cluster.
"""

from __future__ import annotations

from typing import Any

# --- pure: verl rollout-contract DataProto assembly (unit-tested w/ fakes) ----


def _left_pad(seq: list[int], width: int, pad: int) -> list[int]:
    return [pad] * (width - len(seq)) + list(seq)


def _right_pad(seq: list, width: int, pad) -> list:
    return list(seq) + [pad] * (width - len(seq))


def trajectories_to_dataproto(
    trajectories: list[Any],
    prompt_token_ids: list[list[int]],
    *,
    pad_token_id: int = 0,
    uids: list[str] | None = None,
):
    """Assemble collected ``Trajectory`` objects into a verl-contract DataProto.

    Mirrors verl's own rollout output (agent_loop.py ``_postprocess``):
        prompts        [B, P]   left-padded prompt segment
        responses      [B, R]   right-padded response segment
        response_mask  [B, R]   1=policy token, 0=observation/pad
        input_ids      [B, P+R] prompts ++ responses
        attention_mask [B, P+R] real-token mask
        position_ids   [B, P+R] cumsum(attention_mask)-1
        rollout_log_probs [B, R] per response token (when available)
    non_tensor_batch carries uid / messages / bucket for downstream
    (reward, buffer ingest, advantage grouping by uid).

    Args:
        trajectories: list of ``rollout.session_pool.Trajectory`` (one per slot,
            per query) carrying ``response_token_ids`` / ``logprobs`` /
            ``meta['response_mask']`` / ``messages`` / ``bucket``.
        prompt_token_ids: the prompt ids for each trajectory (aligned by index),
            i.e. the tokenized seed/follow-up query the slot answered.
        pad_token_id: tokenizer pad id.
        uids: GRPO grouping id per trajectory (default: per-query group id).
    """
    import numpy as np
    import torch

    n = len(trajectories)
    assert len(prompt_token_ids) == n, "prompt_token_ids must align with trajectories"

    resp_ids = [list(t.response_token_ids) for t in trajectories]
    resp_masks = [
        list(t.meta.get("response_mask") or [1] * len(r)) for t, r in zip(trajectories, resp_ids, strict=True)
    ]
    logprobs = [list(t.logprobs or []) for t in trajectories]
    has_logprobs = all(len(lp) == len(r) for lp, r in zip(logprobs, resp_ids, strict=True)) and any(logprobs)

    P = max((len(p) for p in prompt_token_ids), default=1) or 1
    R = max((len(r) for r in resp_ids), default=1) or 1

    prompts = torch.empty((n, P), dtype=torch.long)
    responses = torch.empty((n, R), dtype=torch.long)
    response_mask = torch.zeros((n, R), dtype=torch.long)
    prompt_attn = torch.zeros((n, P), dtype=torch.long)
    resp_attn = torch.zeros((n, R), dtype=torch.long)
    rollout_lp = torch.zeros((n, R), dtype=torch.float32) if has_logprobs else None

    for i in range(n):
        p, r, m = prompt_token_ids[i], resp_ids[i], resp_masks[i]
        prompts[i] = torch.tensor(_left_pad(p, P, pad_token_id), dtype=torch.long)
        prompt_attn[i, P - len(p) :] = 1
        responses[i] = torch.tensor(_right_pad(r, R, pad_token_id), dtype=torch.long)
        resp_attn[i, : len(r)] = 1
        response_mask[i, : len(m)] = torch.tensor(m[:R], dtype=torch.long)
        if rollout_lp is not None:
            rollout_lp[i, : len(logprobs[i])] = torch.tensor(logprobs[i][:R], dtype=torch.float32)

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_attn, resp_attn], dim=1)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    tensors = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if rollout_lp is not None:
        tensors["rollout_log_probs"] = rollout_lp

    # non-tensor: uid (GRPO grouping), messages (transcript/bucket), bucket
    if uids is None:
        uids = [str(t.meta.get("session_id", i)) for i, t in enumerate(trajectories)]
    non_tensor = {
        "uid": np.array(uids, dtype=object),
        "messages": np.array([t.messages for t in trajectories], dtype=object),
        "bucket": np.array([t.bucket for t in trajectories], dtype=object),
    }

    from verl import DataProto  # lazy: only needed on the cluster

    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensor)


def extract_queries_from_prompts(prompts, tokenizer) -> list[str]:
    """Decode the seed query text from a verl gen_batch DataProto.

    verl's ``gen_batch`` carries left-padded prompt ``input_ids`` (+ ``raw_prompt``
    messages when available). We prefer ``raw_prompt`` (the chat messages) and
    fall back to decoding ``input_ids``. Returns one query string per row.
    """
    import numpy as np

    raw = None
    nt = getattr(prompts, "non_tensor_batch", {}) or {}
    if "raw_prompt" in nt:
        raw = nt["raw_prompt"]
    queries: list[str] = []
    if raw is not None:
        for item in raw:
            if isinstance(item, (list, np.ndarray)) and len(item):
                last = item[-1]
                queries.append(last.get("content", "") if isinstance(last, dict) else str(last))
            else:
                queries.append(str(item))
        return queries
    # fallback: decode input_ids
    ids = prompts.batch["input_ids"]
    am = prompts.batch.get("attention_mask")
    for i in range(len(ids)):
        row = ids[i]
        if am is not None:
            row = row[am[i].bool()]
        queries.append(tokenizer.decode(row.tolist()))
    return queries


# --- the custom manager (verl AgentLoopManager subclass) ----------------------


def make_cl_scheduler_manager_cls():
    """Build the ``CLSchedulerAgentLoopManager`` class (verl import deferred).

    Returns a subclass of verl's ``AgentLoopManager`` named ``AgentLoopManager``
    (``load_class_from_fqn`` looks up that attribute name). Set in yaml::

        actor_rollout_ref:
          rollout:
            agent:
              agent_loop_manager_class: trainer.cl_rollout_manager.AgentLoopManager
    """
    from verl.experimental.agent_loop import AgentLoopManager as _Base

    from rollout.collect import make_hermes_agent_fn
    from rollout.scheduler import RolloutScheduler, SessionSpec

    class CLSchedulerAgentLoopManager(_Base):
        """Route rollout through our 16×8 + winner-sync scheduler.

        With Questioner + Observer support: each session runs one seed query
        through the full simulated-session loop (run_simulated_session), generating
        follow-up queries online and observing winner state via diff.
        """

        def _build_scheduler(self) -> Any:
            from agents.observer import Observer
            from agents.questioner import Questioner

            rcfg = self.rollout_config
            agent_cfg = rcfg.get("agent", {}) or {}

            agent_fn = make_hermes_agent_fn(
                model=str(agent_cfg.get("model", "qwen3-8b")),
                model_base=str(agent_cfg.get("model_base", "")),
                max_turns=int(rcfg.get("multi_turn", {}).get("max_turns", 16)),
                timeout=int(agent_cfg.get("timeout", 600)),
            )
            observer = Observer(use_llm=False)  # deterministic diff-driven, no model call
            questioner = Questioner()
            return RolloutScheduler(
                agent_fn,
                sessions_per_step=int(agent_cfg.get("sessions_per_step", 16)),
                slots=int(rcfg.get("n", 8)),
                backend=agent_cfg.get("sandbox_backend", "e2b"),
                simulated=True,
                observer=observer,
                questioner=questioner,
                k_max=int(agent_cfg.get("k_max", 3)),
                score_followups=bool(agent_cfg.get("score_followups", True)),
            )

        async def generate_sequences(self, prompts):  # type: ignore[override]
            # Decode the seed queries, run the winner-sync scheduler, assemble back.
            import asyncio

            tokenizer = getattr(self, "tokenizer", None)
            queries = extract_queries_from_prompts(prompts, tokenizer)
            scheduler = self._build_scheduler()
            specs = [SessionSpec(session_id=str(i), queries=[q]) for i, q in enumerate(queries)]

            # scheduler.run_step is sync (thread pool of sessions); run off the loop.
            trajectories = await asyncio.to_thread(scheduler.run_step, specs)

            prompt_ids = [
                list(tokenizer.apply_chat_template(t.messages[:1], tokenize=True, add_generation_prompt=True))
                for t in trajectories
            ]
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            return trajectories_to_dataproto(trajectories, prompt_ids, pad_token_id=pad_id)

    return CLSchedulerAgentLoopManager


# verl's load_class_from_fqn(fqn, "AgentLoopManager") looks up this attribute.
# Building the class eagerly would import verl at module import; instead expose a
# module-level __getattr__ so ``AgentLoopManager`` resolves lazily on the cluster.
def __getattr__(name: str):
    if name == "AgentLoopManager":
        return make_cl_scheduler_manager_cls()
    raise AttributeError(name)
