"""Custom rollout manager: route verl's rollout through our session scheduler.

verl 0.8.0 exposes an OFFICIAL injection point for replacing rollout
(``ray_trainer.py:931`): set
``actor_rollout_ref.rollout.agent.agent_loop_manager_class`` to an FQN that
``load_class_from_fqn(..., "AgentLoopManager")`` resolves; verl then uses it
instead of the default ``AgentLoopManager`` -- WITHOUT touching ``fit()``.

We subclass ``AgentLoopManager`` and override only ``generate_sequences``:

    verl fit() ── generate_sequences(prompts: DataProto) ──►  CLSchedulerAgentLoopManager
                    (prompts already ×n: verl repeated each query by rollout.n)  │
                    per input ROW → 1 single-slot single-turn rollout            │
                      each step → llm_client.generate (token_ids + log_probs)    │
                      Observer diff per row → meta['observer_report']            ▼
                    list[Trajectory] ── trajectories_to_dataproto ──► DataProto (verl contract)

CONTRACT (bug fix 2026-07-27): verl OWNS the ×n repeat and GRPO grouping. It
repeats each query by ``rollout.n`` (interleave) BEFORE calling us and groups
advantages by ITS OWN ``uid``. So we return EXACTLY one trajectory per input row,
in order, and never emit our own ``uid`` (would collide on ``union``). The OLD
code ran an 8-slot pool PER input row -- ×n on top of verl's ×n = ×n² rows --
which broke the row-count contract and gave every baseline 0 checkpoints (it
crashed at ``_validate``/first step before finishing any training step).

Single-turn: the Questioner/winner-sync path is a no-op (one query per session);
it is the future multi-turn re-enable path, NOT a reason to multiply rows here.

The per-step generation reuses verl's NATIVE rollout LLM server
(``llm_client.generate`` -> token_ids + log_probs) via
``inference.VerlRolloutGenerateFn`` -- no HTTP proxy. The Observer's per-row diff
is carried back on ``observer_report`` and folded into the training judge's
rubric by ``trainer/observer_reward_manager.py`` (the observer never scores; it
supplies ground-truth state evidence).

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
    observer_reports: list[str] | None = None,
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
    non_tensor_batch carries messages / bucket / observer_report for downstream
    (transcript, bucket, and the observer diff evidence the training judge reads).

    verl OWNS the row identity and GRPO grouping: it repeats the gen batch by
    ``rollout.n`` BEFORE calling us (ray_trainer.py:1398, interleave) and groups
    advantages by ITS OWN ``uid`` (dataset uid, repeated ×8). We therefore must
    return EXACTLY ``len(trajectories) == len(input rows)`` rows, in input order,
    and must NOT emit ``uid`` -- a uid we invent would collide with verl's on
    ``union`` (union_numpy_dict asserts conflicting keys are deep-equal) and crash.
    ``uids`` is accepted for signature parity / offline callers but is NOT written
    to the DataProto unless explicitly passed (cold-collect / tests).

    Args:
        trajectories: list of ``rollout.session_pool.Trajectory`` (one per input
            row) carrying ``response_token_ids`` / ``logprobs`` /
            ``meta['response_mask']`` / ``messages`` / ``bucket``.
        prompt_token_ids: the prompt ids for each trajectory (aligned by index).
        pad_token_id: tokenizer pad id.
        uids: OPTIONAL explicit grouping id per trajectory. Only written to the
            DataProto when non-None (off-cluster/cold paths). In verl training
            leave it None so verl's own uid drives GRPO grouping.
        observer_reports: per-row observer diff/state evidence (str). Carried as a
            NEW non_tensor key ``observer_report`` (never ``extra_info`` -- that
            key belongs to the dataset and would collide on union). The custom
            reward manager folds it into extra_info for the training judge.
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
    # rollout_log_probs is emitted only when EVERY row has a length-matched
    # logprob vector (verl consumes it as a dense [B, R] tensor -- a single
    # ragged row would misalign the whole batch). But dropping it silently
    # degrades the GRPO importance ratio for ALL 512 rows because of one bad
    # slot, with no trace. Log loudly which rows are ragged so the cause is
    # diagnosable instead of a mystery reward/ratio drift.
    _mismatched = [i for i, (lp, r) in enumerate(zip(logprobs, resp_ids, strict=True)) if len(lp) != len(r)]
    has_logprobs = not _mismatched and any(logprobs)
    if _mismatched:
        print(
            f"[rollout] WARNING: {len(_mismatched)}/{n} trajectories have logprob "
            f"length != response length (rows {_mismatched[:8]}"
            f"{'...' if len(_mismatched) > 8 else ''}); dropping rollout_log_probs "
            "for the WHOLE batch -> GRPO will recompute old_log_probs. Investigate "
            "the generate backend (inference/generate.py) or a crashed slot.",
            flush=True,
        )

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

    # non_tensor: messages (transcript/bucket), bucket, observer_report (diff
    # evidence for the training judge). uid is emitted ONLY when explicitly
    # passed -- in verl training it stays absent so verl's own uid (dataset uid
    # repeated ×n) drives GRPO grouping and no union collision occurs.
    non_tensor: dict[str, Any] = {
        "messages": np.array([t.messages for t in trajectories], dtype=object),
        "bucket": np.array([t.bucket for t in trajectories], dtype=object),
        # multi_modal_inputs: verl's fit() unconditionally iterates
        # batch.non_tensor_batch["multi_modal_inputs"] (ray_trainer.py:1463) after
        # rollout. verl's OWN default AgentLoopManager only sets this key when a
        # sample actually has multi-modal data (agent_loop.py:953
        # `if any(mmi is not None)`), so text-only + custom-rollout hits
        # KeyError: 'multi_modal_inputs'. We are TEXT-ONLY by design (195 tasks,
        # no multimodal -- see CLAUDE.md), so the semantically-correct value is an
        # empty dict per row = "this row has no multi-modal input". verl's loop
        # does `if "image_grid_thw" not in mmi: continue`, so {} is skipped
        # cleanly and images_seqlens stays empty -- exactly the text-only truth.
        # This is a contract placeholder, NOT fabricated data: if real multimodal
        # is ever added, these empty dicts must be replaced with actual inputs
        # (they will stand out precisely because they are empty).
        "multi_modal_inputs": np.array([{} for _ in trajectories], dtype=object),
    }
    if observer_reports is not None:
        assert len(observer_reports) == n, "observer_reports must align with trajectories"
        non_tensor["observer_report"] = np.array([str(x or "") for x in observer_reports], dtype=object)
    if uids is not None:
        assert len(uids) == n, "uids must align with trajectories"
        non_tensor["uid"] = np.array([str(u) for u in uids], dtype=object)

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
    from verl.utils.ray_utils import auto_await  # verl 基类用它让 async generate_sequences 可同步调

    from rollout.collect import make_react_agent_fn
    from rollout.scheduler import RolloutScheduler, SessionSpec

    class CLSchedulerAgentLoopManager(_Base):
        """Route rollout through our per-row single-turn scheduler.

        verl OWNS the ×n repeat and GRPO grouping: it repeats each query by
        ``rollout.n`` (ray_trainer.py:1398, interleave) BEFORE calling us and
        groups advantages by its own ``uid``. So ``generate_sequences`` receives
        an already-repeated batch (n_query × n rows) and must return EXACTLY that
        many trajectories, one per input row, in order. We therefore run ONE
        single-slot single-turn rollout per input row -- NOT an 8-slot pool per
        query (that double-counted ×n → ×n² and crashed the row-count contract).

        The Observer still runs per row (diff-driven, deterministic) and its
        state evidence is carried back on each trajectory's ``meta['observer_report']``
        so the TRAINING judge (custom reward manager) can ground completion on it.
        Winner-sync / multi-turn Questioner is a no-op under single-turn and is
        the future path if multi-turn is re-enabled.
        """

        def _build_scheduler(self) -> Any:
            from agents.observer import Observer
            from agents.questioner import Questioner

            rcfg = self.rollout_config
            agent_cfg = rcfg.get("agent", {}) or {}

            # 训练 rollout：agent 在沙箱里跑 ReAct，但每步生成走 verl 的 LLM server
            # （lightllm 后端）→ token_ids + log_probs 原生带回（GRPO 必需）。不能用
            # make_hermes_agent_fn（那是冷采集的 CLI stdout 路径，response_token_ids=[]）。
            from inference.generate import VerlRolloutGenerateFn
            gen_fn = VerlRolloutGenerateFn(
                self.llm_client,
                self._get_tokenizer(),
                sampling_params={
                    "temperature": float(rcfg.get("temperature", 1.0)),
                    "max_tokens": int(rcfg.get("response_length", 1024) or 1024),
                },
            )
            agent_fn = make_react_agent_fn(
                gen_fn,
                max_turns=int(rcfg.get("multi_turn", {}).get("max_turns", 16)),
            )
            observer = Observer(use_llm=False)  # deterministic diff-driven, no model call
            questioner = Questioner()
            # slots=1: each input row is ONE independent single-turn rollout. verl
            # already repeated the query ×n, so the n GRPO samples of a query are n
            # separate input rows here (n separate 1-slot sessions), not n slots of
            # one pool. sessions_per_step = concurrency cap over rows.
            return RolloutScheduler(
                agent_fn,
                sessions_per_step=int(agent_cfg.get("sessions_per_step", 64)),
                slots=1,
                backend=agent_cfg.get("sandbox_backend", "e2b"),
                simulated=True,
                observer=observer,
                questioner=questioner,
                k_max=int(agent_cfg.get("k_max", 8)),
                score_followups=bool(agent_cfg.get("score_followups", True)),
            )

        def _get_tokenizer(self):
            # verl 的 AgentLoopManager（本类的基类）不持有 tokenizer（它在 worker 层），
            # self.tokenizer 恒为 None。从 model_config.path lazy 加载并缓存。
            tk = getattr(self, "_cl_tokenizer", None)
            if tk is None:
                from transformers import AutoTokenizer
                path = self.model_config.path
                tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                self._cl_tokenizer = tk
            return tk

        @auto_await
        async def generate_sequences(self, prompts):  # type: ignore[override]
            # PER-ROW single-turn rollout (2026-07-27). verl already repeated each
            # query by rollout.n (interleave) BEFORE calling us, so `prompts` holds
            # n_query × n rows and verl expects EXACTLY that many trajectories back,
            # one per row, in order (ray_trainer union asserts equal row count; GRPO
            # groups by verl's own uid). We run ONE single-slot single-turn rollout
            # per input row. The OLD code ran an 8-slot pool per row → ×n again →
            # ×n² rows → the row-count crash that gave every baseline 0 checkpoints.
            import asyncio

            tokenizer = self._get_tokenizer()
            queries = extract_queries_from_prompts(prompts, tokenizer)
            scheduler = self._build_scheduler()

            # One SessionSpec per input row; scheduler runs up to
            # sessions_per_step in parallel (each a 1-slot single-turn session ->
            # exactly 1 trajectory). Batched so we never spawn all N sandboxes at
            # once. Order is preserved: batch k covers rows [k*S : (k+1)*S].
            step = max(1, int(scheduler.sessions_per_step))
            all_trajs: list[Any] = []
            for start in range(0, len(queries), step):
                chunk = queries[start : start + step]
                specs = [
                    SessionSpec(session_id=str(start + j), queries=[q]) for j, q in enumerate(chunk)
                ]
                chunk_trajs = await asyncio.to_thread(scheduler.run_step, specs)
                all_trajs.extend(chunk_trajs)

            # verl contract: 1 trajectory per input row, same order. `prompts.batch`
            # is always present after `_get_gen_batch` (train AND val), so compare
            # against it directly. A mismatch means a row's rollout was dropped
            # (crashed slot / isolated session error) -- that is a bug to surface,
            # not data to pad.
            expected_n = len(prompts.batch) if prompts.batch is not None else len(queries)
            assert len(all_trajs) == expected_n, (
                f"per-row yield {len(all_trajs)} != verl expected {expected_n} "
                f"(queries={len(queries)}); a row's rollout was dropped "
                "(isolated session error?). Investigate rollout/scheduler.run_step logs."
            )
            trajectories = all_trajs

            def _safe_tokenize(messages):
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                # Qwen3.5 返回 BatchEncoding（非 dict 子类），需提取 input_ids
                if hasattr(ids, "get") and "input_ids" in ids:
                    ids = ids["input_ids"]
                elif isinstance(ids, dict):
                    ids = ids["input_ids"]
                # If a mis-set chat template returns the templated TEXT instead of
                # ids (tokenize=True ignored), list(str) yields a list of single
                # CHARACTERS -> torch.tensor(..., long) later dies with the opaque
                # "too many dimensions 'str'" (the exact failure that killed the
                # 07-23 baseline at _validate). Re-encode the string here instead.
                if isinstance(ids, str):
                    ids = tokenizer.encode(ids, add_special_tokens=False)
                ids = list(ids)
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = list(ids[0])
                # Final guard: every element MUST be an int token id. A stray str
                # (nested token-string list, or the char-list case above) would
                # otherwise reach torch.tensor and crash mid-run with no context.
                if any(not isinstance(x, int) for x in ids):
                    raise TypeError(
                        f"_safe_tokenize produced non-int token ids "
                        f"(sample: {ids[:8]!r}); apply_chat_template likely "
                        "returned text instead of ids -- check the tokenizer's "
                        "chat_template / tokenize handling."
                    )
                return ids

            # Prompt tokens per row. Prefer the trajectory's first message, but a
            # FAILED single-slot rollout returns a Trajectory with EMPTY messages
            # (scheduler's per-slot error fallback), so t.messages[:1] == [] and
            # apply_chat_template([]) dies with IndexError deep in transformers
            # (the 10:05 "success"-but-0-step crash). trajectories align 1:1 with
            # `queries` by index, so fall back to the KNOWN input query -- the
            # prompt is deterministic input, not something a failed rollout can
            # lose. This keeps the row (its empty response is handled downstream)
            # instead of crashing the whole step on one bad slot.
            def _prompt_msgs(traj, idx):
                m = traj.messages[:1] if getattr(traj, "messages", None) else []
                if m:
                    return m
                q = queries[idx] if idx < len(queries) else ""
                return [{"role": "user", "content": str(q)}]

            prompt_ids = [_safe_tokenize(_prompt_msgs(t, i)) for i, t in enumerate(trajectories)]
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            # Carry each row's observer diff evidence so the training judge can
            # ground completion on it. Do NOT pass uids -- verl's own uid drives
            # GRPO grouping and an invented uid would collide on union.
            observer_reports = [str(t.meta.get("observer_report", "") or "") for t in trajectories]
            out = trajectories_to_dataproto(
                trajectories,
                prompt_ids,
                pad_token_id=pad_id,
                observer_reports=observer_reports,
            )
            # verl's fit() does `timing_raw.update(gen_output.meta_info["timing"])`
            # right after rollout (ray_trainer.py:1425) and the default
            # AgentLoopManager always returns meta_info={"timing": {...}, ...}
            # (agent_loop.py:1091). Our custom manager must honour that contract or
            # fit() dies with KeyError: 'timing' AFTER a full rollout (the 08:37
            # crash). We don't have verl's per-request perf breakdown, so emit an
            # empty timing dict -- update() with {} is a no-op, keeps the key present.
            try:
                out.meta_info = {**getattr(out, "meta_info", {}), "timing": {}}
            except Exception:  # noqa: BLE001 -- fake DataProto in off-cluster tests
                pass
            return out

    return CLSchedulerAgentLoopManager


# verl's load_class_from_fqn(fqn, "AgentLoopManager") looks up this attribute.
# Building the class eagerly would import verl at module import; instead expose a
# module-level __getattr__ so ``AgentLoopManager`` resolves lazily on the cluster.
def __getattr__(name: str):
    if name == "AgentLoopManager":
        return make_cl_scheduler_manager_cls()
    raise AttributeError(name)
