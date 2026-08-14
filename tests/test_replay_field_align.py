"""离线单测：回放 concat 字段对齐逻辑（plan_replay_fields，RunLog §64/§65）。

不依赖 torch/tensordict/transfer_queue（本机没装）—— 纯验证"回放行该带哪些字段、
哪些复用哪些补零"，提前抓字段对齐 bug，减少集群往返。

concat 要求（transfer_queue 实测）：回放 chunk 字段集必须 == rollout(batch.fields)。
故回放 plan 的 key 集合应恰好 = rollout 字段 ∪ 回放专属 3 字段。
"""

from __future__ import annotations

from trainer.cl_replay_hook_v1 import _REPLAY_ONLY_FIELDS, plan_replay_fields

# 模拟 v1 session_worker 写 tq 的 rollout 字段集（worker.py:1055 起）。
_ROLLOUT_FIELDS = [
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
    "old_log_probs",  # verl compute 阶段追加
    "advantages",
]

# build_replay_rows 产出的回放字段（含 3 个 replay 专属）。
_REPLAY_FIELDS = [
    "prompts",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "response_mask",
    "old_log_probs",
    "ref_log_prob",
    "advantages",
    "is_replay",
    "replay_response_mask",
    "replay_token_weights",
]


def test_plan_key_set_equals_rollout_union_replay_only():
    """concat 通过的充要条件：回放 plan 的 key 集 == rollout 字段 ∪ 3 replay 专属。"""
    plan = plan_replay_fields(_REPLAY_FIELDS, _ROLLOUT_FIELDS)
    assert plan is not None
    expected = set(_ROLLOUT_FIELDS) | set(_REPLAY_ONLY_FIELDS)
    assert set(plan.keys()) == expected, set(plan.keys()) ^ expected


def test_rollout_field_replay_has_is_use():
    """rollout 声明且回放也有的字段 → 复用回放值（use）。"""
    plan = plan_replay_fields(_REPLAY_FIELDS, _ROLLOUT_FIELDS)
    for f in ("prompts", "responses", "input_ids", "response_mask", "old_log_probs", "advantages"):
        assert plan[f] == "use", f


def test_rollout_field_replay_missing_is_zero():
    """rollout 声明但回放没有的字段（如 loss_mask/rollout_log_probs/rm_scores）→ 补零。"""
    plan = plan_replay_fields(_REPLAY_FIELDS, _ROLLOUT_FIELDS)
    for f in ("loss_mask", "rollout_log_probs", "rm_scores"):
        assert plan[f] == "zero", f


def test_replay_only_fields_present_and_use():
    """3 个 replay 专属字段必须在 plan 里且为 use（rollout 侧另补零对齐）。"""
    plan = plan_replay_fields(_REPLAY_FIELDS, _ROLLOUT_FIELDS)
    for f in _REPLAY_ONLY_FIELDS:
        assert plan.get(f) == "use", f


def test_no_batch_fields_returns_none():
    """拿不到 rollout 字段集（本机/降级）→ None（调用方走旧行为，直接用回放全字段）。"""
    assert plan_replay_fields(_REPLAY_FIELDS, None) is None
    assert plan_replay_fields(_REPLAY_FIELDS, []) is None


def test_replay_missing_some_replay_only():
    """回放若缺某个 replay 专属字段，plan 不硬塞它（只放回放实际有的）。"""
    partial = [f for f in _REPLAY_FIELDS if f != "replay_token_weights"]
    plan = plan_replay_fields(partial, _ROLLOUT_FIELDS)
    assert "replay_token_weights" not in plan
    assert plan["is_replay"] == "use"
