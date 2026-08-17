"""离线单测：回放 concat 字段对齐逻辑（plan_replay_fields，RunLog §64/§65）。

不依赖 torch/tensordict/transfer_queue（本机没装）—— 纯验证"回放行该带哪些字段、
哪些复用哪些补零"，提前抓字段对齐 bug，减少集群往返。

concat 要求（transfer_queue 实测）：回放 chunk 字段集必须 == rollout(batch.fields)。
故回放 plan 的 key 集合应恰好 = rollout 字段 ∪ 回放专属 3 字段。
"""

from __future__ import annotations

from trainer.cl_replay_hook_v1 import _REPLAY_ONLY_FIELDS, plan_replay_fields, replay_zero_fill_kind

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


# --- 补零 dtype/形状分类（r0 16:03 dtype mismatch 的根因回归测试）------------------


def test_zero_fill_kind_sequence_long():
    """2D int64 序列字段（含 loss_mask）→ long_seq，补零必须 torch.long。"""
    for f in (
        "prompts",
        "responses",
        "response_mask",
        "loss_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
    ):
        assert replay_zero_fill_kind(f) == "long_seq", f


def test_zero_fill_kind_float_sequence():
    """2D float32 序列字段 → float（含 compute 阶段写回的 returns/entropy/token_level_*）。"""
    for f in (
        "rollout_log_probs",
        "rm_scores",
        "old_log_probs",
        "ref_log_prob",
        "advantages",
        "returns",
        "entropy",
        "token_level_scores",
        "token_level_rewards",
    ):
        assert replay_zero_fill_kind(f) == "float", f


def test_zero_fill_kind_scalar_long():
    """1D int64 标量张量字段 → long_scalar（形状 [n]，非 [n,R]）。"""
    assert replay_zero_fill_kind("num_turns") == "long_scalar"


def test_zero_fill_kind_int_scalar():
    """python int 元数据 → int_scalar（补真实 int，非空串，避免下游 int-str 崩）。"""
    for f in ("global_steps", "session_id", "min_global_steps", "max_global_steps"):
        assert replay_zero_fill_kind(f) == "int_scalar", f


def test_zero_fill_kind_empty_dict():
    """multi_modal_inputs → empty_dict（补 {}，非空串，避免 forward .get 崩）。"""
    assert replay_zero_fill_kind("multi_modal_inputs") == "empty_dict"


def test_zero_fill_kind_non_tensor():
    """纯文本非张量字段（raw_prompt/extra_info/uid/...）→ non_tensor，补空串 NonTensorStack。"""
    for f in (
        "raw_prompt",
        "extra_info",
        "uid",
        "data_source",
        "reward_model",
        "trace_type",
        "tools",
        "tools_kwargs",
        "agent_name",
        "env_name",
        "routed_experts",
    ):
        assert replay_zero_fill_kind(f) == "non_tensor", f


def test_zero_fill_kind_unknown_defaults_float():
    """未分类的字段默认 float（保守，避免漏网 dtype 崩成 long）。"""
    assert replay_zero_fill_kind("some_new_field") == "float"
