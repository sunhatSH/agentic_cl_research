"""回放行 TensorDict 的 dtype/形状/非张量字段回归测试（需 torch + tensordict）。

r0 在集群上连崩四次，根因都在 `_replay_tensordict` 补零分支与 `_append_replay_rows_v1`
的 concat。本机装了 torch+tensordict 后可离线验证（无需 transfer_queue）：

  1. loss_mask 是 int64 但曾漏在 `_LONG_FIELDS` 外 → 补成 float32 → dtype mismatch 崩。
  2. num_turns 是 1D int64 但曾按 2D float 补 → 形状+dtype 崩。
  3. 非张量字段曾补 None → transfer_queue `_pack_field_values` 崩。

本测试只 import torch/tensordict（无则 skip），覆盖以上三类回归。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from trainer.cl_replay_hook_v1 import _replay_tensordict  # noqa: E402

# 模拟 v1 session_worker 写进 tq 的 rollout 字段集（含张量 + 非张量 + 标量）。
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
    "num_turns",
    "uid",
    "raw_prompt",
    "data_source",
    "reward_model",
    "extra_info",
    "tools_kwargs",
    "tools",
    "agent_name",
    "env_name",
    "trace_type",
    "multi_modal_inputs",
    "routed_experts",
    "min_global_steps",
    "max_global_steps",
    "session_id",
    "global_steps",
]


class _FakeBatch:
    fields = _ROLLOUT_FIELDS


def _make_replay_rows(n=4, P=10, R=8):
    return {
        "prompts": torch.zeros((n, P), dtype=torch.long),
        "responses": torch.zeros((n, R), dtype=torch.long),
        "input_ids": torch.zeros((n, P + R), dtype=torch.long),
        "attention_mask": torch.zeros((n, P + R), dtype=torch.long),
        "position_ids": torch.zeros((n, P + R), dtype=torch.long),
        "response_mask": torch.zeros((n, R), dtype=torch.long),
        "replay_response_mask": torch.zeros((n, R), dtype=torch.long),
        "old_log_probs": torch.zeros((n, R), dtype=torch.float32),
        "ref_log_prob": torch.zeros((n, R), dtype=torch.float32),
        "advantages": torch.zeros((n, R), dtype=torch.float32),
        "replay_token_weights": torch.zeros((n, R), dtype=torch.float32),
        "is_replay": torch.ones(n, dtype=torch.bool),
        "replay_tids": [f"t{i}" for i in range(n)],
    }


def test_replay_tensordict_int64_sequence_fields():
    """loss_mask 等 int64 序列字段必须补 int64（曾是 float32 → dtype mismatch 崩）。"""
    td = _replay_tensordict(_make_replay_rows(), _FakeBatch())
    assert td is not None
    for k in (
        "loss_mask",
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
    ):
        assert td[k].dtype == torch.int64, f"{k} dtype={td[k].dtype}"


def test_replay_tensordict_scalar_long_1d():
    """num_turns 是 1D int64，不是 2D float。"""
    td = _replay_tensordict(_make_replay_rows(), _FakeBatch())
    assert td["num_turns"].dtype == torch.int64
    assert td["num_turns"].dim() == 1


def test_replay_tensordict_float_sequence_fields():
    """rollout_log_probs / rm_scores 等 float 字段补 float32。"""
    td = _replay_tensordict(_make_replay_rows(), _FakeBatch())
    for k in ("rollout_log_probs", "rm_scores"):
        assert td[k].dtype == torch.float32, f"{k} dtype={td[k].dtype}"


def test_replay_tensordict_non_tensor_not_none():
    """非张量字段补空字符串，不能是 None（曾 None → _pack_field_values 崩）。"""
    td = _replay_tensordict(_make_replay_rows(), _FakeBatch())
    for k in (
        "uid",
        "raw_prompt",
        "data_source",
        "reward_model",
        "extra_info",
        "tools_kwargs",
        "tools",
        "agent_name",
        "env_name",
        "trace_type",
        "multi_modal_inputs",
        "routed_experts",
        "min_global_steps",
        "max_global_steps",
        "session_id",
        "global_steps",
    ):
        v = td[k]
        assert not isinstance(v, torch.Tensor), f"{k} 不该是张量"
        vals = list(v) if hasattr(v, "__iter__") else [v]
        assert all(x is not None for x in vals), f"{k} 含 None 值"
