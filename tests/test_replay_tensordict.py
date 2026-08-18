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
    # attention_mask 全 1（无 left/right pad），让 _replay_tensordict 能恢复每行实际长度
    # （否则切成 0 长 nested）。
    return {
        "prompts": torch.zeros((n, P), dtype=torch.long),
        "responses": torch.zeros((n, R), dtype=torch.long),
        "input_ids": torch.zeros((n, P + R), dtype=torch.long),
        "attention_mask": torch.ones((n, P + R), dtype=torch.long),
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


def test_replay_tensordict_sequence_fields_are_nested():
    """序列字段必须是 nested（变长），对齐 v1 rollout 的 nested 格式（GDN 崩根因回归）。

    之前回放行写成固定 2D padded，与 rollout 的 nested 混在一起 → remove_padding 把
    padded 宽当序列长 → cu_seqlens 错乱 → GDN kernel "invalid argument"（r0 step2）。
    """
    td = _replay_tensordict(_make_replay_rows(), _FakeBatch())
    for k in (
        "prompts",
        "responses",
        "response_mask",
        "loss_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rollout_log_probs",
        "rm_scores",
        "replay_response_mask",
        "replay_token_weights",
    ):
        assert td[k].is_nested, f"{k} 应该是 nested tensor，实际 {td[k].shape}"
    # 标量字段不该 nested
    assert not td["is_replay"].is_nested
    assert not td["num_turns"].is_nested


def test_replay_tensordict_variable_lengths_preserved():
    """变长切分后每行的真实长度保留（左 pad prompt / 右 pad response 被正确去掉）。"""
    n, P, R = 3, 5, 4
    rows = _make_replay_rows(n=n, P=P, R=R)
    # 手动构造不同长度：行0 prompt=3/response=2，行1 全满，行2 prompt=1/response=1。
    attn = rows["attention_mask"]
    resp = rows["responses"]
    for i, (pl, rl) in enumerate([(3, 2), (5, 4), (1, 1)]):
        attn[i, :P] = 0
        attn[i, P - pl : P] = 1
        attn[i, P:] = 0
        attn[i, P : P + rl] = 1
        resp[i, :] = 0
        resp[i, :rl] = 2
    td = _replay_tensordict(rows, _FakeBatch())
    # 每行 response 长度 = [2, 4, 1]
    resp_lens = [r.shape[0] for r in td["responses"].unbind()]
    assert resp_lens == [2, 4, 1], resp_lens
    # 每行 input_ids 长度 = prompt+response = [5, 9, 2]
    seq_lens = [r.shape[0] for r in td["input_ids"].unbind()]
    assert seq_lens == [5, 9, 2], seq_lens
