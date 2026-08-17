"""Tests for cl_loss composition and differentiable replay aggregation."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest

from trainer.cl_loss import compute_replay_loss, make_cl_loss
from trainer.replay_forward import (
    IS_REPLAY_KEY,
    REPLAY_MASK_KEY,
    REPLAY_WEIGHTS_KEY,
    align_token_weights,
    build_replay_rows,
    pad_rows_to_seq_len,
    select_replay_rows,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    import torch


def _ensure_verl_losses_mock():
    """Allow cl_loss tests without a full verl install."""
    if importlib.util.find_spec("verl") is not None:
        import verl.workers.utils.losses as losses_mod

        return losses_mod
    import types

    def _module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__path__ = []  # type: ignore[attr-defined]
        return mod

    verl = _module("verl")
    workers = _module("verl.workers")
    utils = _module("verl.workers.utils")
    losses = _module("verl.workers.utils.losses")
    verl_utils = _module("verl.utils")
    td_utils = _module("verl.utils.tensordict_utils")
    td_utils.get_non_tensor_data = lambda data, key, default=None: default

    utils.losses = losses
    workers.utils = utils
    verl.workers = workers
    verl.utils = verl_utils
    verl_utils.tensordict_utils = td_utils

    for name, mod in [
        ("verl", verl),
        ("verl.workers", workers),
        ("verl.workers.utils", utils),
        ("verl.workers.utils.losses", losses),
        ("verl.utils", verl_utils),
        ("verl.utils.tensordict_utils", td_utils),
    ]:
        sys.modules[name] = mod
    return losses


def test_make_cl_loss_zero_replay_uses_no_replay_branch():
    # New closure signature (bug-1 fix): (model_output, data, dp_group) -- NO config.
    # verl absent -> RL term falls back to a 0 scalar; we assert the replay metrics.
    _ensure_verl_losses_mock()
    loss_fn = make_cl_loss(replay_enabled=False, lambda_replay=0.0)
    total, metrics = loss_fn({"log_probs": torch.zeros(2, 4)}, MagicMock(), None)
    assert metrics["actor/replay_enabled"] == 0.0
    assert metrics["actor/replay_loss"] == 0.0


def test_align_token_weights_pads_and_tail_aligns():
    # 3 rows total, 1 replay traj of length 2 -> last row filled, others zero.
    out = align_token_weights([[0.5, 0.25]], num_rows=3, seq_len=4)
    assert out.shape == (3, 4)
    assert out[0].sum().item() == 0.0
    assert out[2, :2].tolist() == pytest.approx([0.5, 0.25])
    assert out[2, 2:].sum().item() == 0.0


def test_select_replay_rows_is_differentiable_and_weighted():
    log_probs = torch.tensor([[0.0, 0.0], [-1.0, -2.0]], requires_grad=True)
    # replay_response_mask: real span of the replay rows.
    replay_mask = torch.tensor([[True, True], [True, True]])
    weights = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
    is_replay = torch.tensor([False, True])
    loss = select_replay_rows(log_probs, replay_mask, weights, is_replay)
    # -(-1)*2 + -(-2)*1 = 2 + 2 = 4, mean over 2 tokens = 2
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    # gradient must flow into the replay row only
    assert log_probs.grad[0].abs().sum().item() == pytest.approx(0.0)
    assert log_probs.grad[1].abs().sum().item() > 0.0


def test_select_replay_rows_respects_real_span_mask():
    # replay row 1 has a real span of only its first token (mask [1, 0]).
    log_probs = torch.tensor([[0.0, 0.0], [-1.0, -5.0]], requires_grad=True)
    replay_mask = torch.tensor([[0, 0], [1, 0]])
    weights = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    is_replay = torch.tensor([False, True])
    loss = select_replay_rows(log_probs, replay_mask, weights, is_replay)
    # Only token 0 of the replay row counts: -(-1)*1 / 1 = 1.0 (the -5 is masked).
    assert loss.item() == pytest.approx(1.0)


def test_compute_replay_loss_reads_model_output_and_data():
    _ensure_verl_losses_mock()
    model_output = {"log_probs": torch.tensor([[0.0, 0.0], [-1.0, -1.0]])}
    data = {
        IS_REPLAY_KEY: torch.tensor([False, True]),
        # PPO mask is 0 for the replay row; L_replay must NOT read this.
        "response_mask": torch.tensor([[1, 1], [0, 0]]),
        REPLAY_MASK_KEY: torch.tensor([[0, 0], [1, 1]]),
        REPLAY_WEIGHTS_KEY: torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    }
    loss = compute_replay_loss(model_output, data)
    assert loss.item() == pytest.approx(1.0)


def test_pad_rows_to_seq_len_right_pads_2d_only():
    rows = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        IS_REPLAY_KEY: torch.ones(2, dtype=torch.bool),
    }
    out = pad_rows_to_seq_len(rows, target_seq_len=5)
    assert out["input_ids"].shape == (2, 5)
    assert out["input_ids"][:, 3:].sum().item() == 0  # zero padded
    assert out[IS_REPLAY_KEY].shape == (2,)  # 1D untouched


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # one token per message char count, deterministic
        return list(range(1, 1 + sum(len(m.get("content", "")) for m in messages)))


def test_build_replay_rows_two_mask_layout():
    samples = [
        ("t1", None, {"messages": [{"role": "assistant", "content": "abc"}]}),
        ("t2", None, {"messages": [{"role": "assistant", "content": "de"}]}),
    ]
    rows = build_replay_rows(samples, token_weights=None, tokenizer=_FakeTokenizer())
    n = rows["is_replay"].shape[0]
    assert n == 2
    # PPO response_mask is all zeros -> ppo_loss ignores replay rows (B8).
    assert rows["response_mask"].sum().item() == 0
    # The real span lives in replay_response_mask.
    assert rows[REPLAY_MASK_KEY].sum().item() > 0
    # Placeholders present (B10).
    assert "old_log_probs" in rows and "ref_log_prob" in rows
    assert bool(rows["is_replay"].all())
    # Trajectory-id sidecar aligned with built rows (for forgetting backfill).
    from trainer.replay_forward import REPLAY_TIDS_KEY

    assert rows[REPLAY_TIDS_KEY] == ["t1", "t2"]


def test_build_replay_rows_tids_skip_empty_messages():
    # Rows with empty messages are skipped; tids stay aligned with built rows.
    samples = [
        ("t1", None, {"messages": [{"role": "assistant", "content": "abc"}]}),
        ("t2", None, {"messages": []}),
        ("t3", None, {"messages": [{"role": "assistant", "content": "de"}]}),
    ]
    rows = build_replay_rows(samples, token_weights=None, tokenizer=_FakeTokenizer())
    from trainer.replay_forward import REPLAY_TIDS_KEY

    assert rows[REPLAY_TIDS_KEY] == ["t1", "t3"]


def test_build_replay_rows_messages_in_traj_payload():
    """冷启动 sqlite：messages 存在 traj payload dict 里、不在 meta。build_replay_rows
    应从 traj 回退取到 messages（经 replay_sample_to_metadata），不再整条跳过（回放空转 bug）。"""
    from trainer.replay_forward import REPLAY_TIDS_KEY

    samples = [
        # traj 是 payload dict（含 messages），meta 里没有 messages —— 冷启动 warmup 的真实结构
        (
            "cold1",
            {
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "abc"}],
                "response_token_ids": [],
                "response_mask": [],
            },
            {"bucket": "workflow"},
        ),
    ]
    rows = build_replay_rows(samples, token_weights=None, tokenizer=_FakeTokenizer())
    # 关键：这条不被跳过，成功建成回放行。
    assert rows[REPLAY_TIDS_KEY] == ["cold1"]
    assert rows["is_replay"].shape[0] == 1
    assert rows[REPLAY_MASK_KEY].sum().item() > 0  # response 段非空


def test_build_replay_rows_token_ids_clamp_to_model_len():
    """token-ids 路径：response 截到 max_length，prompt+response 总长受 max_model_len 约束。"""
    samples = [
        ("t1", None, {"prompt_token_ids": list(range(50)), "response_token_ids": list(range(300))}),
    ]
    # max_length(response 上限)=100, max_model_len(总)=120 → prompt50 全留, response 截到 120-50=70
    rows = build_replay_rows(
        samples, token_weights=None, tokenizer=_FakeTokenizer(), max_length=100, max_model_len=120
    )
    # responses 段宽 R = 实际 response 长度（此处 70），replay_response_mask 该有 70 个 1
    assert rows[REPLAY_MASK_KEY].sum().item() == 70
    assert rows["is_replay"].shape[0] == 1


def test_cl_loss_with_replay_adds_weighted_term():
    # Inject a fake RL loss via base_loss_fn (bug-1: closure takes no config).
    _ensure_verl_losses_mock()
    base = MagicMock(return_value=(torch.tensor(2.0), {}))
    loss_fn = make_cl_loss(replay_enabled=True, lambda_replay=0.5, weighting_scheme="W0", base_loss_fn=base)
    data = MagicMock()
    with patch("trainer.cl_loss._replay_is_empty", return_value=False):
        with patch("trainer.cl_loss.compute_replay_loss", return_value=torch.tensor(1.0)):
            total, metrics = loss_fn({}, data, None)
    assert total.item() == pytest.approx(2.0 + 0.5 * 1.0)
    assert metrics["actor/replay_empty"] == 0.0
    # RL term was called with verl's keyword convention (no config arg).
    base.assert_called_once()
    assert "config" not in base.call_args.kwargs


def test_cl_loss_replay_empty_skips_forward():
    _ensure_verl_losses_mock()
    base = MagicMock(return_value=(torch.tensor(1.0), {}))
    loss_fn = make_cl_loss(replay_enabled=True, lambda_replay=0.5, base_loss_fn=base)
    data = MagicMock()
    with patch("trainer.cl_loss._replay_is_empty", return_value=True):
        total, metrics = loss_fn({}, data, None)
    assert total.item() == pytest.approx(1.0)
    assert metrics["actor/replay_empty"] == 1.0
