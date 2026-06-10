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
    REPLAY_WEIGHTS_KEY,
    align_token_weights,
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
    losses = _ensure_verl_losses_mock()
    loss_fn = make_cl_loss(replay_enabled=False, lambda_replay=0.0)
    losses.ppo_loss = MagicMock(return_value=(torch.tensor(1.5), {"actor/pg_loss": 1.5}))
    total, metrics = loss_fn(MagicMock(), {"log_probs": torch.zeros(2, 4)}, MagicMock())
    assert total.item() == pytest.approx(1.5)
    assert metrics["actor/replay_enabled"] == 0.0


def test_align_token_weights_pads_and_tail_aligns():
    # 3 rows total, 1 replay traj of length 2 -> last row filled, others zero.
    out = align_token_weights([[0.5, 0.25]], num_rows=3, seq_len=4)
    assert out.shape == (3, 4)
    assert out[0].sum().item() == 0.0
    assert out[2, :2].tolist() == pytest.approx([0.5, 0.25])
    assert out[2, 2:].sum().item() == 0.0


def test_select_replay_rows_is_differentiable_and_weighted():
    log_probs = torch.tensor([[0.0, 0.0], [-1.0, -2.0]], requires_grad=True)
    response_mask = torch.tensor([[True, True], [True, True]])
    weights = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
    is_replay = torch.tensor([False, True])
    loss = select_replay_rows(log_probs, response_mask, weights, is_replay)
    # -(-1)*2 + -(-2)*1 = 2 + 2 = 4, mean over 2 tokens = 2
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    # gradient must flow into the replay row only
    assert log_probs.grad[0].abs().sum().item() == pytest.approx(0.0)
    assert log_probs.grad[1].abs().sum().item() > 0.0


def test_compute_replay_loss_reads_model_output_and_data():
    _ensure_verl_losses_mock()
    model_output = {"log_probs": torch.tensor([[0.0, 0.0], [-1.0, -1.0]])}
    data = {
        IS_REPLAY_KEY: torch.tensor([False, True]),
        "response_mask": torch.tensor([[True, True], [True, True]]),
        REPLAY_WEIGHTS_KEY: torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    }
    loss = compute_replay_loss(model_output, data)
    assert loss.item() == pytest.approx(1.0)


def test_cl_loss_with_replay_adds_weighted_term():
    losses = _ensure_verl_losses_mock()
    loss_fn = make_cl_loss(replay_enabled=True, lambda_replay=0.5, weighting_scheme="W0")
    data = MagicMock()
    losses.ppo_loss = MagicMock(return_value=(torch.tensor(2.0), {}))
    with patch("trainer.cl_loss._replay_is_empty", return_value=False):
        with patch("trainer.cl_loss.compute_replay_loss", return_value=torch.tensor(1.0)):
            total, metrics = loss_fn(MagicMock(), {}, data)
    assert total.item() == pytest.approx(2.0 + 0.5 * 1.0)
    assert metrics["actor/replay_empty"] == 0.0


def test_cl_loss_replay_empty_skips_forward():
    losses = _ensure_verl_losses_mock()
    loss_fn = make_cl_loss(replay_enabled=True, lambda_replay=0.5)
    data = MagicMock()
    losses.ppo_loss = MagicMock(return_value=(torch.tensor(1.0), {}))
    with patch("trainer.cl_loss._replay_is_empty", return_value=True):
        total, metrics = loss_fn(MagicMock(), {}, data)
    assert total.item() == pytest.approx(1.0)
    assert metrics["actor/replay_empty"] == 1.0
