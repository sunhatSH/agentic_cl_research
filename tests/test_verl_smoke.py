"""Minimal verl integration smoke tests.

GPU/full-stack tests are marked ``@pytest.mark.gpu`` and skipped in CI.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_VERL = importlib.util.find_spec("verl") is not None


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
def test_import_ppo_loss():
    from verl.workers.utils.losses import ppo_loss

    assert callable(ppo_loss)


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
def test_make_cl_loss_no_replay_imports():
    from trainer.cl_loss import make_cl_loss

    loss_fn = make_cl_loss(replay_enabled=False, lambda_replay=0.0)
    assert callable(loss_fn)


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
def test_actor_rollout_worker_has_set_loss_fn():
    from verl.workers.engine_workers import ActorRolloutRefWorker

    assert hasattr(ActorRolloutRefWorker, "set_loss_fn")


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CUDA = False
if HAS_TORCH:
    import torch

    HAS_CUDA = torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_replay_path_differentiable_on_cuda():
    """Single-GPU smoke for the replay row layout + differentiable aggregation.

    Validates on a REAL CUDA device that:
      - build_replay_rows emits the two-mask layout (PPO response_mask == 0,
        replay_response_mask carries the real span) -- bug B8;
      - pad_rows_to_seq_len reconciles differing seq lengths -- bug B9;
      - select_replay_rows backpropagates gradient into the replay rows.
    This does NOT exercise the full RayPPOTrainer (see the cluster smoke below).
    """
    from trainer.replay_forward import (
        IS_REPLAY_KEY,
        REPLAY_MASK_KEY,
        REPLAY_WEIGHTS_KEY,
        build_replay_rows,
        pad_rows_to_seq_len,
        select_replay_rows,
    )

    class _Tok:
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            return list(range(1, 1 + sum(len(m.get("content", "")) for m in messages)))

    samples = [
        ("t1", None, {"messages": [{"role": "assistant", "content": "abcd"}]}),
        ("t2", None, {"messages": [{"role": "assistant", "content": "ef"}]}),
    ]
    rows = build_replay_rows(samples, [[0.4, 0.3, 0.2, 0.1], [0.6, 0.4]], _Tok())
    assert int(rows["response_mask"].sum()) == 0  # B8: PPO ignores replay rows
    rows = pad_rows_to_seq_len(rows, max(rows["responses"].shape[-1], 6))  # B9

    n, t = rows["responses"].shape
    log_probs = torch.randn(n, t, device="cuda", requires_grad=True)
    loss = select_replay_rows(
        log_probs,
        rows[REPLAY_MASK_KEY].cuda(),
        rows[REPLAY_WEIGHTS_KEY].cuda(),
        rows[IS_REPLAY_KEY].cuda(),
    )
    loss.backward()
    assert log_probs.grad.norm().item() > 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
def test_toy_cl_loss_forward_full_stack():
    """Full B1/R4 1-2 step smoke -- requires the multi-GPU training cluster.

    Not runnable on a single GPU / without the real base model weights + dataset.
    Run manually on the cluster, e.g.::

        bash scripts/train.sh configs/run/b1_9b_16gpu.yaml \
            actor_rollout_ref.model.path=/mnt/afs/models/qwen3.6-27b \
            data.train_files=... data.val_files=... \
            trainer.total_training_steps=2 trainer.val_before_train=false

    Verify: backward runs, grad_norm finite, actor/replay_* metrics reported,
    and (R4) buffer.stats() total_size grows after the step.
    """
    pytest.skip("Full-stack 1-2 step smoke — run manually on the training cluster")
