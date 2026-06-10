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


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
def test_toy_cl_loss_forward_gpu():
    """Run on GPU cluster manually: pytest -m gpu tests/test_verl_smoke.py -k toy."""
    pytest.skip("GPU 1-2 step smoke — run manually on training cluster")
