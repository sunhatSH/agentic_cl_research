"""Driver-level integration smoke for install_buffer_hooks.

Runs the FULL patched ``_update_actor`` orchestration end-to-end against a fake
trainer + real verl DataProto, WITHOUT model weights / GPU:

    replay sampling -> _append_replay_rows (DataProto.concat) -> (fake) update
    -> forgetting backfill (graceful no-op, no worker group)
    -> buffer ingest of the RL rows ONLY -> stats JSONL + wandb metrics
    -> periodic buffer.dump snapshot.

This is the most complete smoke runnable on a single dev box. The REAL actor
forward (27B weights, multi-GPU) and the full RL-batch tensor schema are still
validated on the cluster (see doc/Migration_64GPU.md §4).
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_VERL = importlib.util.find_spec("verl") is not None
pytestmark = pytest.mark.skipif(not HAS_VERL, reason="verl not installed")


class _Tok:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return list(range(1, 1 + sum(len(m.get("content", "")) for m in messages)))


def _make_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "trainer": {"save_freq": 2, "experiment_name": "smoke"},
            "cl": {
                "lambda_replay": 0.5,
                "replay_batch_size": 2,
                "replay_warmup_size": 0,
                "buffer_stats_log_freq": 1,
                "forgetting_update_freq": 1,
                "buffer": {
                    "enabled": True,
                    "num_buckets": 7,
                    "total_capacity": 1000,
                    "q_min": 50,
                    "priority_type": "anti_forgetting",
                },
                "weighting": {"scheme": "W2", "gamma": 0.88, "delta": 0.88},
            },
        }
    )


def _rl_batch(n: int):
    """A minimal RL DataProto: tensor keys mirror the replay-row layout so
    DataProto.concat aligns, plus messages/bucket non_tensor for ingest."""
    import numpy as np
    import torch
    from verl import DataProto

    from trainer.replay_forward import (
        IS_REPLAY_KEY,
        REPLAY_MASK_KEY,
        REPLAY_WEIGHTS_KEY,
    )

    t = 4
    tensors = {
        "input_ids": torch.ones(n, t, dtype=torch.long),
        "attention_mask": torch.ones(n, t, dtype=torch.long),
        "position_ids": torch.arange(t).repeat(n, 1),
        "responses": torch.ones(n, t, dtype=torch.long),
        "response_mask": torch.ones(n, t, dtype=torch.long),
        "old_log_probs": torch.zeros(n, t),
        "ref_log_prob": torch.zeros(n, t),
        "advantages": torch.zeros(n, t),
        REPLAY_WEIGHTS_KEY: torch.zeros(n, t),
        REPLAY_MASK_KEY: torch.zeros(n, t, dtype=torch.long),
        IS_REPLAY_KEY: torch.zeros(n, dtype=torch.bool),
    }
    messages = np.empty(n, dtype=object)
    for i in range(n):
        messages[i] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans"},
        ]
    buckets = np.array((["Workflow", "SysOps"] * n)[:n], dtype=object)
    return DataProto.from_dict(tensors=tensors, non_tensors={"messages": messages, "bucket": buckets})


class _FakeTrainer:
    """Captures the batch handed to the (real) update and returns verl-style metrics."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.global_steps = 0
        self.last_batch = None

    def _update_actor(self, batch):
        from verl import DataProto

        self.last_batch = batch
        return DataProto.from_single_dict({}, meta_info={"metrics": {}})


def test_buffer_hooks_full_orchestration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from trainer.cl_main import build_buffer
    from trainer.verl_runner import install_buffer_hooks

    cfg = _make_cfg()
    buffer = build_buffer(cfg)
    assert buffer is not None

    # Seed a few replayable trajectories (messages + original_logprobs).
    for i in range(4):
        buffer.add_trajectory(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a" * (i + 2)}],
            "Workflow",
            metadata={
                "trajectory_id": f"seed-{i}",
                "pattern_id": f"p{i}",
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a" * (i + 2)},
                ],
                "response_token_ids": list(range(1, 7)),
                "original_logprobs": [-0.2, -0.2, -0.2],
            },
        )
    seed_total = len(buffer.store)

    trainer = _FakeTrainer(_Tok())
    install_buffer_hooks(trainer, buffer, cfg)

    n_rl = 2
    for step in (1, 2, 3):
        trainer.global_steps = step
        trainer._update_actor(_rl_batch(n_rl))
        combined = trainer.last_batch
        # Replay rows were appended (RL rows + replay rows).
        assert int(combined.batch["is_replay"].sum()) > 0
        assert len(combined) > n_rl

    # Ingested ONLY the RL rows each step (replay rows excluded): +2 per step.
    assert len(buffer.store) == seed_total + n_rl * 3

    # Buffer-dynamics evidence: one JSONL line per step + wandb metrics merged.
    stats_file = tmp_path / "logs" / "buffer_stats" / "smoke.jsonl"
    assert stats_file.exists()
    assert len(stats_file.read_text().strip().splitlines()) == 3

    # Periodic snapshot at save_freq=2 -> step 2 dumped (steps 1,3 not).
    assert (tmp_path / "buffer_dumps" / "smoke-step-2.sqlite").exists()
    assert not (tmp_path / "buffer_dumps" / "smoke-step-1.sqlite").exists()


def test_metrics_merged_into_update_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from trainer.cl_main import build_buffer
    from trainer.verl_runner import install_buffer_hooks

    cfg = _make_cfg()
    buffer = build_buffer(cfg)
    buffer.add_trajectory(
        [{"role": "assistant", "content": "abc"}],
        "Workflow",
        metadata={
            "trajectory_id": "s0",
            "pattern_id": "p0",
            "messages": [{"role": "assistant", "content": "abc"}],
            "response_token_ids": list(range(1, 7)),
        },
    )
    trainer = _FakeTrainer(_Tok())
    install_buffer_hooks(trainer, buffer, cfg)

    trainer.global_steps = 1
    result = trainer._update_actor(_rl_batch(2))
    metrics = result.meta_info["metrics"]
    assert "buffer/total_size" in metrics
    assert "buffer/fill_ratio_mean" in metrics
