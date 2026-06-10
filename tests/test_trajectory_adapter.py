"""Tests for verl batch -> buffer trajectory conversion."""

from __future__ import annotations

from trainer.trajectory_adapter import extract_trajectories_from_batch, replay_sample_to_metadata


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def test_extract_messages_trajectories():
    batch = FakeBatch(
        {
            "messages": [
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            ],
            "bucket": ["SysOps"],
            "reward": [0.8],
            "task_id": ["task-1"],
        }
    )
    out = extract_trajectories_from_batch(batch)
    assert len(out) == 1
    traj, bucket, meta = out[0]
    assert bucket == "SysOps"
    assert meta["reward"] == 0.8
    assert meta["task_id"] == "task-1"
    assert traj[1]["content"] == "hello"


def test_extract_prompt_response_fallback():
    batch = FakeBatch(
        {
            "prompt": ["question"],
            "response": ["answer"],
            "category": ["Workflow"],
        }
    )
    out = extract_trajectories_from_batch(batch)
    assert len(out) == 1
    traj, bucket, _meta = out[0]
    assert bucket == "Workflow"
    assert traj[0]["role"] == "user"
    assert traj[1]["content"] == "answer"


def test_extract_empty_when_no_fields():
    assert extract_trajectories_from_batch(FakeBatch({})) == []


def test_replay_sample_to_metadata_injects_messages():
    meta = replay_sample_to_metadata(
        ("tid1", [{"role": "assistant", "content": "x"}], {"priority": 1.0})
    )
    assert meta["messages"][0]["role"] == "assistant"
