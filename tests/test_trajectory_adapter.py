"""Tests for verl batch -> buffer trajectory conversion."""

from __future__ import annotations

from trainer.trajectory_adapter import extract_trajectories_from_batch, replay_sample_to_metadata


class FakeBatch:
    def __init__(self, non_tensor_batch, batch=None):
        self.non_tensor_batch = non_tensor_batch
        if batch is not None:
            self.batch = batch


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
    meta = replay_sample_to_metadata(("tid1", [{"role": "assistant", "content": "x"}], {"priority": 1.0}))
    assert meta["messages"][0]["role"] == "assistant"


def test_backfills_priority_signals():
    # Bug D2: original_logprobs (from rollout) + success_rate must be backfilled
    # so forgetting_risk / within_bucket_difficulty are not stuck at 0.
    batch = FakeBatch(
        {
            "messages": [[{"role": "assistant", "content": "a"}]],
            "bucket": ["SysOps"],
            "task_id": ["t1"],
            "success_rate": [0.5],
        },
        batch={"rollout_log_probs": [[-1.0, -1.2, -0.8]]},
    )
    _traj, _bucket, meta = extract_trajectories_from_batch(batch)[0]
    assert meta["original_logprobs"] == [-1.0, -1.2, -0.8]
    assert meta["success_rate"] == 0.5


def test_signals_feed_nonzero_priority():
    # End-to-end: a backfilled trajectory yields priority driven by >1 signal.
    from replay_buffer.priority import Priority

    p = Priority()  # default: forgetting 0.5, rarity 0.25, diversity 0 (disabled), difficulty 0.25
    traj = {
        "original_logprobs": [-1.0, -1.0],
        "current_logprobs": [-1.5, -1.5],  # drift 0.5 -> forgetting_risk 0.5
        "pattern_id": "p",
        "success_rate": 0.4,  # difficulty 0.6
    }
    score = p.compute(traj, {"pattern_counts": {"p": 0}})  # rarity 1.0
    # 0.5*0.5 + 0.25*1.0 + 0 + 0.25*0.6 = 0.25 + 0.25 + 0.15 = 0.65
    assert abs(score - 0.65) < 1e-9
