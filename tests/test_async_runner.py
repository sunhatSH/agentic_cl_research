"""Import-level + pure-logic tests for the fully-async CL runner scaffold."""

from __future__ import annotations

from trainer.verl_async_runner import cl_actor_update


def test_module_imports_without_verl():
    # Importing the module must not require verl (deferred import in factory).
    import trainer.verl_async_runner as m

    assert hasattr(m, "make_cl_fully_async_trainer_cls")
    assert callable(m.cl_actor_update)


class _FakeBuffer:
    def __init__(self):
        self.step = None
        self.added = []

    def set_step(self, s):
        self.step = s

    def add_trajectory(self, traj, bucket, meta):
        self.added.append((traj, bucket, meta))


def test_cl_actor_update_appends_and_ingests():
    buf = _FakeBuffer()
    calls = {}

    def prepare(buffer, weighting, tok, bs, warmup_size):
        calls["prepared"] = True
        return {"is_replay": True}

    def append(batch, rows):
        calls["appended"] = rows
        return batch + ["+replay"]

    def extract(batch):
        return [("traj", "Workflow", {"pattern_id": "p"})]

    def original_update(batch):
        calls["update_batch"] = batch
        return "updated"

    out = cl_actor_update(
        ["rl"],
        original_update,
        buffer=buf,
        lambda_replay=0.5,
        prepare_replay_rows=prepare,
        append_replay_rows=append,
        extract_trajectories=extract,
        weighting=object(),
        tokenizer=object(),
        replay_batch_size=8,
        replay_warmup_size=0,
        current_step=3,
    )
    assert out == "updated"
    assert calls["prepared"] is True
    assert calls["update_batch"] == ["rl", "+replay"]  # replay appended pre-update
    assert buf.step == 3
    assert buf.added == [("traj", "Workflow", {"pattern_id": "p"})]


def test_cl_actor_update_skips_replay_when_lambda_zero():
    buf = _FakeBuffer()

    def prepare(*a, **k):
        raise AssertionError("prepare_replay_rows must not be called when lambda_replay=0")

    out = cl_actor_update(
        ["rl"],
        lambda b: "updated",
        buffer=buf,
        lambda_replay=0.0,
        prepare_replay_rows=prepare,
        append_replay_rows=lambda b, r: b,
        extract_trajectories=lambda b: [],
        weighting=None,
        tokenizer=None,
        replay_batch_size=8,
        replay_warmup_size=0,
        current_step=1,
    )
    assert out == "updated"
    assert buf.step == 1  # still advances buffer step (ingestion path runs)
