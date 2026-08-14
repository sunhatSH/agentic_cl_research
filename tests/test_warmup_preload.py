"""Tests for warm-starting the buffer from a sqlite snapshot via cl.buffer.warmup_path.

Exercises trainer.cl_main.build_buffer (CPU-only; no verl / Ray / GPU). Verifies
that a dumped buffer reloaded through the config field has the same trajectory
count and per-bucket distribution, that the experiment's config is preserved
(not overridden by the dump), and that a missing path fails loud.
"""

import pytest
from omegaconf import OmegaConf

from replay_buffer.bucket import BucketReplayBuffer
from trainer.cl_main import build_buffer


def _make_warmup_dump(path) -> dict[str, int]:
    """Dump a small buffer; return its per-bucket distribution."""
    buf = BucketReplayBuffer(total_capacity=14000, bucket_floors=[163,145,131,97,72,72,65,30,53], seed=0)
    buf.set_step(3)
    plan = {"workflow": 4, "ops": 3, "qa": 2}
    for bucket, n in plan.items():
        for i in range(n):
            buf.add_trajectory({"messages": []}, bucket, metadata={"pattern_id": f"{bucket}-{i}"})
    buf.dump(path)
    return plan


def _cfg(warmup_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "cl": {
                "lambda_replay": 1.0,
                "buffer": {
                    "enabled": True,
                    "total_capacity": 14000,
                    "bucket_floors": [163, 145, 131, 97, 72, 72, 65, 30, 53],
                    "warmup_path": warmup_path,
                },
            }
        }
    )


def test_build_buffer_preloads_warmup(tmp_path):
    snap = tmp_path / "warmup.sqlite"
    plan = _make_warmup_dump(snap)

    buf = build_buffer(_cfg(str(snap)))

    assert buf is not None
    assert len(buf.store) == sum(plan.values())
    for bucket, n in plan.items():
        assert buf.store.bucket_size(bucket) == n
    # Counters restored from the dump's buffer_state sidecar.
    assert buf._step == 3
    # Loaded buffer is functional.
    assert len(buf.sample(3)) == 3


def test_warmup_preserves_experiment_capacity(tmp_path):
    # Dump uses capacity 14000; experiment cfg uses a DIFFERENT capacity.
    snap = tmp_path / "warmup.sqlite"
    _make_warmup_dump(snap)

    cfg = _cfg(str(snap))
    cfg.cl.buffer.total_capacity = 7000  # experiment quota, must win over dump
    buf = build_buffer(cfg)

    assert buf.total_capacity == 7000
    assert sum(buf.soft_target.values()) == 7000


def test_no_warmup_path_starts_empty(tmp_path):
    buf = build_buffer(_cfg(None))
    assert buf is not None
    assert len(buf.store) == 0


def test_missing_warmup_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(FileNotFoundError):
        build_buffer(_cfg(str(missing)))
