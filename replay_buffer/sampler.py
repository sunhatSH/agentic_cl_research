"""Replay sampling strategies with pluggable bucket-level selection.

BucketStrategy (Protocol):
    Determines how much weight each bucket gets when sampling replay trajectories.
    Three implementations are provided, swappable by config key:

    - ``uniform`` (UniformStrategy):       equal weight for all buckets.
    - ``quota`` (QuotaStrategy):           proportional to soft_target quota,
                                           mixed with uniform + starvation boost.
                                           (current TwoLevelSampler default)
    - ``distance`` (DistanceStrategy):     weighted by capability-space distance
                                           from the currently-trained bucket.
                                           Farther = higher forgetting risk =
                                           higher replay weight.

Within-bucket sampling remains unchanged (priority-weighted random or uniform).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


def _weighted_choice_without_replacement(
    rng: random.Random,
    items: Sequence,
    weights: Sequence[float],
    k: int,
) -> list:
    if k >= len(items):
        return list(items)
    out = []
    chosen = set()
    items_list = list(items)
    weights_list = list(weights)
    tries = 0
    max_tries = k * 10
    while len(out) < k and tries < max_tries:
        idx = rng.choices(range(len(items_list)), weights=weights_list, k=1)[0]
        if idx not in chosen:
            chosen.add(idx)
            out.append(items_list[idx])
        tries += 1
    if len(out) < k:
        for i, item in enumerate(items_list):
            if i not in chosen:
                out.append(item)
                if len(out) >= k:
                    break
    return out


# ── BucketStrategy Protocol ────────────────────────────────────────────────


class BucketStrategy(Protocol):
    """Pluggable bucket-level selection strategy.

    Given the buffer state and the currently-trained bucket name (or None),
    return a dict mapping each bucket name to a sampling weight.
    """

    def get_weights(
        self,
        buffer,
        *,
        current_bucket: str | None = None,
    ) -> dict[str, float]:
        ...


# ── Built-in strategies ────────────────────────────────────────────────────


class UniformStrategy:
    """Every non-empty bucket gets equal weight."""

    def get_weights(self, buffer, *, current_bucket=None) -> dict[str, float]:
        return {b: 1.0 for b in buffer.bucket_names if buffer.store.bucket_size(b) > 0}


class QuotaStrategy:
    """Mixed quota-proportional + uniform + starvation boost (current default)."""

    def __init__(
        self,
        mix_ratio: float = 0.7,
        starvation_boost: float = 0.1,
        starvation_window: int = 50,
    ):
        self.mix_ratio = mix_ratio
        self.starvation_boost = starvation_boost
        self.starvation_window = starvation_window
        self._last_sample_step: dict[str, int] = {}

    def get_weights(self, buffer, *, current_bucket=None) -> dict[str, float]:
        names = buffer.bucket_names
        k = len(names)
        c = buffer.total_capacity
        cur_step = buffer._step
        weights = {}
        for b in names:
            if buffer.store.bucket_size(b) == 0:
                weights[b] = 0.0
                continue
            quota_w = buffer.soft_target[b] / c if c else 0.0
            uniform_w = 1.0 / k
            w = self.mix_ratio * quota_w + (1.0 - self.mix_ratio) * uniform_w
            last_step = self._last_sample_step.get(b, -1)
            if cur_step - last_step > self.starvation_window:
                w += self.starvation_boost
            weights[b] = w
        return weights

    def mark_sampled(self, bucket: str, step: int) -> None:
        self._last_sample_step[bucket] = step


class DistanceStrategy:
    """Bucket weight proportional to capability-space distance from current_bucket.

    Loads 5-D bucket coordinates from a JSON file (produced by
    ``scripts/score_bucket_coords.py``).  When the currently-trained bucket is
    known, non-current buckets receive weight = d(current, other) / mean(d).
    When current_bucket is None (cold-start / warmup), falls back to uniform.
    """

    def __init__(self, coords_path: str | Path | None = None):
        if coords_path is None:
            coords_path = Path(__file__).resolve().parent.parent / "configs" / "bucket_coords.json"
        with open(coords_path, encoding="utf-8") as f:
            data = json.load(f)
        self._coords: dict[str, list[float]] = data["coordinates"]
        self._dimensions: list[str] = data["dimensions"]

    def get_weights(self, buffer, *, current_bucket=None) -> dict[str, float]:
        names = buffer.bucket_names
        if current_bucket is None or current_bucket not in self._coords:
            return {b: 1.0 for b in names if buffer.store.bucket_size(b) > 0}

        cur = self._coords[current_bucket]
        distances = {}
        for b in names:
            if b == current_bucket or buffer.store.bucket_size(b) == 0:
                distances[b] = 0.0
                continue
            other = self._coords.get(b)
            if other is None:
                distances[b] = 0.0
                continue
            d = math.sqrt(sum((cur[i] - other[i]) ** 2 for i in range(len(cur))))
            distances[b] = d

        mean_d = sum(distances.values()) / max(1, sum(1 for v in distances.values() if v > 0))
        if mean_d == 0:
            return {b: 1.0 for b in names if buffer.store.bucket_size(b) > 0}

        return {b: (d / mean_d) for b, d in distances.items()}


# ── Strategy registry ──────────────────────────────────────────────────────


def make_strategy(name: str, **kwargs) -> BucketStrategy:
    """Build a BucketStrategy by name.

    ``name`` is one of ``uniform``, ``quota``, ``distance``.
    Extra kwargs are forwarded to the strategy constructor.
    """
    if name == "uniform":
        return UniformStrategy()
    elif name == "quota":
        return QuotaStrategy(
            mix_ratio=kwargs.get("mix_ratio", 0.7),
            starvation_boost=kwargs.get("starvation_boost", 0.1),
            starvation_window=kwargs.get("starvation_window", 50),
        )
    elif name == "distance":
        return DistanceStrategy(coords_path=kwargs.get("coords_path"))
    else:
        raise ValueError(f"unknown bucket strategy {name!r}")


# ── TwoLevelSampler (updated) ───────────────────────────────────────────────


class TwoLevelSampler:
    """Two-level sampler: bucket strategy + within-bucket priority.

    Args:
        buffer: BucketReplayBuffer instance.
        bucket_strategy: BucketStrategy instance or name string (e.g. ``"quota"``).
        within_bucket_sampling: ``"priority"`` (default) or ``"uniform"``.
        rng: optional random.Random for reproducibility.
    """

    def __init__(
        self,
        buffer,
        bucket_strategy: BucketStrategy | str = "quota",
        within_bucket_sampling: str = "priority",
        rng: random.Random | None = None,
        **strategy_kwargs,
    ):
        if isinstance(bucket_strategy, str):
            bucket_strategy = make_strategy(bucket_strategy, **strategy_kwargs)
        if within_bucket_sampling not in ("priority", "uniform"):
            raise ValueError(f"unknown within_bucket_sampling {within_bucket_sampling!r}")
        self.buffer = buffer
        self.bucket_strategy = bucket_strategy
        self.within_bucket_sampling = within_bucket_sampling
        self.rng = rng or random.Random()
        self._current_bucket: str | None = None

    def set_current_bucket(self, bucket: str | None) -> None:
        """Tell the sampler which bucket is currently being trained."""
        self._current_bucket = bucket

    def sample(self, batch_size: int):
        if batch_size <= 0:
            return []

        total = len(self.buffer.store)
        target = min(batch_size, total)
        out = []
        chosen: set[str] = set()
        max_tries = target * 20
        tries = 0

        weights = self.bucket_strategy.get_weights(
            self.buffer, current_bucket=self._current_bucket,
        )
        names = list(weights.keys())
        wlist = [weights[b] for b in names]
        wsum = sum(wlist)
        if wsum == 0:
            return []

        while len(out) < target and tries < max_tries:
            tries += 1
            [bucket] = self.rng.choices(names, weights=wlist, k=1)
            tid = self._sample_one_within_bucket(bucket)
            if tid is None or tid in chosen:
                continue
            got = self.buffer.store.get(tid)
            if got is None:
                continue
            chosen.add(tid)
            traj, meta = got
            out.append((tid, traj, meta))
            if isinstance(self.bucket_strategy, QuotaStrategy):
                self.bucket_strategy.mark_sampled(bucket, self.buffer._step)

        # Fallback sweep
        if len(out) < target:
            for b in self.buffer.bucket_names:
                for tid in self.buffer.store.list_by_bucket(b):
                    if tid in chosen:
                        continue
                    got = self.buffer.store.get(tid)
                    if got is None:
                        continue
                    chosen.add(tid)
                    traj, meta = got
                    out.append((tid, traj, meta))
                    if len(out) >= target:
                        break
                if len(out) >= target:
                    break

        self.buffer.mark_replayed([tid for tid, _, _ in out])
        return out

    def _sample_one_within_bucket(self, bucket: str) -> str | None:
        ids = self.buffer.store.list_by_bucket(bucket)
        if not ids:
            return None
        if self.within_bucket_sampling == "uniform":
            return self.rng.choice(ids)
        priorities = [max(self.buffer.store.get_metadata(tid)["priority"], 1e-9) for tid in ids]
        return self.rng.choices(ids, weights=priorities, k=1)[0]


class BaselineSampler:
    """Simple proportional-random sampler — the initial implementation.
    Kept for backward compatibility with existing tests."""

    def __init__(self, buffer, rng: random.Random | None = None):
        self.buffer = buffer
        self.rng = rng or random.Random()

    def sample(self, batch_size: int):
        if batch_size <= 0:
            return []

        total = len(self.buffer.store)
        target = min(batch_size, total)
        out = []
        chosen: set[str] = set()
        names = self.buffer.bucket_names
        weights = [self.buffer.soft_target.get(b, 0) for b in names]

        max_tries = target * 20
        tries = 0
        while len(out) < target and tries < max_tries:
            tries += 1
            [bucket] = self.rng.choices(names, weights=weights, k=1)
            ids = self.buffer.store.list_by_bucket(bucket)
            if not ids:
                continue
            tid = self.rng.choice(ids)
            if tid in chosen:
                continue
            got = self.buffer.store.get(tid)
            if got is None:
                continue
            chosen.add(tid)
            traj, meta = got
            out.append((tid, traj, meta))

        if len(out) < target:
            for b in names:
                for tid in self.buffer.store.list_by_bucket(b):
                    if tid in chosen:
                        continue
                    got = self.buffer.store.get(tid)
                    if got is None:
                        continue
                    chosen.add(tid)
                    traj, meta = got
                    out.append((tid, traj, meta))
                    if len(out) >= target:
                        break
                if len(out) >= target:
                    break

        self.buffer.mark_replayed([tid for tid, _, _ in out])
        return out
