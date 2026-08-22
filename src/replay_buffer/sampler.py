"""Replay sampling strategies.

Two bucket-level samplers remain (others archived 2026-07-23):

- ``DistanceStrategy`` (main): replay weight = capability-space distance from
  the *current batch's* bucket distribution. A batch may mix buckets (multi-turn
  sessions expand to per-turn GRPO groups); the "current position" is the
  batch-weighted centroid of the buckets being trained. Farther buckets carry
  higher forgetting risk -> higher replay weight.

- ``BaselineSampler`` (CLEAR baseline, Rolnick et al. 2019): uniform random
  over the WHOLE pool, no bucket awareness, no current-bucket signal. This is
  the no-CL-strategy control.

Within-bucket sampling is unchanged: ``priority`` (default) or ``uniform``
(R0 / R3 ablation). That is a separate axis from the bucket-level strategy.
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
    """Bucket-level selection: given the buffer + the current batch's bucket
    distribution, return a dict mapping each bucket name to a sampling weight.

    ``current_distribution`` maps bucket name -> share of the current training
    batch (e.g. {"finance": 0.56, "ops": 0.44}); it is None during cold-start /
    warmup when no batch has run yet.
    """

    def get_weights(
        self,
        buffer,
        *,
        current_distribution: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ...


# ── Main strategy: distance from the batch's bucket centroid ───────────────


class DistanceStrategy:
    """Replay weight = capability-space distance from the current batch centroid.

    The current batch may mix buckets (multi-turn sessions expand to per-turn
    GRPO groups, and the tail of one bucket blends into the head of the next).
    The "current position" is the batch-weighted centroid of the buckets being
    trained:

        centroid = Σ_i  share_i * coords[bucket_i]      (share_i = n_i / batch_size)

    Each replay bucket b gets weight = distance(b, centroid) / mean(distance),
    so buckets far from what is being trained (higher forgetting risk) are
    replayed more. Buckets in the current batch sit at/near the centroid and
    get ~0 weight (they are being trained on-policy right now).

    When ``current_distribution`` is None (cold-start / warmup), falls back to
    uniform so every non-empty bucket is eligible.
    """

    def __init__(self, coords_path: str | Path | None = None, metric: str = "euclidean"):
        if coords_path is None:
            coords_path = Path(__file__).resolve().parent.parent.parent / "configs" / "bucket_coords.json"
        with open(coords_path, encoding="utf-8") as f:
            data = json.load(f)
        self._coords: dict[str, list[float]] = data["coordinates"]
        self._metric = metric
        self._dimensions: list[str] = data["dimensions"]

    def _centroid(self, distribution: dict[str, float]) -> list[float] | None:
        """Batch-weighted centroid of the buckets in the current batch."""
        acc = [0.0] * len(self._dimensions)
        total = 0.0
        for bucket, share in distribution.items():
            c = self._coords.get(bucket)
            if c is None or share <= 0:
                continue
            for i in range(len(acc)):
                acc[i] += share * c[i]
            total += share
        if total <= 0:
            return None
        return [v / total for v in acc]

    def get_weights(
        self,
        buffer,
        *,
        current_distribution: dict[str, float] | None = None,
    ) -> dict[str, float]:
        names = buffer.bucket_names
        # Cold-start / warmup: no batch yet -> uniform over non-empty buckets.
        if not current_distribution:
            return {b: 1.0 for b in names if buffer.store.bucket_size(b) > 0}

        centroid = self._centroid(current_distribution)
        if centroid is None:
            return {b: 1.0 for b in names if buffer.store.bucket_size(b) > 0}

        distances = {}
        for b in names:
            if buffer.store.bucket_size(b) == 0:
                distances[b] = 0.0
                continue
            other = self._coords.get(b)
            if other is None:
                distances[b] = 0.0
                continue
            d = math.sqrt(sum((centroid[i] - other[i]) ** 2 for i in range(len(centroid))))
            if self._metric == "manhattan":
                d = sum(abs(centroid[i] - other[i]) for i in range(len(centroid)))
            distances[b] = d

        mean_d = sum(distances.values()) / max(1, sum(1 for v in distances.values() if v > 0))
        if mean_d == 0:
            return {b: 1.0 for b in names if buffer.store.bucket_size(b) > 0}

        return {b: (d / mean_d) for b, d in distances.items()}


# ── CLEAR baseline: uniform random over the whole pool ─────────────────────


class BaselineSampler:
    """CLEAR experience-replay baseline (Rolnick et al. 2019).

    Uniform random sampling over the ENTIRE pool -- no bucket awareness, no
    current-batch signal, no priority. This is the no-CL-strategy control: the
    only CL mechanism active is "keep old trajectories and resample them", with
    no anti-forgetting weighting. Contrast with ``DistanceStrategy`` which adds
    capability-distance-based replay weighting.
    """

    def __init__(self, buffer, rng: random.Random | None = None):
        self.buffer = buffer
        self.rng = rng or random.Random()

    def sample(self, batch_size: int):
        if batch_size <= 0:
            return []

        total = len(self.buffer.store)
        # buffer 不足 batch_size 时【尽可能回放】（采全部可用），不报错 ——
        #   冷启动前几步 buffer 必然不满，报错会让训练起步就崩；这里 min 平滑过渡。
        target = min(batch_size, total)
        out = []
        chosen: set[str] = set()

        # Flat uniform over all trajectory ids, ignoring buckets entirely.
        all_ids: list[str] = []
        for b in self.buffer.bucket_names:
            all_ids.extend(self.buffer.store.list_by_bucket(b))
        if not all_ids:
            return []

        max_tries = target * 20
        tries = 0
        while len(out) < target and tries < max_tries:
            tries += 1
            tid = self.rng.choice(all_ids)
            if tid in chosen:
                continue
            got = self.buffer.store.get(tid)
            if got is None:
                continue
            chosen.add(tid)
            traj, meta = got
            out.append((tid, traj, meta))

        # Fallback sweep if random sampling stalled (small pool / collisions).
        if len(out) < target:
            for tid in all_ids:
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

        self.buffer.mark_replayed([tid for tid, _, _ in out])
        return out


# ── TwoLevelSampler (distance bucket strategy + within-bucket priority) ─────


class TwoLevelSampler:
    """Two-level sampler: distance bucket strategy + within-bucket priority.

    Args:
        buffer: BucketReplayBuffer instance.
        bucket_strategy: a BucketStrategy instance (default DistanceStrategy).
        within_bucket_sampling: ``"priority"`` (default) or ``"uniform"``.
        rng: optional random.Random for reproducibility.
    """

    def __init__(
        self,
        buffer,
        bucket_strategy: BucketStrategy | None = None,
        within_bucket_sampling: str = "priority",
        rng: random.Random | None = None,
    ):
        if bucket_strategy is None:
            bucket_strategy = DistanceStrategy()
        if within_bucket_sampling not in ("priority", "uniform"):
            raise ValueError(f"unknown within_bucket_sampling {within_bucket_sampling!r}")
        self.buffer = buffer
        self.bucket_strategy = bucket_strategy
        self.within_bucket_sampling = within_bucket_sampling
        self.rng = rng or random.Random()
        self._current_distribution: dict[str, float] | None = None

    def set_current_distribution(self, distribution: dict[str, float] | None) -> None:
        """Tell the sampler the current training batch's bucket distribution.

        ``distribution`` maps bucket name -> share (n_i / batch_size). None
        during cold-start / warmup. Drives the distance-based replay weights.
        """
        self._current_distribution = distribution

    def sample(self, batch_size: int):
        if batch_size <= 0:
            return []

        total = len(self.buffer.store)
        # buffer 不足 batch_size 时【尽可能回放】（采全部可用），不报错 ——
        #   冷启动前几步 buffer 必然不满，报错会让训练起步就崩；这里 min 平滑过渡。
        target = min(batch_size, total)
        out = []
        chosen: set[str] = set()
        max_tries = target * 20
        tries = 0

        weights = self.bucket_strategy.get_weights(
            self.buffer,
            current_distribution=self._current_distribution,
        )
        names = list(weights.keys())
        wlist = [weights[b] for b in names]
        # Coerce non-finite bucket weights (NaN/inf from a corrupt coords file or a
        # degenerate distance) to 0 -- else `wsum == 0` is False for NaN and
        # rng.choices raises "Total of weights must be finite" far from the cause.
        import math

        wlist = [w if (isinstance(w, (int, float)) and math.isfinite(w) and w > 0) else 0.0 for w in wlist]
        wsum = sum(wlist)
        if wsum <= 0:
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
        # priority is recomputed on the GPU path (log-prob drift); a NaN/inf there
        # poisons this whole bucket's draw -- max(nan, 1e-9) returns nan and
        # random.choices raises "Total of weights must be finite". Floor to 1e-9
        # AND coerce non-finite -> 1e-9 so a bad priority degrades that trajectory
        # to ~uniform weight instead of crashing the main sampling path.
        import math

        def _safe_prio(tid: str) -> float:
            try:
                p = float(self.buffer.store.get_metadata(tid)["priority"])
            except (TypeError, ValueError, KeyError):
                return 1e-9
            return p if (math.isfinite(p) and p > 1e-9) else 1e-9

        priorities = [_safe_prio(tid) for tid in ids]
        return self.rng.choices(ids, weights=priorities, k=1)[0]
