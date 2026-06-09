"""Trajectory storage backend.

Per-trajectory metadata (from ``doc/BucketDesign.md``):
    trajectory_id, bucket, source_task / category, priority,
    insert_step, last_replay_step, replay_count, token_length,
    pattern_id / template_id, forgetting flags

Layout: main store + lightweight indexes
- by id (exact lookup)
- by bucket (bucket scan)
- by priority (top/bottom-k within bucket)
- by pattern (pattern-aware queries)

First-version queries supported: full bucket dump, top/bottom-k priority,
long-unreplayed trajectories, by-pattern / by-task lookup.

Backend choices: in-memory dict (default), SQLite, or LMDB.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any


class TrajectoryStore:
    """Pluggable trajectory storage with metadata indexes.

    The default in-memory backend keeps a flat dict of trajectory_id ->
    (trajectory, metadata), plus secondary indexes for bucket / pattern /
    priority lookups. All indexes are maintained on put/delete so reads
    are O(1) for exact lookup, O(B) for bucket scan, O(B log B) for
    top-k priority (B = bucket size).

    Args:
        backend: 'memory' (default) / 'sqlite' / 'lmdb' (latter two not
                 implemented in v1; raise NotImplementedError).
        path: storage path for persistent backends (ignored for memory).
    """

    def __init__(self, backend: str = "memory", path: str | None = None):
        if backend != "memory":
            raise NotImplementedError(
                f"backend={backend!r} not supported in v1; only 'memory' available."
            )
        self.backend = backend
        self.path = path
        self._store: dict[str, tuple[Any, dict]] = {}
        self._by_bucket: dict[str, set[str]] = defaultdict(set)
        self._by_pattern: dict[str, set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, trajectory_id: str) -> bool:
        return trajectory_id in self._store

    def put(self, trajectory_id: str, trajectory: Any, metadata: dict) -> None:
        """Insert or replace a trajectory and its metadata.

        Required metadata keys: bucket, priority, insert_step.
        Optional: pattern_id, last_replay_step, replay_count, token_length.
        """
        if "bucket" not in metadata or "priority" not in metadata:
            raise ValueError("metadata must include 'bucket' and 'priority'")

        if trajectory_id in self._store:
            old_meta = self._store[trajectory_id][1]
            self._by_bucket[old_meta["bucket"]].discard(trajectory_id)
            if "pattern_id" in old_meta:
                self._by_pattern[old_meta["pattern_id"]].discard(trajectory_id)

        self._store[trajectory_id] = (trajectory, dict(metadata))
        self._by_bucket[metadata["bucket"]].add(trajectory_id)
        if "pattern_id" in metadata:
            self._by_pattern[metadata["pattern_id"]].add(trajectory_id)

    def get(self, trajectory_id: str) -> tuple[Any, dict] | None:
        return self._store.get(trajectory_id)

    def get_metadata(self, trajectory_id: str) -> dict | None:
        entry = self._store.get(trajectory_id)
        return entry[1] if entry else None

    def update_metadata(self, trajectory_id: str, **updates) -> None:
        """In-place metadata update. Used to bump priority / last_replay_step
        without re-inserting the trajectory payload."""
        if trajectory_id not in self._store:
            raise KeyError(trajectory_id)
        traj, meta = self._store[trajectory_id]
        if "bucket" in updates and updates["bucket"] != meta["bucket"]:
            self._by_bucket[meta["bucket"]].discard(trajectory_id)
            self._by_bucket[updates["bucket"]].add(trajectory_id)
        if "pattern_id" in updates and updates.get("pattern_id") != meta.get("pattern_id"):
            if "pattern_id" in meta:
                self._by_pattern[meta["pattern_id"]].discard(trajectory_id)
            self._by_pattern[updates["pattern_id"]].add(trajectory_id)
        meta.update(updates)

    def delete(self, trajectory_id: str) -> None:
        entry = self._store.pop(trajectory_id, None)
        if entry is None:
            return
        meta = entry[1]
        self._by_bucket[meta["bucket"]].discard(trajectory_id)
        if "pattern_id" in meta:
            self._by_pattern[meta["pattern_id"]].discard(trajectory_id)

    def list_by_bucket(self, bucket: str) -> list[str]:
        return list(self._by_bucket.get(bucket, ()))

    def bucket_size(self, bucket: str) -> int:
        return len(self._by_bucket.get(bucket, ()))

    def top_k_priority(self, bucket: str, k: int) -> list[str]:
        """Return up to k trajectory_ids with the HIGHEST priority in bucket."""
        ids = self._by_bucket.get(bucket, ())
        if not ids:
            return []
        scored = [(self._store[tid][1]["priority"], tid) for tid in ids]
        return [tid for _, tid in heapq.nlargest(k, scored)]

    def bottom_k_priority(self, bucket: str, k: int) -> list[str]:
        """Return up to k trajectory_ids with the LOWEST priority in bucket.

        Used by eviction to find victims.
        """
        ids = self._by_bucket.get(bucket, ())
        if not ids:
            return []
        scored = [(self._store[tid][1]["priority"], tid) for tid in ids]
        return [tid for _, tid in heapq.nsmallest(k, scored)]

    def long_unreplayed(self, threshold_steps: int, current_step: int) -> list[str]:
        """Return trajectory_ids whose last_replay_step is older than
        (current_step - threshold_steps), or never replayed."""
        cutoff = current_step - threshold_steps
        out = []
        for tid, (_, meta) in self._store.items():
            last = meta.get("last_replay_step", -1)
            if last < cutoff:
                out.append(tid)
        return out

    def list_by_pattern(self, pattern_id: str) -> list[str]:
        return list(self._by_pattern.get(pattern_id, ()))

    def pattern_counts(self, bucket: str | None = None) -> dict[str, int]:
        """Histogram of pattern_id -> count, optionally restricted to a bucket."""
        if bucket is None:
            return {p: len(ids) for p, ids in self._by_pattern.items()}
        bucket_ids = self._by_bucket.get(bucket, set())
        counts: dict[str, int] = defaultdict(int)
        for tid in bucket_ids:
            meta = self._store[tid][1]
            pid = meta.get("pattern_id")
            if pid is not None:
                counts[pid] += 1
        return dict(counts)

    def all_ids(self) -> list[str]:
        return list(self._store.keys())
