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
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


class TrajectoryStore:
    """Pluggable trajectory storage with metadata indexes.

    The default in-memory backend keeps a flat dict of trajectory_id ->
    (trajectory, metadata), plus secondary indexes for bucket / pattern /
    priority lookups. All indexes are maintained on put/delete so reads
    are O(1) for exact lookup, O(B) for bucket scan, O(B log B) for
    top-k priority (B = bucket size).

    Persistence: the LIVE backend is always in-memory (fast indexes). On-disk
    durability is provided by ``save_sqlite`` / ``load_sqlite`` SNAPSHOTS (and
    ``BucketReplayBuffer.dump`` / ``load`` which also persist buffer counters),
    not by a DB-backed live store.

    Args:
        backend: 'memory' (default). 'sqlite' / 'lmdb' live backends are not
                 implemented in v1 (raise NotImplementedError) -- use the
                 SQLite snapshot API instead.
        path: storage path for persistent backends (ignored for memory).
    """

    def __init__(self, backend: str = "memory", path: str | None = None):
        if backend != "memory":
            raise NotImplementedError(f"backend={backend!r} not supported in v1; only 'memory' available.")
        self.backend = backend
        self.path = path
        self._store: dict[str, tuple[Any, dict]] = {}
        self._by_bucket: dict[str, set[str]] = defaultdict(set)
        self._by_pattern: dict[str, set[str]] = defaultdict(set)
        # Monotonic insertion counter for FIFO eviction. insert_step ties within a
        # single training step (many trajectories share one step), so it cannot
        # order FIFO victims deterministically; this strictly-increasing seq can.
        # Stamped into meta["_insert_seq"] on every put of a NEW id.
        self._insert_seq: int = 0

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
            # Preserve the original insertion order on in-place replace so a
            # re-put does not jump the FIFO queue to the back.
            metadata = dict(metadata)
            metadata.setdefault("_insert_seq", old_meta.get("_insert_seq", self._insert_seq))
        else:
            metadata = dict(metadata)
            # Honor a persisted seq (SQLite snapshot reload) so FIFO order
            # survives a resume; otherwise stamp a fresh monotonic seq.
            if "_insert_seq" in metadata:
                self._insert_seq = max(self._insert_seq, int(metadata["_insert_seq"]) + 1)
            else:
                metadata["_insert_seq"] = self._insert_seq
                self._insert_seq += 1

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

    def oldest_k(self, bucket: str, k: int) -> list[str]:
        """Return up to k trajectory_ids inserted EARLIEST into the bucket
        (ascending ``_insert_seq``). Used by FIFO eviction to find victims.
        Falls back to insert_step then id for any legacy entry missing the seq.
        """
        ids = self._by_bucket.get(bucket, ())
        if not ids:
            return []
        scored = [
            ((self._store[tid][1].get("_insert_seq", self._store[tid][1].get("insert_step", 0)), tid), tid)
            for tid in ids
        ]
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

    # ------------------------------------------------------------------ #
    # SQLite snapshot persistence                                         #
    # ------------------------------------------------------------------ #
    # The live store stays in-memory (fast indexes). SQLite is used purely
    # as an on-disk SNAPSHOT format so a long run can resume after a crash
    # (doc/Progress.md "Buffer 持久化"). The trajectory payload + metadata are
    # stored as pickle blobs (arbitrary message-list content); bucket and
    # priority are kept as plain columns for ad-hoc inspection / queries.

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS trajectories ("
        " trajectory_id TEXT PRIMARY KEY,"
        " bucket TEXT,"
        " priority REAL,"
        " traj BLOB,"
        " meta BLOB)"
    )

    def save_sqlite(self, path: str | Path) -> None:
        """Write the entire store to a SQLite snapshot at ``path`` (overwrite)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(self._SCHEMA)
            conn.executemany(
                "INSERT OR REPLACE INTO trajectories VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        tid,
                        meta.get("bucket"),
                        float(meta.get("priority", 0.0)),
                        pickle.dumps(traj, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    for tid, (traj, meta) in self._store.items()
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def load_sqlite(self, path: str | Path) -> None:
        """Replace the store contents with a SQLite snapshot from ``path``.

        Rebuilds all in-memory indexes (bucket / pattern) from the loaded rows.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("SELECT trajectory_id, traj, meta FROM trajectories").fetchall()
        finally:
            conn.close()
        self._store.clear()
        self._by_bucket.clear()
        self._by_pattern.clear()
        for tid, traj_blob, meta_blob in rows:
            traj = pickle.loads(traj_blob)
            meta = pickle.loads(meta_blob)
            self.put(tid, traj, meta)
