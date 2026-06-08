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


class TrajectoryStore:
    """Pluggable trajectory storage with metadata indexes.

    Args:
        backend: 'memory' / 'sqlite' / 'lmdb'.
        path: storage path for persistent backends.
    """

    def __init__(self, backend: str = "memory", path: str = None):
        raise NotImplementedError

    def put(self, trajectory_id: str, trajectory, metadata: dict) -> None:
        raise NotImplementedError

    def get(self, trajectory_id: str):
        raise NotImplementedError

    def delete(self, trajectory_id: str) -> None:
        raise NotImplementedError

    def list_by_bucket(self, bucket: str):
        raise NotImplementedError

    def top_k_priority(self, bucket: str, k: int):
        raise NotImplementedError

    def long_unreplayed(self, threshold_steps: int):
        raise NotImplementedError
