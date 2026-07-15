"""Training entry point.

Constructs the BucketReplayBuffer (only when needed), wires the cl_loss into
verl's RayPPOTrainer, and starts training. We do NOT fork verl -- the loss
is injected via ``actor.set_loss_fn`` (see ``doc/VerlIntegration.md`` §3.1).

Zero-coefficient short-circuit (mandatory project rule):
- When cl.buffer.enabled = false OR cl.lambda_replay = 0, the buffer is NOT
  instantiated at all -- saves ~25k * trajectory memory and skips all
  buffer hooks. The cl_loss closure also picks a no-replay branch.

Usage:
    python -m trainer.cl_main --config configs/phase1/b1.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint to resume.")
    return parser.parse_args()


def load_config(config_path: str):
    """Resolve OmegaConf inheritance (defaults: [../base])."""
    cfg_path = Path(config_path).resolve()
    cfg = OmegaConf.load(cfg_path)

    defaults = cfg.pop("defaults", None)
    if defaults:
        merged = OmegaConf.create({})
        for entry in defaults:
            ref = str(entry).strip()
            if ref.startswith("../"):
                base_path = (cfg_path.parent / f"{ref}.yaml").resolve()
            elif ref.startswith("/"):
                base_path = Path(ref)
            else:
                base_path = (cfg_path.parent / f"{ref}.yaml").resolve()
            merged = OmegaConf.merge(merged, OmegaConf.load(base_path))
        cfg = OmegaConf.merge(merged, cfg)

    # runs/ layout: if CKPT_DIR is set (start_train.sh), write real checkpoints
    # there (runs-external ckpts/<exp>/, symlinked from runs/<phase>/<exp>/checkpoints).
    # See runs/README.md + doc/eval/训练与评测总思路_产物结构.md.
    ckpt_dir = os.environ.get("CKPT_DIR")
    if ckpt_dir:
        OmegaConf.update(cfg, "trainer.default_local_dir", ckpt_dir, force_add=True)
    return cfg


def build_buffer(cfg):
    """Instantiate BucketReplayBuffer from cfg, or None if disabled.

    Short-circuits when:
      - cl.buffer.enabled is false, or
      - cl.lambda_replay == 0 (no replay term -> no need for buffer).
    """
    cl = cfg.get("cl", {})
    if not cl:
        return None
    if not cl.get("buffer", {}).get("enabled", False):
        return None
    if float(cl.get("lambda_replay", 0.0)) == 0.0:
        return None

    from replay_buffer.bucket import BucketReplayBuffer
    from replay_buffer.priority import Priority, RewardPriority, UniformPriority

    bcfg = cl["buffer"]
    priority_type = bcfg.get("priority_type", "anti_forgetting")
    if priority_type == "reward":
        priority = RewardPriority()
    elif priority_type == "uniform":
        priority = UniformPriority()
    else:
        priority = Priority()

    buffer = BucketReplayBuffer(
        num_buckets=int(bcfg.get("num_buckets", 9)),
        total_capacity=int(bcfg.get("total_capacity", 25000)),
        bucket_names=list(bcfg.get("bucket_names", [])) or None,
        bucket_task_counts=list(bcfg.get("bucket_task_counts", [])) or None,
        bucket_floors=list(bcfg.get("bucket_floors", [])) or None,
        alpha=float(bcfg.get("alpha", 0.5)),
        priority=priority,
        eviction_type=bcfg.get("eviction_type", "priority"),
        within_bucket_sampling=bcfg.get("within_bucket_sampling", "priority"),
        bucket_strategy=bcfg.get("bucket_strategy", "quota"),
        seed=bcfg.get("seed", None),
    )

    _preload_warmup(buffer, bcfg.get("warmup_path", None))
    return buffer


def _preload_warmup(buffer, warmup_path) -> None:
    """Warm-start the buffer from a sqlite snapshot (CL cold-start).

    ``BucketReplayBuffer.load`` restores trajectories + counters only; the
    buffer's config (capacity / bucket_names / soft_target) stays as constructed
    here, so a warmup dumped with a different capacity does not override the
    experiment's quota. A non-empty ``warmup_path`` that does not exist is a hard
    error -- a mis-set path should fail loud, not silently start empty.
    """
    if not warmup_path:
        return
    path = Path(str(warmup_path))
    if not path.exists():
        raise FileNotFoundError(
            f"cl.buffer.warmup_path={path} does not exist; "
            "dump one with scripts/warmup_buffer.py or set it to null."
        )
    buffer.load(path)
    stats = buffer.stats()
    dist = {b: v["size"] for b, v in stats["per_bucket"].items()}
    print(
        f"[cl] warm-started buffer from {path}: " f"{stats['total_size']} trajectories, per-bucket={dist}",
        flush=True,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    from trainer.verl_runner import run_cl_ppo

    run_cl_ppo(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
