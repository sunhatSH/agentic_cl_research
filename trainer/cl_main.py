"""Training entry point.

Constructs the BucketReplayBuffer (only when needed), wires the cl_loss into
verl's RayPPOTrainer, and starts training. We do NOT fork verl -- the loss
is injected via ``actor.set_loss_fn`` (see ``doc/VerlIntegration.md`` §3.1).

Zero-coefficient short-circuit (mandatory project rule):
- When cl.buffer.enabled = false OR cl.lambda_replay = 0, the buffer is NOT
  instantiated at all -- saves ~25k * trajectory memory and skips all
  buffer hooks. The cl_loss closure also picks a no-replay branch.
- See project memory `short_circuit_zero_coefficient`.

Usage:
    python -m trainer.cl_main --config configs/phase1/b1.yaml
"""

from __future__ import annotations

import argparse
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
        # Lambda is zero -> buffer would never be sampled; skip allocation.
        return None

    from replay_buffer.bucket import BucketReplayBuffer
    from replay_buffer.priority import Priority, RewardPriority

    bcfg = cl["buffer"]
    priority_type = bcfg.get("priority_type", "anti_forgetting")
    priority = RewardPriority() if priority_type == "reward" else Priority()

    return BucketReplayBuffer(
        num_buckets=int(bcfg.get("num_buckets", 7)),
        total_capacity=int(bcfg.get("total_capacity", 25000)),
        q_min=int(bcfg.get("q_min", 2000)),
        bucket_names=list(bcfg.get("bucket_names", [])) or None,
        bucket_task_counts=list(bcfg.get("bucket_task_counts", [])) or None,
        alpha=float(bcfg.get("alpha", 0.5)),
        priority=priority,
    )


def build_trainer(cfg, buffer):
    """Construct RayPPOTrainer and inject cl_loss via actor.set_loss_fn.

    Implementation deferred until verl version is pinned (see VerlIntegration.md §4.2).
    """
    raise NotImplementedError(
        "Trainer construction pending verl integration smoke test (see VerlIntegration.md §4.2)."
    )


def install_buffer_hooks(trainer, buffer):
    """Wrap update_actor / generate_sequences to feed new trajectories into buffer.

    Two implementation choices (see ``doc/VerlIntegration.md`` §3.3):
    - on_step_end callback registered with the dataloader sampler.
    - monkey-patch around ``trainer.actor_rollout_wg.update_actor``.

    No-op when buffer is None.
    """
    if buffer is None:
        return
    raise NotImplementedError("Buffer hook installation pending verl integration.")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    buffer = build_buffer(cfg)
    trainer = build_trainer(cfg, buffer)
    install_buffer_hooks(trainer, buffer)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)  # noqa: F841 -- verl-specific
    trainer.fit()


if __name__ == "__main__":
    main()
