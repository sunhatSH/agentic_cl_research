"""Training entry point.

Constructs the BucketReplayBuffer, wires the cl_loss into verl's RayPPOTrainer,
and starts training. We do NOT fork verl -- the loss is injected via
``actor.set_loss_fn`` (see ``doc/VerlIntegration.md`` section 3.1).

Usage:
    python -m trainer.cl_main --config configs/b1.yaml
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint to resume.")
    return parser.parse_args()


def build_buffer(cfg):
    """Instantiate BucketReplayBuffer from cfg."""
    raise NotImplementedError


def build_trainer(cfg, buffer):
    """Construct RayPPOTrainer and inject cl_loss via actor.set_loss_fn."""
    raise NotImplementedError


def install_buffer_hooks(trainer, buffer):
    """Wrap update_actor / generate_sequences to feed new trajectories into buffer.

    Two implementation choices:
    - on_step_end callback registered with the dataloader sampler.
    - monkey-patch around ``trainer.actor_rollout_wg.update_actor``.

    See ``doc/VerlIntegration.md`` section 3.3.
    """
    raise NotImplementedError


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
