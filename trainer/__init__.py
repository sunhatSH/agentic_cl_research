"""Training entry and CL loss for the verl-based trainer.

This package wires together:
- a custom ``cl_loss`` injected into verl via ``actor.set_loss_fn``,
- the ``BucketReplayBuffer`` as a separate Python module,
- a thin ``cl_main`` entry that constructs ``RayPPOTrainer`` and starts training.

We do NOT fork verl. See ``doc/VerlIntegration.md`` for the integration plan.
"""
