"""verl RayPPOTrainer construction and CL hook installation.

Wraps ``verl.trainer.main_ppo.TaskRunner`` so buffer + custom loss stay on the
same Ray actor as the trainer driver (buffer is not serialized across Ray).
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def merge_verl_config(cl_cfg: Any) -> Any:
    """Ensure OmegaConf object is compatible with verl RayPPOTrainer.

    verl expects a full Hydra-style config. Experiment yamls inherit
    ``configs/base.yaml``; additional verl fields should be supplied via
    experiment overrides or a verl defaults yaml on the cluster.
    """
    if not OmegaConf.is_config(cl_cfg):
        cl_cfg = OmegaConf.create(cl_cfg)
    OmegaConf.set_struct(cl_cfg, False)
    return cl_cfg


def make_cl_loss_from_cfg(cfg: Any):
    """Build ``make_cl_loss`` closure from OmegaConf ``cfg.cl`` section.

    The buffer is never captured here -- the closure is pickled to the actor
    worker via ``set_loss_fn`` and must stay free of the (driver-only) buffer.
    """
    from trainer.cl_loss import make_cl_loss

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    scheme = (cl.get("weighting", {}) or {}).get("scheme", "W2")
    return make_cl_loss(
        replay_enabled=lambda_replay > 0.0,
        lambda_replay=lambda_replay,
        replay_batch_size=int(cl.get("replay_batch_size", 32)),
        use_token_weighting=scheme != "W0",
        weighting_scheme=scheme,
    )


def inject_cl_loss(trainer: Any, cfg: Any, buffer: Any | None = None) -> None:
    """Call ``actor_rollout_wg.set_loss_fn`` after ``init_workers``.

    ``buffer`` is accepted for signature parity but intentionally unused -- it
    must not travel into the loss closure.
    """
    trainer.actor_rollout_wg.set_loss_fn(make_cl_loss_from_cfg(cfg))


def install_buffer_hooks(trainer: Any, buffer: Any | None, cfg: Any) -> None:
    """Patch RayPPOTrainer to feed trajectories into buffer and pre-stage replay.

    Hooks ``_update_actor`` (pre: attach replay sidecar; post: ingest batch)
    and ``fit`` step counter (sync buffer step).
    """
    if buffer is None:
        return

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    replay_batch_size = int(cl.get("replay_batch_size", 32))
    weighting = None
    if lambda_replay > 0:
        from trainer.cl_loss import build_weighting_from_cfg

        weighting = build_weighting_from_cfg(cl)

    from trainer.replay_batch import prepare_replay_rows
    from trainer.trajectory_adapter import extract_trajectories_from_batch

    tokenizer = getattr(trainer, "tokenizer", None)
    original_update = trainer._update_actor

    def patched_update(batch):
        # 1. Pre: append gradient-carrying replay rows to the actor batch.
        if lambda_replay > 0:
            replay_rows = prepare_replay_rows(
                buffer, weighting, tokenizer, replay_batch_size
            )
            if replay_rows:
                batch = _append_replay_rows(batch, replay_rows)

        result = original_update(batch)

        # 2. Post: ingest the step's new trajectories into the buffer.
        buffer.set_step(getattr(trainer, "global_steps", buffer._step))
        for trajectory, bucket, meta in extract_trajectories_from_batch(batch):
            buffer.add_trajectory(trajectory, bucket, meta)
        return result

    trainer._update_actor = patched_update


def _append_replay_rows(batch: Any, replay_rows: dict[str, Any]) -> Any:
    """Concatenate replay row tensors onto a verl DataProto batch.

    Replay rows carry ``is_replay=True``, zeroed advantages, and per-token
    ``replay_token_weights``. Non-replay rows are back-filled with ``is_replay=False``
    and zero weights so the concatenated batch is rectangular.

    GPU-cluster validation point: the exact DataProto concat helper / padding mode
    depends on the configured engine (see doc/Progress.md verl checklist).
    """
    try:
        import torch
        from verl import DataProto
    except ImportError:
        return batch

    from trainer.replay_forward import IS_REPLAY_KEY, REPLAY_WEIGHTS_KEY

    n_rl = len(batch.batch["responses"]) if "responses" in batch.batch else len(batch.batch)
    seq_len = replay_rows["responses"].shape[-1]

    # Mark existing RL rows as non-replay with zero weights.
    batch.batch[IS_REPLAY_KEY] = torch.zeros(n_rl, dtype=torch.bool)
    batch.batch[REPLAY_WEIGHTS_KEY] = torch.zeros((n_rl, seq_len), dtype=torch.float32)

    replay_dp = DataProto.from_single_dict(replay_rows)
    return DataProto.concat([batch, replay_dp])


class CLTaskRunner:
    """Drop-in replacement for ``verl.trainer.main_ppo.TaskRunner`` with CL hooks.

    Usage from ``cl_main``::

        run_cl_ppo(cfg)  # ray.init + remote CLTaskRunner.run
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    # Delegate worker registration to verl TaskRunner helpers
    def add_actor_rollout_worker(self, config):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.add_actor_rollout_worker(self, config)

    def add_critic_worker(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_critic_worker(self, config)

    def init_resource_pool_mgr(self, config):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.init_resource_pool_mgr(self, config)

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_reward_model_resource_pool(self, config)

    def add_teacher_model_resource_pool(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_teacher_model_resource_pool(self, config)

    def add_ref_policy_worker(self, config, ref_policy_cls):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.add_ref_policy_worker(self, config, ref_policy_cls)

    def run(self, config, resume_from: str | None = None) -> None:
        """Mirror ``TaskRunner.run`` with CL buffer + loss injection."""
        from pprint import pprint

        import socket

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.dataset.rl_dataset import collate_fn

        config = merge_verl_config(config)
        print(f"CLTaskRunner hostname: {socket.gethostname()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from trainer.cl_main import build_buffer

        buffer = build_buffer(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        inject_cl_loss(trainer, config, buffer)
        install_buffer_hooks(trainer, buffer, config)
        if resume_from:
            trainer.load_checkpoint(resume_from)
        trainer.fit()


def run_cl_ppo(cfg: Any, resume_from: str | None = None) -> None:
    """Initialize Ray and run CL training (production entry)."""
    import ray
    from omegaconf import OmegaConf

    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.main_ppo import run_ppo

    cfg = merge_verl_config(cfg)

    if not ray.is_initialized():
        ray_init_kwargs = cfg.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**OmegaConf.to_container(ray_init_kwargs), "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(CLTaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(cfg, resume_from))


def build_trainer(cfg: Any, buffer: Any | None = None):
    """Construct RayPPOTrainer without ``init_workers`` (for tests / local wiring).

    Does not start Ray. Caller must call ``trainer.init_workers()``,
    ``inject_cl_loss()``, ``install_buffer_hooks()``, then ``trainer.fit()``.
    """
    from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.config import validate_config
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.dataset.rl_dataset import collate_fn

    cfg = merge_verl_config(cfg)
    runner = TaskRunner()
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(cfg)
    runner.add_critic_worker(cfg)
    runner.add_reward_model_resource_pool(cfg)
    runner.add_teacher_model_resource_pool(cfg)
    runner.add_ref_policy_worker(cfg, actor_rollout_cls)
    validate_config(
        config=cfg,
        use_reference_policy=need_reference_policy(cfg),
        use_critic=need_critic(cfg),
    )
    local_path = copy_to_local(
        cfg.actor_rollout_ref.model.path,
        use_shm=cfg.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = cfg.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    resource_pool_manager = runner.init_resource_pool_mgr(cfg)
    train_dataset = create_rl_dataset(
        cfg.data.train_files, cfg.data, tokenizer, processor, is_train=True,
        max_samples=cfg.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        cfg.data.val_files, cfg.data, tokenizer, processor, is_train=False,
        max_samples=cfg.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(cfg.data, train_dataset)
    return RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
