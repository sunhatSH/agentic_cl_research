"""verl RayPPOTrainer construction and CL hook installation.

Wraps ``verl.trainer.main_ppo.TaskRunner`` so buffer + custom loss stay on the
same Ray actor as the trainer driver (buffer is not serialized across Ray).
"""

from __future__ import annotations

from pathlib import Path
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

    Passes the ``actor_rollout_ref.actor`` subtree so the closure can rebuild the
    ``ActorConfig`` that verl's ``ppo_loss`` needs (bug-1: verl calls the loss
    without a ``config`` arg). This subtree is plain OmegaConf -> pickle-safe.
    """
    from trainer.cl_loss import make_cl_loss

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    scheme = (cl.get("weighting", {}) or {}).get("scheme", "W2")
    actor_cfg = cfg.get("actor_rollout_ref", {}).get("actor", None)
    return make_cl_loss(
        replay_enabled=lambda_replay > 0.0,
        lambda_replay=lambda_replay,
        replay_batch_size=int(cl.get("replay_batch_size", 512)),
        use_token_weighting=scheme != "W0",
        weighting_scheme=scheme,
        actor_cfg=actor_cfg,
    )


def inject_cl_loss(trainer: Any, cfg: Any, buffer: Any | None = None) -> None:
    """Call ``actor_rollout_wg.set_loss_fn`` after ``init_workers``.

    ``buffer`` is accepted for signature parity but intentionally unused -- it
    must not travel into the loss closure.
    """
    trainer.actor_rollout_wg.set_loss_fn(make_cl_loss_from_cfg(cfg))


def _persist_winners(winners: list, exp_name: str, step: int) -> None:
    """Write winner trajectories as JSONL for offline analysis."""
    if not winners:
        return
    import json
    from pathlib import Path

    out_dir = Path(f"rollouts/training/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"step-{step}.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for traj, bucket, meta in winners:
            tid = meta.get("task_id", "")
            row = {
                "task_id": tid,
                "bucket": bucket,
                "reward": meta.get("reward"),
                "messages": traj.get("messages", []),
                "step": step,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[persist] {len(winners)} winners → {out_file}", flush=True)


def install_buffer_hooks(trainer: Any, buffer: Any | None, cfg: Any) -> None:
    """Patch RayPPOTrainer to feed trajectories into buffer and pre-stage replay.

    Hooks ``_update_actor`` (pre: attach replay sidecar; post: ingest batch)
    and ``fit`` step counter (sync buffer step).
    """
    if buffer is None:
        return

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    replay_batch_size = int(cl.get("replay_batch_size", 512))
    replay_warmup_size = int(cl.get("replay_warmup_size", 0))
    # Paper-evidence cadences (0 disables). See doc/Progress.md / replay_metrics.
    stats_log_freq = int(cl.get("buffer_stats_log_freq", 1))
    forgetting_update_freq = int(cl.get("forgetting_update_freq", 1))
    trainer_cfg = cfg.get("trainer", {}) or {}
    save_freq = int(trainer_cfg.get("save_freq", 0))
    exp_name = trainer_cfg.get("experiment_name", "cl")
    weighting = None
    if lambda_replay > 0:
        from trainer.cl_loss import build_weighting_from_cfg

        weighting = build_weighting_from_cfg(cl)

    from trainer.replay_batch import prepare_replay_rows
    from trainer.replay_forward import REPLAY_TIDS_KEY
    from trainer.replay_metrics import (
        BufferStatsLogger,
        backfill_forgetting,
        compute_replay_current_logprobs,
        flatten_buffer_stats,
    )
    from trainer.trajectory_adapter import extract_trajectories_from_batch

    stats_logger = BufferStatsLogger(f"logs/buffer_stats/{exp_name}.jsonl")
    tokenizer = getattr(trainer, "tokenizer", None)
    original_update = trainer._update_actor

    def _merge_buffer_metrics(result, stats) -> None:
        """Surface flattened buffer stats into verl's logged metrics (wandb)."""
        meta = getattr(result, "meta_info", None)
        if isinstance(meta, dict) and isinstance(meta.get("metrics"), dict):
            meta["metrics"].update(flatten_buffer_stats(stats))

    def patched_update(batch):
        # Keep a handle to the ORIGINAL rollout batch: the post-append batch
        # also contains replay rows (is_replay=True) which must NOT be ingested
        # back into the buffer as if they were fresh trajectories.
        rl_batch = batch
        # 1. Pre: append gradient-carrying replay rows to the actor batch.
        replay_rows: dict = {}
        if lambda_replay > 0:
            replay_rows = prepare_replay_rows(
                buffer,
                weighting,
                tokenizer,
                replay_batch_size,
                warmup_size=replay_warmup_size,
            )
            if replay_rows:
                shuffle_seed = int(getattr(trainer, "global_steps", 0) or 0)
                batch = _append_replay_rows(batch, replay_rows, shuffle_seed=shuffle_seed)

        result = original_update(batch)

        step = getattr(trainer, "global_steps", buffer._step)
        buffer.set_step(step)

        # 2. Post: activate forgetting_risk -- recompute current-policy log-probs
        #    for the just-replayed trajectories and backfill their priority.
        tids = replay_rows.get(REPLAY_TIDS_KEY) if replay_rows else None
        if tids and forgetting_update_freq > 0 and step % forgetting_update_freq == 0:
            means = compute_replay_current_logprobs(trainer, replay_rows)
            if means is not None:
                backfill_forgetting(buffer, tids, means)

        # 3. Post: GRPO produces 8 trajectories per query for advantage
        #    computation, but only WINNERS (max reward per task_id) go into the
        #    buffer for replay — losers would dilute anti-forgetting quality.
        from data.cleaning import strip_zw

        # Group by task_id to pick winners
        groups: dict[str, list[tuple[Any, str, dict]]] = {}
        for trajectory, bucket, meta in extract_trajectories_from_batch(
            rl_batch, valid_buckets=getattr(buffer, "bucket_names", None)
        ):
            for msg in trajectory.get("messages", []):
                if isinstance(msg.get("content"), str):
                    msg["content"] = strip_zw(msg["content"])
            tid = meta.get("task_id") or ""
            groups.setdefault(tid, []).append((trajectory, bucket, meta))

        for tid, candidates in groups.items():
            best = max(candidates, key=lambda x: float(x[2].get("reward", 0) or 0))
            buffer.add_trajectory(*best)

        # 3b. Persist winners to JSONL (offline analysis / reproducibility).
        all_rows = [best for best in groups.values()]
        _persist_winners(all_rows, exp_name, step)

        # 4. Post: buffer-dynamics evidence (wandb metrics + sidecar JSONL).
        if stats_log_freq > 0 and step % stats_log_freq == 0:
            stats = buffer.stats()
            _merge_buffer_metrics(result, stats)
            stats_logger.log(step, stats)

        # 5. Post: periodic full buffer snapshot for reproducibility / analysis.
        if save_freq > 0 and step > 0 and step % save_freq == 0:
            snap = Path(f"buffer_dumps/{exp_name}-step-{step}.sqlite")
            snap.parent.mkdir(parents=True, exist_ok=True)
            buffer.dump(snap)

        return result

    trainer._update_actor = patched_update


def _append_replay_rows(batch: Any, replay_rows: dict[str, Any], shuffle_seed: int = 0) -> Any:
    """Concatenate replay row tensors onto a verl DataProto batch.

    Replay rows carry ``is_replay=True``, a real ``replay_response_mask`` (with
    ``response_mask`` = 0 so ppo_loss ignores them, bug B8), zeroed advantages,
    and per-token ``replay_token_weights``. RL rows are back-filled with the
    mirror-image zero fields so the two row sets are rectangular and never
    contaminate each other.

    Alignment (bug B9 / bug-2b): verl rollout rows are LEFT-padded prompt + RIGHT
    -padded response. Replay rows from ``build_replay_rows`` share that layout. We
    align the two sets on BOTH dims independently -- prompts left-padded to max P,
    responses right-padded to max R -- then rebuild ``input_ids`` / ``attention_mask``
    / ``position_ids`` so ``no_padding_2_padding`` slices the response correctly.
    The exact engine concat path is validated on the GPU cluster (doc/Progress.md).
    """
    try:
        import torch
        import torch.nn.functional as F
        from verl import DataProto
    except ImportError:
        return batch

    from trainer.replay_forward import (
        IS_REPLAY_KEY,
        REPLAY_MASK_KEY,
        REPLAY_TIDS_KEY,
        REPLAY_WEIGHTS_KEY,
    )

    # Non-tensor sidecar (trajectory ids) never enters the TensorDict.
    replay_rows = {k: v for k, v in replay_rows.items() if k != REPLAY_TIDS_KEY}

    bb = batch.batch
    if "prompts" not in bb or "responses" not in bb:
        # Unexpected layout (no prompt/response split) -> skip replay append
        # rather than corrupt the batch; logged upstream by metrics.
        return batch

    n_rl = len(bb["responses"])
    P_rl, R_rl = bb["prompts"].shape[-1], bb["responses"].shape[-1]
    P_re, R_re = replay_rows["prompts"].shape[-1], replay_rows["responses"].shape[-1]
    P, R = max(P_rl, P_re), max(R_rl, R_re)

    # Response-width fields (right-pad), prompt-width fields (left-pad).
    resp_width_keys = {
        "responses",
        "response_mask",
        REPLAY_MASK_KEY,
        REPLAY_WEIGHTS_KEY,
        "old_log_probs",
        "ref_log_prob",
        "advantages",
    }

    def _align(rows: dict[str, Any]) -> dict[str, Any]:
        out = dict(rows)
        # left-pad prompts
        if out["prompts"].shape[-1] < P:
            out["prompts"] = F.pad(out["prompts"], (P - out["prompts"].shape[-1], 0), value=0)
        # right-pad response-width fields
        for k in resp_width_keys:
            v = out.get(k)
            if v is not None and hasattr(v, "dim") and v.dim() == 2 and v.shape[-1] < R:
                out[k] = F.pad(v, (0, R - v.shape[-1]), value=0)
        # rebuild full-sequence fields from aligned prompt + response segments
        prompt_attn = (out["prompts"] != 0).long()
        # attention over the response segment = response tokens that are real
        # (pad id 0 for placeholder positions in both RL and replay rows).
        resp_real = (out["responses"] != 0).long()
        out["input_ids"] = torch.cat([out["prompts"], out["responses"]], dim=1)
        out["attention_mask"] = torch.cat([prompt_attn, resp_real], dim=1)
        out["position_ids"] = (out["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
        return out

    # Back-fill RL rows with mirror-image replay fields (response-width).
    bb[IS_REPLAY_KEY] = torch.zeros(n_rl, dtype=torch.bool)
    bb[REPLAY_WEIGHTS_KEY] = torch.zeros((n_rl, R_rl), dtype=torch.float32)
    bb[REPLAY_MASK_KEY] = torch.zeros((n_rl, R_rl), dtype=torch.long)

    # Align RL batch tensors in place (prompts left, responses right).
    rl_aligned = _align({k: bb[k] for k in bb.keys()})
    for k, v in rl_aligned.items():
        bb[k] = v

    replay_rows = _align(replay_rows)
    replay_dp = DataProto.from_single_dict(replay_rows)

    # DataProto.concat requires both sides to share non_tensor_batch keys with a
    # length == batch size. Replay rows carry no rollout non-tensor columns
    # (messages/bucket/...), so back-fill placeholder columns of length n_replay.
    # These placeholders are never read: replay rows are excluded from buffer
    # ingest (patched_update uses the pre-append rl_batch).
    rl_non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    if rl_non_tensor:
        import numpy as np

        n_replay = len(replay_dp)
        for k, arr in rl_non_tensor.items():
            trailing = getattr(arr, "shape", (0,))[1:]
            replay_dp.non_tensor_batch[k] = np.full((n_replay, *trailing), None, dtype=object)

    merged = DataProto.concat([batch, replay_dp])

    # Shuffle rows so the replay block (concatenated at the tail) is spread
    # across ALL mini batches. verl slices mini batches sequentially WITHOUT
    # shuffle (tensordict_utils.make_iterator -> DataLoader shuffle defaults
    # False), so without this the replay rows would land only in the final
    # ceil(n_replay / ppo_mini_batch_size) mini batches -- every earlier mini
    # batch's optimizer.step() would carry ZERO replay gradient, splitting PPO
    # learning and CL anti-forgetting apart in time. A per-step permutation
    # (seeded off the buffer step for reproducibility) distributes replay rows
    # uniformly so each mini batch's combined loss (rl + lambda_3 * replay)
    # sees replay. Row-level shuffle does not touch the loss composition, so
    # lambda_3 semantics are unchanged.
    n_total = len(merged)
    gen = torch.Generator()
    gen.manual_seed(int(shuffle_seed))
    merged.reorder(torch.randperm(n_total, generator=gen))
    return merged


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
        import socket
        from pprint import pprint

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        config = merge_verl_config(config)
        print(f"CLTaskRunner hostname: {socket.gethostname()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        # Rebuild config from a plain container so that list-valued nodes (e.g.
        # trainer.logger = ['console','swanlab']) become native lists instead of
        # ListConfig nodes.  OmegaConf 2.3 resolve() rejects ListConfig as a
        # "non-primitive" value; recreating from container sidesteps that.
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
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

    cfg = merge_verl_config(cfg)

    if not ray.is_initialized():
        ray_init_kwargs = cfg.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create(
            {**OmegaConf.to_container(ray_init_kwargs), "runtime_env": runtime_env}
        )
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Pre-import verl agent_loop to avoid circular import in Ray workers.
    # verl's agent_loop/__init__.py has a known circular dependency:
    # single_turn_agent_loop.py imports from agent_loop which is still being
    # initialized. Pre-importing here ensures the module is fully loaded
    # before any Ray worker tries to import it.
    import verl.experimental.agent_loop  # noqa: F401

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
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.config import validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

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
        cfg.data.train_files,
        cfg.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=cfg.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        cfg.data.val_files,
        cfg.data,
        tokenizer,
        processor,
        is_train=False,
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
