"""阶段 D：9桶 buffer 在 verl v1 (custom_sync / KVBatchMeta) 下的 hook 适配.

见 plan swift-juggling-toast 阶段 D。用户已定：9桶 buffer 用【我们自己的】方案
（BucketReplayBuffer + 掺回放行），【不】子类化 v1 的 ReplayBuffer —— 后者是 online
off-policy 采样器（按 model version 新鲜度门控、选完即 kv_clear），跟我们"跨桶持久回放
防遗忘"是正交的两个维度。

与旧 install_buffer_hooks（trainer/verl_runner.py，hook DataProto 版 _update_actor）的差异：
v1 的 ``_update_actor(batch: KVBatchMeta, metrics)`` —— batch 是 transfer_queue 的元数据
句柄（keys/tags/partition_id），实际张量在 tq 里。所以：
  · PRE 掺回放行：把回放张量 ``kv_batch_put`` 写进 tq → ``KVBatchMeta.concat`` 合并 keys。
  · POST 抽 winner：``kv_batch_get_by_meta`` 从 tq 取 rollout 张量 → 按 task_id 挑 winner
    → buffer.add_trajectory。
b1 baseline（buffer.enabled=false / lambda_replay=0）不装此 hook。仅 R 系列触发。

⚠️ 本文件强依赖 transfer_queue / KVBatchMeta 运行时行为，本机无 GPU/tq server 无法端到端
验证；集群 R 系列跑前需实测（见文末 CLUSTER-TODO）。b1 不受影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# tq 字段：回放行与 rollout 行必须共享同一组 fields 才能 concat（KVBatchMeta.concat 校验
# fields 集合一致）。回放行补齐 rollout 的字段名，rollout 行补齐回放专属字段（镜像零值）。
_REPLAY_PARTITION = "train"


def install_buffer_hooks_v1(trainer: Any, buffer: Any | None, cfg: Any) -> None:
    """给 v1 custom_sync trainer 装 9桶 buffer hook（掺回放行 + 抽 winner 入库）.

    调用时机：CLTaskRunnerV1.run 里 ``trainer.init()`` 之后、``fit()`` 之前
    （与 inject_cl_loss 并列）。buffer is None（b1）时空转。
    """
    cl = cfg.get("cl", {}) or {}
    # lambda_replay：CL loss 里 L_replay 项的系数（=0 时无回放 loss，buffer 也不装）。
    lambda_replay = float(cl.get("lambda_replay", 0.0))

    if buffer is None and lambda_replay <= 0:
        # 无 buffer 且无 replay：只装 std metrics hook，不装回放逻辑
        pass

    # replay_batch_size：回放量【上限兜底】。replay_ratio>0 时实际回放量由 ratio 动态算，
    #   此值只作封顶（buffer 再大也不超它）。replay_ratio<=0 时退回固定用它作回放量。
    replay_batch_size = int(cl.get("replay_batch_size", 512))
    # replay_ratio：训练 batch 里【新:旧】的比值（默认 5 = 每 5 条新 rollout 配 1 条回放）。
    #   关键语义（易误解，务必分清两个独立概念）：
    #     · "新" = 本 step 实际新 rollout 行数（32 query × multi-turn ≈ 300 行，全部参与训练，
    #       不限于 winner）。winner 只是【进 buffer 存起来】的那 1 条/query。
    #     · "旧" = 从 buffer 采样的回放行数 = ceil(新行数 / ratio)。
    #   所以回放占比 = 1/(ratio+1)：ratio=5 → 1/6≈16.7%，ratio=2 → 1/3≈33%，ratio=1 → 1/2=50%。
    #   >0 时回放量按新轨迹数动态算（新数据因沙箱失败缩水时回放同步缩小、占比不漂移，见 RunLog 2026-07-30）。
    #   <=0 时退回固定 replay_batch_size。replay_batch_size 仍作上限兜底。
    replay_ratio = float(cl.get("replay_ratio", 5.0))
    # replay_warmup_size：冷启动 ramp。>0 时回放量随 buffer 填充线性爬坡（0→满额），
    #   <=0 关闭 ramp（buffer 一有数据就满额回放）。R0 空启动 buffer 前几步不足时，
    #   sampler 会 min(采样量, buffer 实际大小) 尽可能回放、不报错。
    replay_warmup_size = int(cl.get("replay_warmup_size", 0))
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

    stats_logger = BufferStatsLogger(f"logs/buffer_stats/{exp_name}.jsonl")
    tokenizer = getattr(trainer, "tokenizer", None)
    original_update = trainer._update_actor  # v1 签名: (batch: KVBatchMeta, metrics: dict)

    def patched_update(batch, metrics):
        rl_batch = batch  # 抽 winner 用【掺入前】的 rollout-only batch（回放行不回灌 buffer）
        replay_rows: dict = {}
        replay_meta = None

        # ── 1. PRE：从 9桶 buffer 采旧桶 winner，掺进 tq + 合并 KVBatchMeta ──
        if lambda_replay > 0:
            # 回放量 eff_replay 的动态计算：
            #   new_count = 本 step 非 padding 行数（≈300，32 query × multi-turn 每 query ~9-10 行；
            #     ⚠️ 是【rollout 行数】不是 query 数(32)）。这些行【全部参与训练】，不限于 winner。
            #   eff_replay = ceil(new_count / replay_ratio) → 每 ratio 条新行配 1 条回放，
            #     replay_batch_size 作上限封顶。replay_ratio<=0 时退回固定 replay_batch_size。
            eff_replay = replay_batch_size
            if replay_ratio > 0:
                new_count = sum(
                    not tag.get("is_padding", False) for tag in getattr(rl_batch, "tags", []) or []
                )
                # ceil(new_count / ratio)，整数除法实现，避免 import math
                eff_replay = -(-new_count // int(replay_ratio)) if new_count > 0 else 0
                # replay_batch_size 作上限兜底（buffer 再大也不超它）
                if replay_batch_size > 0:
                    eff_replay = min(eff_replay, replay_batch_size)
            # 采样 eff_replay 条回放行；buffer 前几步不足时 prepare_replay_rows→sampler 会
            # min(采样量, buffer 大小) 尽可能回放（不报错），随 step 累积爬满。
            replay_rows = prepare_replay_rows(
                buffer, weighting, tokenizer, eff_replay, warmup_size=replay_warmup_size
            )
            if replay_rows:
                shuffle_seed = int(getattr(trainer, "global_steps", 0) or 0)
                batch, replay_meta = _append_replay_rows_v1(
                    batch, replay_rows, rl_batch, shuffle_seed=shuffle_seed
                )

        # ── 2. 原生 actor 更新（worker 从 tq 按 batch.keys 取张量训练）──
        result = original_update(batch, metrics)

        # ── 2b. 计算 reward std / group std 并写入 metrics（无论有无 buffer 都落盘）──
        _merge_std_metrics(rl_batch, metrics)

        step = getattr(trainer, "global_steps", 0)

        # 以下 buffer 操作仅在 buffer 启用时执行
        if buffer is not None:
            buffer.set_step(step)

            # ── 3. POST：forgetting_risk 回填（对刚回放的轨迹重算 current-policy logprob）──
            tids = replay_rows.get(REPLAY_TIDS_KEY) if replay_rows else None
            if tids and forgetting_update_freq > 0 and step % forgetting_update_freq == 0:
                means = compute_replay_current_logprobs(trainer, replay_rows)
                if means is not None:
                    backfill_forgetting(buffer, tids, means)

            # ── 4. POST：抽 winner（每 task_id reward 最高）入 9桶 buffer ──
            _ingest_winners(rl_batch, buffer, exp_name, step)

            # ── 5. POST：buffer 动态证据（metrics + sidecar JSONL）──
            if stats_log_freq > 0 and step % stats_log_freq == 0:
                stats = buffer.stats()
                _merge_buffer_metrics(metrics, stats, flatten_buffer_stats)
                stats_logger.log(step, stats)

            # ── 6. POST：周期性 buffer 全量快照 ──
            if save_freq > 0 and step > 0 and step % save_freq == 0:
                snap = Path(f"buffer_dumps/{exp_name}-step-{step}.sqlite")
                snap.parent.mkdir(parents=True, exist_ok=True)
                buffer.dump(snap)

        # ── 7. 清理本 step 掺入的回放 keys（不让它们污染下一 step 的 tq 元数据）──
        if replay_meta is not None:
            _clear_replay_keys(replay_meta)

        return result

    trainer._update_actor = patched_update
    tag = "9桶 buffer + std metrics" if buffer is not None else "std metrics only (no buffer)"
    print(f"[cl] v1: hook 已装 ({tag}, patched _update_actor, KVBatchMeta 适配)", flush=True)


def _merge_buffer_metrics(metrics: dict, stats: Any, flatten_fn) -> None:
    """把扁平化的 buffer stats 并进 v1 fit 的 metrics dict（wandb/file logger 可见）。

    v1 的 _update_actor(batch, metrics) 直接拿到 metrics 字典（不同于 v0 的 meta_info），
    所以这里直接 update 即可。"""
    if isinstance(metrics, dict):
        metrics.update(flatten_fn(stats))


def _ingest_winners(rl_batch: Any, buffer: Any, exp_name: str, step: int) -> None:
    """从 rollout batch 抽每 task_id 的 winner（reward 最高）入 9桶 buffer.

    v1: rl_batch 是 KVBatchMeta，需从 tq 取张量 + tags。用 v1 版 extractor。"""
    from data.cleaning import strip_zw
    from trainer.trajectory_adapter_v1 import extract_trajectories_from_kvbatch

    bucket_names = getattr(buffer, "bucket_names", None)
    # 单桶实验(R0 CLEAR):所有轨迹默认归入唯一桶,不做 bucket 过滤
    default_bucket = bucket_names[0] if bucket_names and len(bucket_names) == 1 else None

    groups: dict[str, list[tuple[Any, str, dict]]] = {}
    for trajectory, bucket, meta in extract_trajectories_from_kvbatch(
        rl_batch, default_bucket=default_bucket, valid_buckets=bucket_names
    ):
        # trajectory from v1 extractor is a list of messages; v0 returned {"messages": [...]}
        msgs = trajectory if isinstance(trajectory, list) else trajectory.get("messages", [])
        for msg in msgs:
            if isinstance(msg.get("content"), str):
                msg["content"] = strip_zw(msg["content"])
        tid = meta.get("task_id") or ""
        groups.setdefault(tid, []).append((trajectory, bucket, meta))

    winners = []
    for _tid, candidates in groups.items():
        best = max(candidates, key=lambda x: float(x[2].get("reward", 0) or 0))
        buffer.add_trajectory(*best)
        winners.append(best)

    _persist_winners(winners, exp_name, step)
    _persist_rollout_status(groups, exp_name, step)


def _persist_rollout_status(groups: dict, exp_name: str, step: int) -> None:
    """记录每 query(task_id) 的 n 条 rollout 全量轨迹 + 成功状态 + reward。

    排查 reward 波动/rollout 失败用。写 rollouts/training/<exp>/rollout_status-<step>.jsonl，
    全量 messages 占空间，但每次实验启动时由 _train_impl.sh 清掉上一轮目录(防磁盘膨胀)。
    winner 不入此文件——winner 单独进训练回放池(buffer.add_trajectory)。
    """
    if not groups:
        return
    import json

    out_dir = Path(f"rollouts/training/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rollout_status-{step}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for tid, candidates in groups.items():
            bucket = candidates[0][1] if candidates else None
            rollouts = []
            n_success = 0
            for traj, _bkt, meta in candidates:
                st = str(meta.get("status", "unknown"))
                rw = meta.get("reward")
                if st == "success":
                    n_success += 1
                msgs = traj if isinstance(traj, list) else traj.get("messages", [])
                rollouts.append({"status": st, "reward": rw, "messages": msgs})
            row = {
                "step": step,
                "task_id": tid,
                "bucket": bucket,
                "n_rollouts": len(rollouts),
                "n_success": n_success,
                "rollouts": rollouts,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[persist] rollout 全量轨迹 {len(groups)} queries → {out_file}", flush=True)


def _append_replay_rows_v1(batch, replay_rows: dict, rl_batch, shuffle_seed: int = 0):
    """把回放张量写进 tq，返回 (合并后的 KVBatchMeta, 回放 KVBatchMeta).

    方案 A（RunLog §65）：concat 要求两 chunk 的 **field 名集合完全相等**
    （transfer_queue concat: ``set(chunk.fields) != base_fields_set`` 即崩）。
    rollout 行天然没有 3 个 replay 专属字段（is_replay / replay_response_mask /
    replay_token_weights），故：
      1. 回放行 TensorDict 对齐 rollout 字段（缺补零）+ 带 3 个 replay 专属字段。
      2. 写回放行进 tq(train 分区)，建 replay_meta。
      3. **给 rollout keys 补 3 个 replay 专属字段的零值**（镜像 verl 给已存在 key 加
         old_log_probs/advantages 的 async_put 追加语义，transferqueue_utils.py:255）。
      4. 用扩展后的 fields 重建 batch meta → concat 两侧字段集一致 → 通过。

    ⚠️ 集群验证项：本机无 transfer_queue，kv_batch_put 对已存在 key 是否"追加字段"
    （非整行替换）需集群实测（verl 自身用同路径加字段，强证据为追加）。"""
    try:
        import torch
        import transfer_queue as tq
        from tensordict import TensorDict
        from transfer_queue import KVBatchMeta
    except ImportError:
        return batch, None

    fields_td = _replay_tensordict(replay_rows, batch)
    if fields_td is None or fields_td.batch_size[0] == 0:
        return batch, None

    n = fields_td.batch_size[0]
    base_step = int(shuffle_seed)
    replay_keys = [f"replay-{base_step}-{i}_{0}_{0}" for i in range(n)]
    tags = [{"is_padding": False, "is_replay": True, "status": "finished"} for _ in range(n)]

    tq.kv_batch_put(keys=replay_keys, partition_id=_REPLAY_PARTITION, fields=fields_td, tags=tags)
    replay_meta = KVBatchMeta(
        keys=replay_keys,
        tags=tags,
        partition_id=_REPLAY_PARTITION,
        fields=list(fields_td.keys()),
    )

    # ── 给 rollout keys 补 3 个 replay 专属字段（零值），使字段集与回放行一致 ──
    rollout_keys = list(getattr(batch, "keys", []) or [])
    batch_fields = list(getattr(batch, "fields", None) or [])
    replay_only = [k for k in ("is_replay", "replay_response_mask", "replay_token_weights") if k not in batch_fields]
    if rollout_keys and replay_only:
        m = len(rollout_keys)
        # response 段宽度 R：优先取回放行的（与回放 mask/weights 同宽），回退 1。
        rmask = fields_td.get("replay_response_mask") if "replay_response_mask" in fields_td.keys() else None
        R = rmask.shape[1] if rmask is not None and rmask.dim() == 2 else 1
        zeros: dict = {}
        for k in replay_only:
            if k == "is_replay":
                zeros[k] = torch.zeros(m, dtype=torch.bool)
            elif k == "replay_response_mask":
                zeros[k] = torch.zeros((m, R), dtype=torch.long)
            else:  # replay_token_weights
                zeros[k] = torch.zeros((m, R), dtype=torch.float32)
        tq.kv_batch_put(
            keys=rollout_keys,
            partition_id=getattr(batch, "partition_id", _REPLAY_PARTITION),
            fields=TensorDict(zeros, batch_size=m),
        )
        # 用扩展后的 fields 重建 batch meta（concat 读 data[0].fields 作 base）。
        batch = KVBatchMeta(
            keys=rollout_keys,
            tags=list(getattr(batch, "tags", []) or []),
            partition_id=getattr(batch, "partition_id", _REPLAY_PARTITION),
            fields=batch_fields + replay_only,
            extra_info=getattr(batch, "extra_info", None),
        )

    merged = KVBatchMeta.concat([batch, replay_meta])
    return merged, replay_meta


_REPLAY_ONLY_FIELDS = ("is_replay", "replay_response_mask", "replay_token_weights")
_SEQ_FIELDS = ("input_ids", "attention_mask", "position_ids")
_LONG_FIELDS = ("input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask")


def plan_replay_fields(replay_field_names, batch_fields):
    """纯函数：算回放 TensorDict 应含哪些字段、各字段来源（复用回放值还是补零）。

    方案 A（RunLog §65）：concat 要求回放字段集 == rollout(batch_fields)。返回：
      {field: "use"}  —— rollout 声明且回放有 → 用回放值
      {field: "zero"} —— rollout 声明但回放没有 → 补零
      + 3 个 replay 专属字段（回放有则 "use"）
    batch_fields 为空（拿不到 rollout 字段）→ 返回 None（调用方走旧行为：直接用回放全字段）。

    抽成纯函数是为了【离线单测】字段对齐（tensordict/torch 本机没装，真函数进不去）。"""
    batch_fields = list(batch_fields or [])
    if not batch_fields:
        return None
    have = set(replay_field_names)
    plan: dict[str, str] = {}
    for f in batch_fields:
        plan[f] = "use" if f in have else "zero"
    for f in _REPLAY_ONLY_FIELDS:
        if f in have:
            plan[f] = "use"
    return plan


def _replay_tensordict(replay_rows: dict, batch):
    """把 prepare_replay_rows 产出的张量 dict 转成 tq 需要的 TensorDict，字段对齐 rollout.

    方案 A（RunLog §65）：concat 要求回放行字段集 == rollout(``batch.fields``)。
    字段选择逻辑抽到纯函数 ``plan_replay_fields``（离线可单测）；本函数只按 plan
    组装 torch 张量。补零形状：序列字段 [n,P+R]，response 段字段 [n,R]。"""
    try:
        import torch
        from tensordict import TensorDict
    except ImportError:
        return None

    from trainer.replay_forward import REPLAY_TIDS_KEY

    rows = {k: v for k, v in replay_rows.items() if k != REPLAY_TIDS_KEY and hasattr(v, "shape")}
    if not rows:
        return None
    n = next(iter(rows.values())).shape[0]

    plan = plan_replay_fields(list(rows.keys()), getattr(batch, "fields", None))
    if plan is None:
        # 拿不到 rollout 字段集（本机/降级）→ 按回放自有字段走（旧行为）。
        return TensorDict(rows, batch_size=n)

    resp = rows.get("responses")
    R = resp.shape[1] if resp is not None and resp.dim() == 2 else 1
    inp = rows.get("input_ids")
    T = inp.shape[1] if inp is not None and inp.dim() == 2 else R

    aligned: dict = {}
    for f, src in plan.items():
        if src == "use":
            aligned[f] = rows[f]
        else:  # zero
            width = T if f in _SEQ_FIELDS else R
            dtype = torch.long if f in _LONG_FIELDS else torch.float32
            aligned[f] = torch.zeros((n, width), dtype=dtype)
    return TensorDict(aligned, batch_size=n)



def _clear_replay_keys(replay_meta) -> None:
    """清掉本 step 掺入的回放 keys（下一 step 的 tq 元数据不含它们）。"""
    try:
        import transfer_queue as tq

        tq.kv_clear(keys=list(replay_meta.keys), partition_id=replay_meta.partition_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[cl] v1: 清理回放 keys 失败(不致命): {exc}", flush=True)


def _persist_winners(winners: list, exp_name: str, step: int) -> None:
    """Write winner trajectories as JSONL for offline analysis（与 v0 版一致）。"""
    if not winners:
        return
    import json

    out_dir = Path(f"rollouts/training/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"step-{step}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for traj, bucket, meta in winners:
            # v1 trajectory 是 list，v0 是 {"messages": [...]}
            msgs = traj if isinstance(traj, list) else traj.get("messages", [])
            row = {
                "task_id": meta.get("task_id", ""),
                "bucket": bucket,
                "reward": meta.get("reward"),
                "messages": msgs,
                "step": step,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[persist] {len(winners)} winners → {out_file}", flush=True)


def _merge_std_metrics(rl_batch: Any, metrics: dict) -> None:
    """从 v1 rollout batch 的 rm_scores 张量直接算 reward_std / group_reward_std。

    v1 的 reward 在 tq 的 rm_scores 张量里（[N, R]，最后有效 token 位是 reward，
    其余为 0，sum(-1) 得标量）。不走 extract_trajectories_from_kvbatch —— 后者强制
    要求 bucket 解析成功（v1 实测 bucket 未进 tag，会 skip 大半轨迹），而 std 指标
    不需要 bucket。group by task_id（从 tags 取）后统计组内 std（GRPO 组内一致性）。
    """
    try:
        import numpy as np
        import transfer_queue as tq
    except ImportError:
        return

    try:
        td = tq.kv_batch_get_by_meta(rl_batch, select_fields=["rm_scores", "extra_info"])
    except Exception:  # noqa: BLE001
        return
    if td is None or "rm_scores" not in td:
        return

    rm = td["rm_scores"]
    try:
        rewards = rm.detach().float().sum(-1).tolist()
    except (AttributeError, RuntimeError):
        return

    tags = getattr(rl_batch, "tags", None)
    by_task: dict[str, list[float]] = {}
    valid: list[float] = []
    for i, r in enumerate(rewards):
        tag = tags[i] if tags and i < len(tags) else None
        if isinstance(tag, dict) and (tag.get("is_replay") or tag.get("is_padding")):
            continue
        rv = float(r)
        valid.append(rv)
        tid = ""
        # v1 的 task_id(record_id) 在 extra_info field 里，不在 tag；优先从 extra_info 取。
        if "extra_info" in td:
            try:
                val = td["extra_info"][i]
                ei = val.data if hasattr(val, "data") and not hasattr(val, "detach") else val
                if isinstance(ei, dict):
                    tid = ei.get("record_id") or ei.get("task_id") or ""
            except (IndexError, KeyError, TypeError):
                pass
        if not tid and isinstance(tag, dict):
            tid = tag.get("task_id") or tag.get("record_id") or tag.get("id") or ""
        if not tid:
            tid = f"row{i}"
        by_task.setdefault(tid, []).append(rv)

    if not valid:
        return
    arr = np.array(valid, dtype=np.float64)
    metrics["cl/reward_mean"] = float(arr.mean())
    metrics["cl/reward_std"] = float(arr.std())

    group_stds = []
    for t_rewards in by_task.values():
        if len(t_rewards) >= 2:
            group_stds.append(float(np.std(t_rewards, dtype=np.float64)))
    if group_stds:
        gs = np.array(group_stds, dtype=np.float64)
        metrics["cl/group_reward_std"] = float(gs.mean())
        metrics["cl/group_reward_std_max"] = float(gs.max())
        metrics["cl/num_groups"] = len(group_stds)

    # advantage_std: if verl already computed it, reuse critic/advantages/mean
    # vs std; otherwise leave empty so plotter skips it.


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER-TODO（R 系列跑前集群实测）：
#  1. v1 rollout 写进 tq(train 分区)的字段清单 —— 回放行必须完全对齐（concat 校验）。
#     实测：训练时 dump 一条 rollout key 的 fields，据此补齐 _replay_tensordict。
#  2. 回放行 key 格式 "replay-{step}-{i}_{0}_{0}" 是否与 ReplayBuffer 的 uid_session_index
#     解析兼容（replay_buffer.py:203 用 key.split("_")[0] 取 uid；我们的 uid=replay-{step}-{i}）。
#  3. concat 后的 batch 进 _balance_batch / _compute_old_log_prob 是否对回放行(response_mask=0)
#     正确处理（回放行不该重算 advantage —— 已在 rollout 阶段前置，但 v1 流水线顺序需实测）。
#  4. extract_trajectories_from_kvbatch（trajectory_adapter_v1.py）能否从 tq 正确取回
#     messages/bucket/reward（v1 non_tensor 字段来源，见该文件 CLUSTER-TODO）。
# ─────────────────────────────────────────────────────────────────────────────
