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
    # replay_max_length：回放行 prompt/response 各自的 token 上限。【必须与 rollout 的
    # max_response_length 一致】——rollout winner 会进 buffer 成为 replay，同一条轨迹前后
    # 长度上限必须相同，否则 replay 把本可完整的轨迹截断。base.yaml 用 ${data.max_response_length}
    # 插值跟随（r0=65536）。旧硬编码 4096 砍掉 64% 回复 ~35% token，是 bug。config 缺失时
    # fallback 到 data.max_response_length，仍缺才退 65536（与 rollout 默认对齐，不再用 4096）。
    _data_cfg = cfg.get("data", {}) or {}
    _default_max_len = int(_data_cfg.get("max_response_length", 65536) or 65536)
    replay_max_length = int(cl.get("replay_max_length", _default_max_len) or _default_max_len)
    # replay_max_model_len：prompt+response 总长上限，对齐 rollout 的 max_model_len(131072)。
    # 与 replay_max_length(response 上限) 一起构成和 rollout 一致的截断契约（prompt 全留、
    # response≤max_response_length、总≤max_model_len）。config 缺失时回退 data.max_model_len，
    # 仍缺则 None（build_replay_rows 退回 legacy：prompt/response 各自截 replay_max_length）。
    _default_model_len = _data_cfg.get("max_model_len")
    if _default_model_len is None:
        _default_model_len = ((cfg.get("actor_rollout_ref", {}) or {}).get("rollout", {}) or {}).get(
            "max_model_len"
        )
    _rmml = cl.get("replay_max_model_len", _default_model_len)
    replay_max_model_len = int(_rmml) if _rmml else None
    stats_log_freq = int(cl.get("buffer_stats_log_freq", 1))
    forgetting_update_freq = int(cl.get("forgetting_update_freq", 1))
    trainer_cfg = cfg.get("trainer", {}) or {}
    save_freq = int(trainer_cfg.get("save_freq", 0))
    exp_name = trainer_cfg.get("experiment_name", "cl")
    # mini_batch_size：verl 的 make_iterator 断言 batch_size % mini_batch_size == 0。
    # 掺回放行后总行数(rollout+replay)可能不是其整数倍 → 需补 is_padding=True 全零行。
    # verl 在 trainer_base._update_actor 里 mini_batch_size = ppo_mini_batch_size * rollout.n。
    _ar = cfg.get("actor_rollout_ref", {}) or {}
    mini_batch_size = int((_ar.get("actor", {}) or {}).get("ppo_mini_batch_size", 0) or 0) * int(
        (_ar.get("rollout", {}) or {}).get("n", 1) or 1
    )

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
                buffer,
                weighting,
                tokenizer,
                eff_replay,
                max_length=replay_max_length,
                warmup_size=replay_warmup_size,
                max_model_len=replay_max_model_len,
            )
            if replay_rows:
                shuffle_seed = int(getattr(trainer, "global_steps", 0) or 0)
                batch, replay_meta = _append_replay_rows_v1(
                    batch,
                    replay_rows,
                    rl_batch,
                    shuffle_seed=shuffle_seed,
                    mini_batch_size=mini_batch_size,
                    eos_token_id=int(getattr(tokenizer, "eos_token_id", 0) or 0),
                )

        # ── 2. 原生 actor 更新（worker 从 tq 按 batch.keys 取张量训练）──
        result = original_update(batch, metrics)

        # ── 2b. 计算 reward std / group std 并写入 metrics（无论有无 buffer 都落盘）──
        _merge_std_metrics(rl_batch, metrics)

        step = getattr(trainer, "global_steps", 0)

        # ── 2c. 记录本 step 全量 rollout 轨迹（无论有无 buffer 都落盘，排查 reward 用）──
        # B1/K2 无 buffer 也要记；这里统一记一次，_ingest_winners 不再重复记。
        _persist_rollout_status(_extract_rollout_groups(rl_batch), exp_name, step)

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

        # ── 7. 回放 keys 不在此提前清理 ──
        # 之前在此 _clear_replay_keys 会在 fit._compute_metrics 之前清掉 replay keys，
        # 导致 _compute_metrics 的 kv_batch_get(keys=batch.keys) 报
        # "keys or partition were not found"（r0 step2 崩，replay-* 在 partition train 找不到）。
        # 改由 fit 末尾的 kv_clear(keys=batch.keys) 统一清（batch 含 replay+padding，
        # 且 rollout/replay 同 partition="train"，fit 的 kv_clear 全覆盖）。

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


def _extract_rollout_groups(rl_batch, bucket_names=None):
    """从 rl_batch 抽轨迹并按 task_id 分组（strip_zw 处理 content）。

    返回 {task_id: [(trajectory, bucket, meta), ...]}。bucket_names 为空时（B1/K2 无
    buffer）不按桶过滤，全部抽取。与 buffer 无关，可独立用于 rollout 轨迹记录。"""
    from data.cleaning import strip_zw
    from trainer.trajectory_adapter_v1 import extract_trajectories_from_kvbatch

    default_bucket = bucket_names[0] if bucket_names and len(bucket_names) == 1 else None

    groups: dict[str, list[tuple[Any, str, dict]]] = {}
    for trajectory, bucket, meta in extract_trajectories_from_kvbatch(
        rl_batch, default_bucket=default_bucket, valid_buckets=bucket_names
    ):
        msgs = trajectory if isinstance(trajectory, list) else trajectory.get("messages", [])
        for msg in msgs:
            if isinstance(msg.get("content"), str):
                msg["content"] = strip_zw(msg["content"])
        tid = meta.get("task_id") or ""
        groups.setdefault(tid, []).append((trajectory, bucket, meta))
    return groups


def _ingest_winners(rl_batch: Any, buffer: Any, exp_name: str, step: int) -> None:
    """从 rollout batch 抽每 task_id 的 winner（reward 最高）入 9桶 buffer.

    v1: rl_batch 是 KVBatchMeta，需从 tq 取张量 + tags。用 v1 版 extractor。
    注意：rollout 轨迹记录(_persist_rollout_status)已上提到 patched_update 统一做，
    本函数只负责抽 winner 入 buffer + 写 winner 轨迹。"""
    bucket_names = getattr(buffer, "bucket_names", None)
    groups = _extract_rollout_groups(rl_batch, bucket_names)

    winners = []
    for _tid, candidates in groups.items():
        best = max(candidates, key=lambda x: float(x[2].get("reward", 0) or 0))
        buffer.add_trajectory(*best)
        winners.append(best)

    _persist_winners(winners, exp_name, step)


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
                rollout = {"status": st, "reward": rw, "messages": msgs}
                # judge 四维度细分（reward 涨不动根因排查用）
                for _k in ("task_done", "correctness", "trajectory", "safety"):
                    _v = meta.get(f"reward_{_k}")
                    if _v is not None:
                        rollout[_k] = _v
                rollouts.append(rollout)
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


def _append_replay_rows_v1(
    batch, replay_rows: dict, rl_batch, shuffle_seed: int = 0, mini_batch_size: int = 0, eos_token_id: int = 0
):
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
      5. 若掺回放行后总行数不是 mini_batch_size 的整数倍，补 is_padding=True 全零行。

    ⚠️ 集群验证项：本机无 transfer_queue，kv_batch_put 对已存在 key 是否"追加字段"
    （非整行替换）需集群实测（verl 自身用同路径加字段，强证据为追加）。"""
    try:
        import torch
        import transfer_queue as tq
        from tensordict import TensorDict
        from transfer_queue import KVBatchMeta
    except ImportError:
        return batch, None

    fields_td = _replay_tensordict(replay_rows, batch, global_steps=int(shuffle_seed))
    if fields_td is None or fields_td.batch_size[0] == 0:
        return batch, None

    # 保留原 batch 的 extra_info（含 _step_once 写入的 temperature 等）。KVBatchMeta.concat
    # 不会自动继承它（replay_meta 未带 extra_info → merged.extra_info=None），而 verl 的
    # _update_actor 会 batch.extra_info.update(...) → None 崩（r0 19:57 第二次崩）。故 concat
    # 后手动回填。
    extra_info = getattr(batch, "extra_info", None)

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

    # ── 给 rollout keys 补 3 个 replay 专属字段（nested 零值），使字段集与回放行一致 ──
    # ⚠️ 变长：rollout 行是 nested，补的 replay 专属字段也必须 nested（否则 mixed nested/2D
    #   又触发 remove_padding cu_seqlens 错乱）。用 rollout 自己的 response_mask（nested）零化
    #   得到同长的 nested 零模板。
    rollout_keys = list(getattr(batch, "keys", []) or [])
    batch_fields = list(getattr(batch, "fields", None) or [])
    replay_only = [
        k for k in ("is_replay", "replay_response_mask", "replay_token_weights") if k not in batch_fields
    ]
    if rollout_keys and replay_only:
        m = len(rollout_keys)
        resp_mask = None
        try:
            rtd = tq.kv_batch_get_by_meta(batch, select_fields=["response_mask"])
            if rtd is not None and "response_mask" in rtd:
                resp_mask = rtd["response_mask"]
        except Exception:  # noqa: BLE001 -- 读不到就降级定长零
            resp_mask = None

        zeros: dict = {}
        for k in replay_only:
            if k == "is_replay":
                zeros[k] = torch.zeros(m, dtype=torch.bool)
            elif k == "replay_response_mask":
                if resp_mask is not None and getattr(resp_mask, "is_nested", False):
                    zeros[k] = resp_mask * 0
                else:
                    zeros[k] = torch.zeros((m, 1), dtype=torch.long)
            else:  # replay_token_weights
                if resp_mask is not None and getattr(resp_mask, "is_nested", False):
                    zeros[k] = resp_mask.float() * 0
                else:
                    zeros[k] = torch.zeros((m, 1), dtype=torch.float32)
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

    # ── 补 padding 行：掺回放行后总行数(rollout+replay)不是 mini_batch_size(=ppo_mini_batch_size×n)
    #    的整数倍 → 用 verl 原生 upsample_batch_to_divisible_size 补最小 [1,1] 模板行。
    #    ⚠️ 之前自造全零行(pad × [P+R] 全零 input_ids/attention_mask/position_ids)会把
    #    remove_padding 的 cu_seqlens 搅乱(全零 attention_mask → seqlen 记账错乱)，导致
    #    GDN Triton kernel "invalid argument"(r0 step2 崩)。verl 的模板行用 EOS 作
    #    prompt/response、seq_len=2、attention_mask=1，是正确的最小填充。
    if mini_batch_size and mini_batch_size > 0:
        from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size

        before = len(list(getattr(merged, "keys", []) or []))
        merged = upsample_batch_to_divisible_size(merged, mini_batch_size, eos_token_id)
        pad = len(list(getattr(merged, "keys", []) or [])) - before
        if pad > 0:
            # padding 行与回放行一样是【临时】行，须随回放行一起 kv_clear（否则跨 step 在
            # tq train 分区累积，污染下一 step 的采样）。
            pad_meta = KVBatchMeta(
                keys=list(getattr(merged, "keys", []))[-pad:],
                tags=list(getattr(merged, "tags", []))[-pad:],
                partition_id=getattr(merged, "partition_id", _REPLAY_PARTITION),
                fields=list(getattr(merged, "fields", []) or []),
            )
            replay_meta = KVBatchMeta.concat([replay_meta, pad_meta])

    merged.extra_info = extra_info
    return merged, replay_meta


_REPLAY_ONLY_FIELDS = ("is_replay", "replay_response_mask", "replay_token_weights")
_SEQ_FIELDS = ("input_ids", "attention_mask", "position_ids")
# 2D int64 序列字段（session_worker 的 tq 张量 schema，worker.py:1054-1076）。
# ⚠️ loss_mask 也是 int64（= response_mask）——之前漏掉 → 回放补零成 float32 →
#   concat 报 dtype mismatch（r0 16:03 崩）。补零 dtype 用本集合 + _LONG_SCALAR_FIELDS 判定。
_LONG_FIELDS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
)
# 1D int64 标量字段（每行一个，如 num_turns）。补零形状 [n]，不是 [n, R]。
_LONG_SCALAR_FIELDS = frozenset({"num_turns"})
# python int 元数据字段（session_worker 存 python int，非张量也非 str）。回放补零若用
# 空串 "" 会让下游数值运算（如 staleness = global_steps - min_global_steps）报 int-str 崩，
# 故补真实 int（调用方传当前 global_steps）。
_INT_SCALAR_FIELDS = frozenset(
    {
        "min_global_steps",
        "max_global_steps",
        "session_id",
        "global_steps",
    }
)
# dict 型非张量字段（multi_modal_inputs：文本任务是空 dict {}）。回放补零若用空串 ""，
# actor forward 里对它 .get(...) 会 AttributeError（str 无 .get），故补空 dict {}。
_EMPTY_DICT_FIELDS = frozenset({"multi_modal_inputs"})
# 纯文本非张量字段（session_worker 用 NonTensorStack 存，list_of_dict_to_tensordict）。回放
# 补零不能补成 torch.float32 张量，否则 concat 报 tensor-vs-non-tensor。补空串 NonTensorStack。
_NON_TENSOR_FIELDS = frozenset(
    {
        "uid",
        "raw_prompt",
        "data_source",
        "reward_model",
        "extra_info",
        "tools_kwargs",
        "tools",
        "agent_name",
        "env_name",
        "trace_type",
        "routed_experts",
        "no_lora_adapter",  # bool 字段(verl infer_batch 里 pop，缺 key 会 assert)；补空串(falsy 等价 False)
    }
)


def replay_zero_fill_kind(field: str) -> str:
    """回放补零字段按 dtype/形状/类型分类（纯函数，离线单测，不依赖 torch）。

    Returns one of:
      "long_seq"    2D int64 序列字段（prompts/responses/response_mask/loss_mask/input_ids/attention_mask/position_ids）
      "long_scalar" 1D int64 标量张量字段（num_turns）
      "int_scalar"  python int 元数据（global_steps/session_id/... → 补真实 int，非空串）
      "empty_dict"  dict 型非张量字段（multi_modal_inputs → 补 {}）
      "non_tensor"  其余非张量字段（raw_prompt/extra_info/uid/... → 补空串 NonTensorStack）
      "float"       2D float32 序列字段（其余：rollout_log_probs/rm_scores/ref_log_prob/returns/entropy/token_level_* 等）
    """
    if field in _LONG_FIELDS:
        return "long_seq"
    if field in _LONG_SCALAR_FIELDS:
        return "long_scalar"
    if field in _INT_SCALAR_FIELDS:
        return "int_scalar"
    if field in _EMPTY_DICT_FIELDS:
        return "empty_dict"
    if field in _NON_TENSOR_FIELDS:
        return "non_tensor"
    return "float"


def _probe_position_id_sections(batch) -> int:
    """探测 rollout 侧 position_ids 每行的 section 数（M-RoPE 段数）。

    Qwen3.5-9B 用 M-RoPE：session_worker 把 position_ids 每行存成 [S, T]（S=4，
    纯文本也 expand 成 4 段，见 multi_modal_postprocess.py:138/168）。而回放行默认按
    1D [T] 构建（replay_forward.py），二者 concat 打 nested tensor 时 dim 不一致 →
    "Found dimension 1 ... dimension 2"（r0 第三次崩，index 127/128 = rollout|replay 边界）。

    返回 rollout position_ids 的 section 数：per-row 是 [S,T]（tq 张量 [m,S,T]，ndim=3）
    → 返回 S；per-row 是 [T]（tq 张量 [m,T]，ndim=2）→ 返回 1；探测失败 → 返回 1（保守）。
    """
    try:
        import transfer_queue as tq

        td = tq.kv_batch_get_by_meta(batch, select_fields=["position_ids"])
        if td is None or "position_ids" not in td:
            return 1
        pid = td["position_ids"]
        # tq 张量 batch 维在最前：[m, S, T] → per-row [S,T]（M-RoPE）；[m, T] → per-row [T]。
        if hasattr(pid, "dim") and pid.dim() == 3:
            return int(pid.shape[1])
        return 1
    except Exception as exc:  # noqa: BLE001 -- 探测失败不致命，退回 1D
        print(f"[cl] v1: 探测 position_ids section 数失败({exc})；按 1 段(plain RoPE)处理", flush=True)
        return 1


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


# response 对齐的序列字段（长度 = 每行 response 长度，与 _SEQ_FIELDS 的 prompt+response 全长相对）。
_RESPONSE_ALIGNED_FIELDS = frozenset(
    {
        "responses",
        "response_mask",
        "loss_mask",
        "replay_response_mask",
        "replay_token_weights",
        "old_log_probs",
        "ref_log_prob",
        "advantages",
        "rollout_log_probs",
        "rm_scores",
    }
)
# 需 nested（变长）化的序列字段，与 session_worker 的 _TQ_NESTED_SEQUENCE_FIELDS 对齐 + 回放专属。
_NESTED_SEQUENCE_FIELDS = (
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
    "old_log_probs",
    "ref_log_prob",
    "advantages",
    "replay_response_mask",
    "replay_token_weights",
)


def _nested_tensor_from_list(tensors, ragged_idx=None):
    """等价 verl.utils.tensordict_utils.nested_tensor_from_tensor_list，只依赖 torch。

    1D 变长 tensor list → jagged nested；2D（如 position_ids [S,T]）→ ragged_idx 指向
    变长的最后一维。本机无 verl 也能测，避免重复 import verl。
    """
    import torch

    sample_dim = tensors[0].dim()
    if ragged_idx is None:
        ragged_idx = sample_dim
    if sample_dim == 1:
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    cat_dim = ragged_idx - 1
    values = torch.cat(tensors, dim=cat_dim)
    lengths = torch.tensor([t.shape[cat_dim] for t in tensors], dtype=torch.long, device=values.device)
    offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=values.device)
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    nested = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
    nested._ragged_idx = ragged_idx
    return nested


def _nested_tensordict_from_rows(row_dicts):
    """list[dict]（每 dict 一行变长 field）→ nested TensorDict，对齐 v1 session_worker 写法。"""
    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    keys = list(row_dicts[0].keys())
    dict_of_lists = {k: [row[k] for row in row_dicts] for k in keys}
    final: dict = {}
    for k, val_list in dict_of_lists.items():
        if (
            val_list
            and all(isinstance(v, torch.Tensor) for v in val_list)
            and all(v.shape == val_list[0].shape for v in val_list)
        ):
            final[k] = torch.stack(val_list)
        elif val_list and all(isinstance(v, torch.Tensor) for v in val_list):
            final[k] = torch.nested.as_nested_tensor(val_list, layout=torch.jagged)
        else:
            final[k] = NonTensorStack(*val_list)
    td = TensorDict(final, batch_size=[len(row_dicts)])
    # 序列字段强制 nested（即使所有行等长，也要 nested，与 rollout 一致，避免 mixed nested/2D）。
    for f in _NESTED_SEQUENCE_FIELDS:
        if f not in row_dicts[0]:
            continue
        values = [row[f] for row in row_dicts]
        if not values or not all(isinstance(v, torch.Tensor) for v in values):
            continue
        if f == "position_ids" and values[0].dim() == 2:
            ragged_idx = 2
        else:
            ragged_idx = None
        td[f] = _nested_tensor_from_list(values, ragged_idx=ragged_idx)
    return td


def _slice_variable_rows(val, f, prompt_lens, resp_lens, P, position_sections):
    """把 padded 字段 val [n,width] 切成变长 per-row list（去 build_replay_rows 的 left/right pad）。"""
    out = []
    for i, (plen, rlen) in enumerate(zip(prompt_lens, resp_lens, strict=True)):
        if f == "prompts":
            v = val[i, P - plen : P]
        elif f in _RESPONSE_ALIGNED_FIELDS:
            v = val[i, :rlen]
        elif f in _SEQ_FIELDS:
            v = val[i, P - plen : P + rlen]
        else:  # 未知字段，保持整行
            v = val[i]
        if f == "position_ids" and position_sections > 1 and v.dim() == 1:
            v = v.unsqueeze(0).expand(position_sections, -1).contiguous()
        out.append(v)
    return out


def _zero_variable_rows(f, prompt_lens, resp_lens, position_sections, global_steps):
    """补零字段的变长 per-row list（对齐每行 prompt/response 长度）。"""
    import torch

    kind = replay_zero_fill_kind(f)
    out = []
    for _i, (plen, rlen) in enumerate(zip(prompt_lens, resp_lens, strict=True)):
        if kind == "non_tensor":
            out.append("")
        elif kind == "int_scalar":
            out.append(int(global_steps))
        elif kind == "empty_dict":
            out.append({})
        elif kind == "long_scalar":
            out.append(torch.zeros((), dtype=torch.long))
        elif kind == "long_seq":
            width = plen + rlen if f in _SEQ_FIELDS else rlen
            v = torch.zeros(width, dtype=torch.long)
            if f == "position_ids" and position_sections > 1:
                v = v.unsqueeze(0).expand(position_sections, -1).contiguous()
            out.append(v)
        else:  # float：response 对齐
            out.append(torch.zeros(rlen, dtype=torch.float32))
    return out


def _replay_tensordict(replay_rows: dict, batch, global_steps: int = 0):
    """把 prepare_replay_rows 产出的 padded 张量 dict 转成 tq 需要的【变长 nested】TensorDict。

    方案 A（RunLog §65）：concat 要求回放行字段集 == rollout(``batch.fields``)。
    字段选择逻辑抽到纯函数 ``plan_replay_fields``（离线可单测）。

    ⚠️ 变长化（r0 step2 GDN 崩根因）：v1 rollout 行是 nested tensor（session_worker 用
    nested_tensor_from_tensor_list 写 tq）。回放行若写成固定 2D padded，会与 rollout 的
    nested 混在一起 → remove_padding 把 padded 宽(P+R)当序列长 → cu_seqlens 错乱 →
    GDN kernel "invalid argument"。故这里用 attention_mask 恢复每行实际 prompt/response
    长，把回放行切成变长，再用 nested tensor 写 tq，与 rollout 完全同构。
    """
    try:
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
        # 拿不到 rollout 字段集（本机/降级）→ 按回放自有字段走（旧行为，固定 2D）。
        return TensorDict(rows, batch_size=n)

    # M-RoPE section 数：rollout 侧 position_ids 每行是 [S,T]（Qwen3.5-9B S=4）。回放行
    # 默认 1D [T]，切成变长后广播成 [S,len] 与 rollout 对齐。
    position_sections = _probe_position_id_sections(batch)

    # 每行实际 prompt/response 长：从 attention_mask 的 prompt/response 段恢复
    # （build_replay_rows 左 pad prompt / 右 pad response 的约定，见 replay_forward.py）。
    attn = rows["attention_mask"]
    P = rows["prompts"].shape[1] if "prompts" in rows else 0
    prompt_lens = attn[:, :P].sum(dim=1).tolist()
    resp_lens = attn[:, P:].sum(dim=1).tolist()

    # 按字段组装变长 per-row list。
    field_lists: dict[str, list] = {}
    for f, src in plan.items():
        if src == "use":
            field_lists[f] = _slice_variable_rows(rows[f], f, prompt_lens, resp_lens, P, position_sections)
        else:
            field_lists[f] = _zero_variable_rows(f, prompt_lens, resp_lens, position_sections, global_steps)

    row_dicts: list[dict] = [{} for _ in range(n)]
    for f, vals in field_lists.items():
        for i in range(n):
            row_dicts[i][f] = vals[i]
    return _nested_tensordict_from_rows(row_dicts)


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
