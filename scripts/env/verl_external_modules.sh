#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VERL_USE_EXTERNAL_MODULES 单一来源（训练 + 评测共用，source 我）
#
# 为什么单独一份：这些 patch 必须在【每个】verl 进程 import verl 时消费——尤其
# AgentSessionWorker（真正跑 AsyncSandbox.create / create_hooks 的进程）。verl/__init__
# 靠 VERL_USE_EXTERNAL_MODULES（逗号分隔）决定 import 哪些外部模块；经 verl_runner.py 的
# runtime_env passthrough 传进所有 Ray worker。
#
# 历史坑（2026-08-26，本文件诞生原因）：评测脚本从不设 VERL_USE_EXTERNAL_MODULES →
# worker 不 import trainer.observer_hook_register → verl 原生 create_hook 不认 FQN hook
# `trainer.observer_hook.ObserverDiffHook`（configs/exps/agent_loop_config.yaml 挂的）→
# 每个 eval session 一起来就 ValueError: Unknown post-run hook → 72 session 全 abort、
# rollout 空转。训练能跑是因为 _train_impl.sh 设了、评测漏了。抽成共享文件根治漂移。
#
# 各 patch 作用：
#   · rollout.e2b_http1_patch      —— 关 e2b SDK 默认 http2 走 HTTP/1.1，从根上消除腾讯
#       AGS 网关的 GOAWAY（单 HTTP/2 连接 ~1000 stream 后回收）。CL_E2B_DISABLE_HTTP2=0 可关。
#   · trainer.observer_hook_register —— monkey-patch recipe_custom hook factory 认 FQN hook
#       名（如 trainer.observer_hook.ObserverDiffHook），使自定义 hook 无需改 verl 源码即可挂载。
#   · trainer.pause_generation_bounded_patch —— lightllm pause_generation 的无限 while True 改成
#       有界等待超时放行（CL_PAUSE_MAX_WAIT 默认 180s）。lightllm refcount 泄漏(4卡16卡都有)让
#       abort_all 僵尸请求回收不掉，16卡每 step 边界 pause 时无限重试→死锁 hang（§59，step6 卡死）。
#       放行让 16卡泄漏像 4卡一样良性(泄漏但不死)。不改 LightLLM 源码。
#   · trainer.dataproto_tensordict_patch —— v1 引擎 worker 的 DataProto→TensorDict 系统性修复。
#       r0 掺回放行后 infer_batch/train_batch/train_mini_batch(engine_workers.py) 的 data 入参
#       实际收到 DataProto(非函数标注的 TensorDict),函数体用 TensorDict API(data.keys()/data.shape[0]/
#       tu.pop/tu.assign_non_tensor/tu.make_iterator) 每 step 崩(b1 无 replay 是 TensorDict 不崩)。
#       在 tqbridge 层统一把 DataProto→to_tensordict(),一处覆盖全部 @register 分发函数(不再逐点
#       打地鼠);并保留 tu.pop 的 DataProto 兼容作兜底。
#   · trainer.empty_batch_skip_patch —— 空 batch skip 守卫(E18):all_failed_policy 只在生成阶段
#       查 num_success_outputs==0 就 skip;但"部分成功、组过滤后又全丢"这条边界会把空 batch 送进
#       _balance_batch → get_seqlen_balanced_partitions assert "number of items:[0] < k_partitions"
#       整训练崩(2026-08-21 实测,config 丢文件恢复瞬间触发)。patch CustomPPOTrainerSync._balance_batch
#       进函数查空→抛 _EmptyBatchSkip,step 捕获→记 rollout/empty_batch_skip 并 return None,复用
#       fit() 既有 batch is None skip 分支,跳过该步继续而非崩。
#
# 注：eval 不训练（不走 _balance_batch），empty_batch_skip_patch 在 eval 里 import 后
# 只是包了个不被调用的方法，无副作用——保留在同一列表里换取"训练/评测一份清单"，不再分叉。
#
# 注：trainer.image_trajectory_drop_patch（含图轨迹过滤 E13）当前【未】列入——沿用 2026-08-26
# 之前训练脚本的实际行为（注释描述过但循环从未加载它）。抽共享文件时保持训练行为逐字不变，
# 不擅自新增 patch；若日后要启用，训练/评测同改此一处即可。
# ─────────────────────────────────────────────────────────────────────────────

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe_custom.bootstrap}"
# 都必须在 worker 进程生效（patch 目标都在 worker/lightllm 副本），故走 VERL_USE_EXTERNAL_MODULES
# 而非 driver-only import。逐个幂等去重。
for _mod in rollout.e2b_http1_patch trainer.observer_hook_register trainer.pause_generation_bounded_patch trainer.dataproto_tensordict_patch trainer.empty_batch_skip_patch; do
  case ",$VERL_USE_EXTERNAL_MODULES," in
    *,"$_mod",*) : ;;  # 已含,不重复追加
    *) export VERL_USE_EXTERNAL_MODULES="$VERL_USE_EXTERNAL_MODULES,$_mod" ;;
  esac
done
