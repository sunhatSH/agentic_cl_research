"""含图/视频 trajectory 全过滤 —— 训练入口丢弃,不进 PPO. 零侵入 monkey-patch,不改 verl 源码.

问题(E13,§57 的完整根因,2026-08-19 定位)：本项目是**纯文本 CL 训练**(train.parquet 无图字段),
但 agent 沙箱工具会产 PNG 截图 → 含图 rollout 轨迹漏进训练。含图行在模型 forward 时视觉 token
被 vision encoder 展开成 patch embedding(如 776 patch ≈ 28×28 grid),导致 `log_probs` 的 token 数
(input_ids.offsets(),含视觉展开) ≠ `responses`/`response_mask` 的 token 数(原始未展开) →
`ppo_loss` 里 `log_prob`(按 responses 切) − `old_log_prob`(按 response_mask 切) 长度错位崩
(`RuntimeError: size of tensor a(23861) != b(24637)`,差值 776 = 一张图的 patch 数)。

为什么在这里过滤(而非 §57 的 gateway 层 / TEXT_MODEL_ONLY=2)：
  · gateway 层拦(`gateway_image_drop_patch.py`,§57)实测漏得多(含图 incremental cache/多轮拼接/
    session 边界复杂),且源码已丢失。用户要求下移到「trajectory→训练」入口全过滤。
  · TEXT_MODEL_ONLY=2(关引擎视觉)能根治但**大幅降模型能力**(Qwen3.5-9B 是多模态基座,disable_vision
    改架构/RoPE,纯文本能力也损)。故保持 TEXT_MODEL_ONLY=1(冻结视觉、保能力),含图在此过滤。

机制(复用现成的组过滤,不造新逻辑)：patch `AgentSessionWorker._is_trainable_trajectory`,含图
trajectory 判为**不可训练** → session_worker 的 `_filter_trainable_trajectories` 自动剔除 →
若某 session 全部轨迹被剔 → `not trainable_trajectories` → 该 session 计 failure →
现成的 `min_group_success_ratio=0.5` 组过滤自动接管：组内含图 ≤半用剩下纯文本继续训、
>半整组丢(worker.py:579-614)。整组被剔空时靠 config 的 `all_failed_policy: skip` 跳过该 step。

判定信号(用户定=并集,任一命中即含图)：
  1. `trajectory.multi_modal_data` 非空(带 images/videos)——最上游最准,session_worker 拿到的
     Trajectory 对象直接带此字段(types.py:32)。
  2. `len(response_ids) != len(response_mask)`——E13 崩溃的直接症状(双保险,防 multi_modal_data
     漏标但长度已分叉的边界情况)。

触发：经 `VERL_USE_EXTERNAL_MODULES`(scripts/_train_impl.sh for 循环追加本模块名)在【每个】verl
进程(含 AgentSessionWorker,即真正跑 rollout 的进程)import verl 时 import 本模块 → import 即 patch。
幂等。b1/纯文本正常轨迹不受影响(multi_modal_data 空 + 长度相等 → 判定不含图)。
"""

from __future__ import annotations

_PATCHED = False


def _is_image_trajectory(trajectory) -> bool:
    """判定 trajectory 是否含图/视频(并集：multi_modal_data 非空 OR responses/mask 长度不等)。

    纯函数,不依赖 verl,可离线单测。任一信号命中即判含图(丢弃)。
    """
    # 信号 1：multi_modal_data 非空(带 images/videos)。session_worker 的 Trajectory 直接带。
    mmd = getattr(trajectory, "multi_modal_data", None)
    if mmd:
        # dict 非空(有 images/videos 键且值非空)才算含图；空 dict {}/None → 纯文本。
        if isinstance(mmd, dict):
            if any(mmd.get(k) for k in ("images", "videos")):
                return True
        else:
            return True  # 非 dict 的非空 multi_modal_data(保守判含图)

    # 信号 2：response_ids 与 response_mask 长度不等(E13 崩溃直接症状,双保险)。
    rid = getattr(trajectory, "response_ids", None)
    rmask = getattr(trajectory, "response_mask", None)
    if rid is not None and rmask is not None:
        try:
            if len(rid) != len(rmask):
                return True
        except TypeError:  # noqa: BLE001 -- 非序列(异常轨迹)不在此判,交回原逻辑
            pass

    return False


def install() -> None:
    """Monkey-patch AgentSessionWorker._is_trainable_trajectory：含图轨迹判不可训练。幂等。"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from recipe_custom.agent.session_worker.worker import AgentSessionWorker
    except Exception as exc:  # noqa: BLE001 -- recipe_custom absent off-cluster
        print(f"[cl] image-trajectory-drop patch 跳过（recipe_custom 不可用: {exc}）", flush=True)
        return

    _orig_is_trainable = AgentSessionWorker._is_trainable_trajectory

    def _is_trainable_no_image(self, trajectory) -> bool:
        # 原判定(trace_type in trainable_trace_types)先过；再叠加「非含图」。
        if not _orig_is_trainable(self, trajectory):
            return False
        if _is_image_trajectory(trajectory):
            # 含图轨迹判不可训练 → _filter_trainable_trajectories 剔除 → 空 session 计 failure →
            # min_group_success_ratio 组过滤接管。这里不打印(每条都打会刷屏),统计靠现成的
            # num_group_size_filtered_* / num_failed_sessions 指标。
            return False
        return True

    AgentSessionWorker._is_trainable_trajectory = _is_trainable_no_image
    _PATCHED = True
    print(
        "[cl] image-trajectory-drop patch 已安装（含图/视频轨迹判不可训练→剔除,复用 "
        "min_group_success_ratio 组过滤,不改 verl 源码）",
        flush=True,
    )


# import 即安装（与 observer_hook_register 的"import 触发 patch"风格一致）。
install()
