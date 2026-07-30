"""omni RewardManager 适配层：把项目的 model judge 接进 recipe_custom 的 omni reward。

背景（迁移 §阶段3）：迁到 verl 原生 agent_loop 后，reward 走 recipe_custom 的
OmniRewardManager（`reward.reward_manager.name=omni`）。omni 调 reward_fn 的约定是::

    reward_fn(response, ground_truth, *, eos_token, extra_info, data_source,
              user_question, judge_model_url, session_id, ..., **kwargs)

而项目的 judge 入口 `trainer.model_reward.compute_score` 签名是::

    compute_score(data_source, solution_str, ground_truth, extra_info=, **kwargs)

二者仅【位置参数顺序】不同（omni 前两位是 response、ground_truth；我们的第一位是
data_source）。本模块做一层薄适配，转发到 model_reward.compute_score，observer diff
证据仍经 extra_info["observer_report"] 传入（保留 §16 那条 claim→diff 取证链）。

接入方式（不改 dependencies/verl 的注册表）：数据 / config 里把
``reward_model.reward_fn`` 设成 dict::

    {"_function_name": "trainer.model_reward_omni.compute_score"}

omni 的 make_object_from_config 会 import_and_get 这个 fqdn（omni.py:45-46），
故无需在 recipe_custom/reward_score/omni_reward/__init__.py 的 _COMPUTE_SCORE_CONFIG
里加 key。返回结构 {"score": float, ...} 与 omni 期望一致。
"""

from __future__ import annotations

from typing import Any

from trainer.model_reward import compute_score as _cl_compute_score


def compute_score(response: str, ground_truth: str = "", **kwargs: Any) -> dict[str, Any]:
    """omni 调用约定 → 转发到 trainer.model_reward.compute_score。

    omni 传法（omni.py:196-212）：第 1 位=response(=solution_str)，第 2 位=ground_truth，
    其余全 kwargs（extra_info / data_source / user_question / judge_model_url /
    data_non_tensor_batch / ...）。我们的 compute_score 需要 (data_source, solution_str,
    ground_truth, extra_info=)。

    阶段 F（observer 回流）：ObserverDiffHook（trainer/observer_hook.py）把沙箱 before/after
    diff 取证写进 state.reward_info["observer_report"] + ["answer_key"]，session_worker 存进
    non_tensor_batch["reward_info"]，omni 又把整个 non_tensor_batch 经 data_non_tensor_batch
    传进来（omni.py:201）。这里【零侵入】从 data_non_tensor_batch["reward_info"] 取出
    observer_report/answer_key 并进 extra_info → judge 拿到确定性证据交叉核对模型自述
    （保留 §16 claim→diff 取证链）。b1 不挂 hook 时 reward_info 无这些 key，自动跳过。
    """
    data_source = kwargs.get("data_source", "agentic_cl")
    extra_info = dict(kwargs.get("extra_info") or {})

    # ── observer 回流：reward_info(hook 写的取证) → extra_info(judge 读) ──
    dntb = kwargs.get("data_non_tensor_batch") or {}
    reward_info = dntb.get("reward_info")
    if isinstance(reward_info, dict):
        for _k in ("observer_report", "answer_key", "state_diff", "deliverable_count"):
            if _k in reward_info and _k not in extra_info:
                extra_info[_k] = reward_info[_k]

    # judge 由 model_reward 内部 env 解析（get_judge()）；judge_model_url 若 omni 传了，
    # 也放进 kwargs 透传（model_reward.compute_score 用 **kwargs 兜住，不影响）。
    result = _cl_compute_score(
        data_source,
        response,
        ground_truth,
        extra_info=extra_info,
        **{k: v for k, v in kwargs.items() if k not in ("data_source", "extra_info")},
    )
    # model_reward 返回 {"score": ..., "completion":..., "judge_error":...}，已含 score。
    # omni 只强依赖 "score" 字段，其余作为 extra 指标带回。
    if isinstance(result, dict):
        return result
    return {"score": float(result)}
