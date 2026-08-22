"""阶段 F：observer diff 取证 hook（反 reward-hacking）+ answer_key 注入.

见 plan swift-juggling-toast 阶段 F。旧自写 rollout 的核心反 reward-hacking 机制：
observer 以【沙箱 before/after diff（含文件内容）】为 ground truth，判断 agent 是否真的
产出了交付物，而非只看模型自述。迁到 verl 原生 agent_loop 后，这条链断了（reward 只喂
模型 response 文本）。本 hook 恢复它。

机制（AgentRunHook 契约，recipe_custom/agent/runners/hooks/base.py）：
  · prepare(sandbox, ctx, state)：agent 跑之前，对沙箱工作区做只读 snapshot（baseline）。
  · run(sandbox, ctx, state)：sandbox 关之前，做 post snapshot + diff，把确定性证据写进
    state.reward_info["observer_report"]；并按 record_id 读 taskspecs_w3/<rid>/answer_key.json
    写进 reward_info["answer_key"]。二者经 reward_info → tq → omni extra_info → judge。

复用 agents/observer.py 的探针常量(_SNAPSHOT_PROBE/_SYS_PROBE)与纯 python diff
(diff_snapshots/diff_system/build_deterministic_report)。唯一差异：recipe_custom hook 是
async + sandbox.exec，而 observer 原探针走同步 sandbox.run_code —— 这里用 async exec 跑
同一段探针脚本，取 stdout 后在 driver 侧 json.loads + diff（diff 是纯 python，无关同步异步）。

⚠️ 本机无 e2b 沙箱无法端到端验证；集群实测（见文末 CLUSTER-TODO）。b1 baseline 可不挂此 hook
（reward 先用 judge 文本打分）；作为 reward 质量增强，R 系列 / 需强 grounding 时挂。
"""

from __future__ import annotations

import json
import os
from typing import Any

from recipe_custom.agent.runners.hooks.base import AgentRunHook

# taskspecs 根（与 cl_agent_dataset 同源）：<record_id>/answer_key.json 是 judge ground truth。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/trainer → repo
TASKSPECS_ROOT = os.environ.get("CL_TASKSPECS_ROOT", os.path.join(_REPO_ROOT, "datasources", "taskspecs_w3"))
# generated_tasks_hermes 根：<D<N>>/<gen_task_id>/answer_key.json 是训练集(unk_* record_id)的
# judge ground truth。与 cl_agent_dataset.GEN_TASKS_ROOT 同源，靠 extra_info.gen_task_id 定位。
GEN_TASKS_ROOT = os.environ.get("CL_GEN_TASKS_ROOT", os.path.join(_REPO_ROOT, "datasources", "generated_tasks_hermes"))
# 沙箱工作区（探针 os.walk 的根）。与 fs-seed 注入的 ./inputs 同一工作目录。
SANDBOX_WORKSPACE = os.environ.get("CL_SANDBOX_WORKSPACE", ".")


async def _run_json_probe_async(sandbox: Any, probe: str, timeout: int = 60) -> dict:
    """async 版 _run_json_probe：sandbox.exec 跑探针脚本，取 stdout → json。永不抛。"""
    try:
        cmd = "python3 -c " + _shquote(probe)
        res = await sandbox.exec(cmd, timeout=timeout)
        out = (getattr(res, "stdout", "") or "").strip()
        if len(out) > 1_000_000:
            out = out[:1_000_000]
        data = json.loads(out) if out else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- 取证绝不能崩会话
        return {}


def _shquote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _record_id_from_ctx(ctx: Any) -> str | None:
    """从 ctx.extra（=runner kwargs，含 dataset 透传字段）取 record_id。"""
    extra = getattr(ctx, "extra", {}) or {}
    ei = extra.get("extra_info") or {}
    if isinstance(ei, dict):
        rid = ei.get("record_id")
        if rid:
            return str(rid)
    # 回退：reward_model.ground_truth 里可能带 task_id
    rm = extra.get("reward_model") or {}
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, dict) and gt.get("task_id"):
            return str(gt["task_id"])
    return None


def _gen_task_id_from_ctx(ctx: Any) -> str | None:
    """从 ctx.extra.extra_info 取 gen_task_id（generated_tasks_hermes 定位用）。"""
    extra = getattr(ctx, "extra", {}) or {}
    ei = extra.get("extra_info") or {}
    if isinstance(ei, dict):
        tid = ei.get("gen_task_id")
        if tid:
            return str(tid)
    return None


def _load_answer_key(record_id: str | None, gen_task_id: str | None = None) -> dict | None:
    """加载 judge ground truth。优先 gen_task_id → generated_tasks_hermes/<D>/<tid>/answer_key.json
    （训练集 record_id 是 unk_* 与 taskspecs_w3 不通）；回退 record_id → taskspecs_w3/<rid>/。"""
    candidates = []
    if gen_task_id:
        d_prefix = gen_task_id.split("_", 1)[0]  # D10_k982304_zh → D10
        candidates.append(os.path.join(GEN_TASKS_ROOT, d_prefix, gen_task_id, "answer_key.json"))
    if record_id:
        candidates.append(os.path.join(TASKSPECS_ROOT, record_id, "answer_key.json"))
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                continue
    return None


class ObserverDiffHook(AgentRunHook):
    """沙箱 before/after diff 取证 + answer_key 注入，写进 state.reward_info。

    run_on_agent_error=True：即使 agent 失败/超时，也采 diff（可能有部分交付物），让 judge
    基于真实产出打分，而非因异常直接 0（避免"失败即静默 0"掩盖部分完成）。
    """

    run_on_agent_error = True

    async def prepare(self, sandbox: Any, ctx: Any, state: Any) -> None:
        # agent 跑之前采 baseline（fs-seed 已由 runner 在 write_agent_assets 阶段注入 ./inputs，
        # 所以 baseline 已含输入文件 → diff 只暴露 agent 新产出/修改，不把输入文件误判为交付物）。
        from agents.observer import _SNAPSHOT_PROBE, _SYS_PROBE

        state.reward_info["_observer_pre_fs"] = await _run_json_probe_async(sandbox, _SNAPSHOT_PROBE)
        state.reward_info["_observer_pre_sys"] = await _run_json_probe_async(sandbox, _SYS_PROBE)

    async def run(self, sandbox: Any, ctx: Any, state: Any) -> None:
        from agents.observer import (
            _SNAPSHOT_PROBE,
            _SYS_PROBE,
            _format_changes,
            _is_runtime_file,
            diff_snapshots,
            diff_system,
        )

        pre_fs = state.reward_info.pop("_observer_pre_fs", None)
        pre_sys = state.reward_info.pop("_observer_pre_sys", None)
        post_fs = await _run_json_probe_async(sandbox, _SNAPSHOT_PROBE)
        post_sys = await _run_json_probe_async(sandbox, _SYS_PROBE)

        # runtime/framework 文件过滤（复用 observer._is_runtime_file：具名 + *.log/*.pid），
        # 与 snapshot_workspace 一致，避免 envd.log/jupyter.log 等噪声进 diff。
        def _filter(snap: dict | None) -> dict:
            if not isinstance(snap, dict):
                return {}
            return {p: r for p, r in snap.items() if not _is_runtime_file(p)}

        diff = diff_snapshots(_filter(pre_fs), _filter(post_fs))
        sys_diff = diff_system(pre_sys, post_sys)

        # observer_report = 人类可读正文（_format_changes）：三段式 新增/改变/删除，每段
        # 含【完整文件内容】，改变的文件含 [BEFORE]/[AFTER] 前后对比。这才是给 judge 看的
        # ground truth 正文（不是 dataclass repr）。judge 靠它核对 actor 自述是否属实。
        state.reward_info["observer_report"] = _format_changes(diff, sys_diff)
        state.reward_info["state_diff"] = diff
        answer_key = _load_answer_key(_record_id_from_ctx(ctx), _gen_task_id_from_ctx(ctx))
        if answer_key is not None:
            state.reward_info["answer_key"] = answer_key
        # 交付物计数（judge/completion 参考）：agent 新增/修改的文件数。
        state.reward_info["deliverable_count"] = len(diff.get("added", [])) + len(diff.get("modified", []))


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER-TODO（集群实测）：
#  1. sandbox.exec("python3 -c <probe>") 在 agentic-cl-sandbox 镜像里能跑（python3 存在、
#     探针依赖的 openpyxl/python-docx/pdfplumber 装了 → 二进制提取生效；缺库降级不崩）。
#  2. ctx.extra 是否确实带 extra_info（record_id）—— 依赖 session_worker 把 dataset 的
#     extra_info 透传进 runner_kwargs（worker.py:497 白名单含 extra_info）。若不带，改从
#     raw_prompt 或 reward_model.ground_truth 取 record_id。
#  3. observer_report/answer_key 经 reward_info → tq → omni._prepare_item 的 extra_info →
#     model_reward_omni.compute_score 是否读得到（见阶段 F reward 回流 CLUSTER-TODO）。
#  4. SANDBOX_WORKSPACE 与 hermes 实际工作目录一致（探针 os.walk 的根）。
# ─────────────────────────────────────────────────────────────────────────────
