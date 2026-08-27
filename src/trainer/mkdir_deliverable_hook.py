"""Per-task 预建【写产出】目录 hook —— 修 write_file 深层路径写盘失败（不碰 Hermes）。

背景（2026-08-22，见 Bug_Fix_精简总表 F8 + doc/debug/Reward_归因方法与调优经验.md）：
Hermes tools/file_operations.py::_atomic_write 把临时文件 .hermes-tmp 建在【目标父目录】，
父目录不存在就崩（No such file）。数据层已把根路径 /workspace→/home/user/workspace（可写），
init_command 预建了 workspace/outputs 两个根；但少量任务要求写【深层】路径（如
/home/user/workspace/tests/test_x.py、.../tcs_wave_a/frontend/js/views/x.js），中间目录不存在
→ 仍写盘失败。

修法（不改 Hermes，per-task 精准建、零污染）：AgentRunHook.prepare 在 agent 命令【之前】跑，
持有 sandbox（可 exec）+ ctx（含 instruction）。从本任务指令解析出引用的
/home/user/{workspace,outputs}/... 路径，取其父目录，`mkdir -p` 一层层建好——只建【本任务】
自己引用的目录（平均 ~1 个），不像全局 init_command 硬编码要给每个沙箱建一堆无关空目录。
读现有代码库的任务缺的是文件（归 F5 补种子），不是空目录，但预建空父目录对它们无害。

注册：configs/exps/agent_loop_config.yaml 的 hooks 段加 FQN
``trainer.mkdir_deliverable_hook.MkdirDeliverableHook``（observer_hook_register 的 factory
patch 已让 verl 认 FQN hook）。与 ObserverDiffHook 并列，互不影响。
"""

from __future__ import annotations

import re
import shlex
from abc import ABC, abstractmethod
from typing import Any

try:
    from recipe_custom.agent.runners.hooks.base import AgentRunHook
except Exception:  # noqa: BLE001 -- recipe_custom absent off-cluster（单测/本机）

    class AgentRunHook(ABC):  # type: ignore[no-redef]
        """Fallback 基类（离集群无 recipe_custom 时可 import + 单测纯函数）。

        ★ run 必须与真实基类一样是 @abstractmethod：真实 recipe_custom 基类里 run 是抽象方法，
        子类不实现则实例化即 TypeError（2026-08-27 MkdirDeliverableHook 缺 run，评测 270+
        session abort）。fallback 若给 run 默认实现会掩盖此问题——本机实例化不报错、上集群才崩。
        故 fallback 也标 abstractmethod，让本机冒烟单测就能抓到缺失。
        """

        run_on_agent_error = False

        async def prepare(self, sandbox: Any, ctx: Any, state: Any) -> None: ...

        @abstractmethod
        async def run(self, sandbox: Any, ctx: Any, state: Any) -> None: ...


# 只认沙箱内可写根下的路径（输入输出统一 workspace，2026-08-26）。
_PATH_RE = re.compile(r"/home/user/workspace/[\w./\-]+")
# 沙箱根（init_command 已建 /home/user/workspace），无需重复建。
_ROOTS = frozenset({"/home/user/workspace"})


def parse_deliverable_dirs(instruction: str) -> list[str]:
    """从指令解析出需要预建的目录集（去重、剔根、按路径升序）。

    规则：对每个 /home/user/workspace/... 引用——
      - 末段像文件（含 '.' 且不是唯一段）→ 取其父目录；
      - 否则整体当目录。
    根目录（workspace 本身）已由 init_command 建，跳过。纯函数，可离线单测。
    """
    if not instruction:
        return []
    dirs: set[str] = set()
    for raw in _PATH_RE.findall(instruction):
        p = raw.rstrip("./")  # 去掉句末标点/斜杠伪影
        if not p or p in _ROOTS:
            continue
        segs = p.split("/")
        # /home/user/workspace/... → 段 0..2 是 '', 'home', 'user'；workspace 在段 3
        tail = segs[-1]
        if "." in tail and len(segs) > 4:
            parent = "/".join(segs[:-1])
        else:
            parent = p
        if parent and parent not in _ROOTS:
            dirs.add(parent)
    return sorted(dirs)


class MkdirDeliverableHook(AgentRunHook):
    """agent 跑之前，per-task 预建本任务【写产出】引用的深层目录（mkdir -p 一层层建）。"""

    # 目录预建与 agent 是否报错无关；只在 prepare 跑一次即可，run 无操作。
    run_on_agent_error = False

    def __init__(self, settings: Any = None) -> None:
        # 与内置 hook 一致接受可选 settings（本 hook 不需要）。
        self._settings = settings or {}

    async def prepare(self, sandbox: Any, ctx: Any, state: Any) -> None:
        instruction = getattr(ctx, "instruction", "") or ""
        dirs = parse_deliverable_dirs(instruction)
        if not dirs:
            return
        # 一条 mkdir -p 建全部（-p 递归建父链、幂等、已存在不报错）。永不因此崩 agent。
        cmd = "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)
        try:
            await sandbox.exec(cmd, timeout=30)
        except Exception:  # noqa: BLE001 -- 预建失败不该拖垮该 rollout
            pass

    async def run(self, sandbox: Any, ctx: Any, state: Any) -> None:
        # AgentRunHook.run 是 @abstractmethod（集群 recipe_custom 基类），子类必须实现
        # 否则实例化即 TypeError（2026-08-27 评测 270+ session abort 根因）。本 hook 只在
        # prepare 阶段建目录、agent 命令【后】无操作，故 run 留空。
        return None
