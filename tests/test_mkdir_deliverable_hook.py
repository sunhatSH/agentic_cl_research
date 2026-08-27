"""MkdirDeliverableHook.parse_deliverable_dirs 纯函数单测（离线，无需沙箱）。"""

from __future__ import annotations

import asyncio

from trainer.mkdir_deliverable_hook import MkdirDeliverableHook, parse_deliverable_dirs


def test_deep_file_takes_parent_dir():
    ins = "Write results to /home/user/workspace/tcs_wave_a/frontend/js/views/app.js"
    assert parse_deliverable_dirs(ins) == ["/home/user/workspace/tcs_wave_a/frontend/js/views"]


def test_root_level_file_needs_no_mkdir():
    # 根下单文件：父目录就是 workspace 根（init_command 已建）→ 不返回
    ins = "Save to /home/user/workspace/out.json"
    assert parse_deliverable_dirs(ins) == []


def test_one_level_dir_file():
    ins = "Create /home/user/workspace/tests/test_x.py"
    assert parse_deliverable_dirs(ins) == ["/home/user/workspace/tests"]


def test_dir_reference_without_file():
    # 末段无扩展名 → 整体当目录
    ins = "Inspect the project at /home/user/workspace/incident-svc/incident_svc"
    assert parse_deliverable_dirs(ins) == ["/home/user/workspace/incident-svc/incident_svc"]


def test_outputs_not_parsed():
    # 输入输出统一 workspace(2026-08-26)：/home/user/outputs 不再单独支持
    ins = "dump to /home/user/outputs/sub/metrics.json"
    assert parse_deliverable_dirs(ins) == []


def test_multiple_dedup_sorted():
    ins = (
        "read /home/user/workspace/tests/test_a.py and "
        "write /home/user/workspace/tests/test_b.py plus "
        "/home/user/workspace/data/out.csv"
    )
    assert parse_deliverable_dirs(ins) == [
        "/home/user/workspace/data",
        "/home/user/workspace/tests",
    ]


def test_trailing_punctuation_stripped():
    ins = "保存到 /home/user/workspace/reports/final.md。"
    assert parse_deliverable_dirs(ins) == ["/home/user/workspace/reports"]


def test_no_path_returns_empty():
    assert parse_deliverable_dirs("Summarize the CSV and reply in chat.") == []
    assert parse_deliverable_dirs("") == []


def test_ignores_other_absolute_paths():
    # 只认 /home/user/workspace；/tmp、/etc、/home/user/outputs 等不碰
    ins = "write /tmp/scratch/x.txt and /etc/foo/bar.conf and /home/user/outputs/y.txt"
    assert parse_deliverable_dirs(ins) == []


# ── 实例化 + prepare/run 冒烟：抓"缺抽象方法 run"这类上集群才崩的坑 ──
# 2026-08-27：MkdirDeliverableHook 只实现 prepare、漏 run（AgentRunHook.run 是 @abstractmethod），
# 实例化即 TypeError，评测 270+ session abort。纯函数单测测不到（从不实例化）→ 加此冒烟。


class _FakeSandbox:
    def __init__(self) -> None:
        self.cmds: list[str] = []

    async def exec(self, cmd: str, timeout: int | None = None) -> None:
        self.cmds.append(cmd)


class _Ctx:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction


class _State:
    def __init__(self) -> None:
        self.reward_info: dict = {}


def test_hook_instantiable_has_run():
    # 缺 run（抽象方法）时这一步就 TypeError；能建成实例即证明 run 已实现。
    hook = MkdirDeliverableHook()
    assert hook.run_on_agent_error is False


def test_prepare_runs_mkdir_for_deep_path():
    hook = MkdirDeliverableHook()
    sb = _FakeSandbox()
    ctx = _Ctx("Write to /home/user/workspace/tests/test_x.py")
    asyncio.run(hook.prepare(sb, ctx, _State()))
    assert sb.cmds == ["mkdir -p /home/user/workspace/tests"]


def test_prepare_noop_without_paths():
    hook = MkdirDeliverableHook()
    sb = _FakeSandbox()
    asyncio.run(hook.prepare(sb, _Ctx("reply in chat"), _State()))
    assert sb.cmds == []


def test_run_is_noop():
    # run 阶段（agent 命令后）无操作，不 exec、不抛。
    hook = MkdirDeliverableHook()
    sb = _FakeSandbox()
    asyncio.run(hook.run(sb, _Ctx("anything"), _State()))
    assert sb.cmds == []
