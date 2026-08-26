"""MkdirDeliverableHook.parse_deliverable_dirs 纯函数单测（离线，无需沙箱）。"""

from __future__ import annotations

from trainer.mkdir_deliverable_hook import parse_deliverable_dirs


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
