#!/usr/bin/env python3
"""验证 write_file 写盘 bug + init_command 修复(真机 e2b 沙箱,CPU 可跑)。

复现 Hermes tools/file_operations.py::_atomic_write 的 shell 逻辑:
  d=<父目录>; tmp="$(mktemp -p "$d" .hermes-tmp.XXXXXX ...)"; cat >"$tmp"; mv -f "$tmp" "$t"

对若干目标路径分别测【修复前】(裸沙箱默认 cwd)与【修复后】(先跑 init_command)。
不依赖 GPU / 模型,只测沙箱文件系统行为。

用法:
  source scripts/env/load_tencent_env.sh
  .venv/bin/python scripts/verify_write_fix.py
"""
from __future__ import annotations

import sys

# Hermes _atomic_write 的 shell 脚本(照抄 tools/file_operations.py:972-988 的核心)
def atomic_write_script(target: str, content: str) -> str:
    q_path = "'" + target.replace("'", "'\"'\"'") + "'"
    import os
    parent = os.path.dirname(target) or "."
    q_parent = "'" + parent.replace("'", "'\"'\"'") + "'"
    tmpl = "'.hermes-tmp.XXXXXX'"
    return (
        "set -e; "
        f"d={q_parent}; t={q_path}; "
        'tmp="$(mktemp -p "$d" ' + tmpl + ' 2>/dev/null '
        '|| mktemp "$d/.hermes-tmp.$$.XXXXXX" 2>/dev/null '
        '|| { tmp="$d/.hermes-tmp.$$"; : > "$tmp" && echo "$tmp"; })"; '
        '[ -n "$tmp" ] || { echo "atomic write: could not create temp file" >&2; exit 1; }; '
        "trap 'rm -f \"$tmp\"' EXIT; "
        f'printf %s {chr(39)}{content}{chr(39)} > "$tmp"; '
        'mv -f "$tmp" "$t"; '
        "trap - EXIT; "
        'echo WROTE_OK "$t"'
    )


# 训练里失败的真实目标路径(取自 harness 日志)
TARGETS = [
    "/home/user/outputs/corr_matrix.json",  # 相对→绝对可写区(成功案例同类)
    "/workspace/app.py",                     # write_missing_parent:/workspace 不存在
    "/compute_correlations.py",              # write_root_noperm:根目录无权限
]

# 拟落地的 init_command(configs/exps/agent_loop_config.yaml)
INIT_CMD = (
    "mkdir -p /home/user/outputs /home/user/workspace && "
    "chmod 777 /home/user 2>/dev/null; "
    "ln -sfn /home/user/workspace /workspace 2>/dev/null; "
    "cd /home/user"
)


def run_one(sb, label: str, prefix: str) -> None:
    print(f"\n===== {label} =====")
    for t in TARGETS:
        script = atomic_write_script(t, "hello")
        cmd = f"{prefix}{script}" if prefix else script
        try:
            out = sb._sb.commands.run(cmd, timeout=60)
            ok = out.exit_code == 0
            msg = (out.stdout or out.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"  [{'OK ' if ok else 'FAIL'}] {t}  ->  {msg[0][:100]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR ] {t}  ->  {str(exc)[:100]}")


def main() -> int:
    sys.path.insert(0, "src")
    from rollout.sandbox_client import make_sandbox

    print("起 e2b 沙箱(agentic-cl-sandbox)...")
    sb = make_sandbox("e2b", template="agentic-cl-sandbox", timeout=600)
    try:
        run_one(sb, "修复前(裸沙箱默认 cwd)", prefix="")
        run_one(sb, "修复后(先跑 init_command)", prefix=f"{INIT_CMD}; ")
    finally:
        sb.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
