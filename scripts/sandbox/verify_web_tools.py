#!/usr/bin/env python3
"""验证新推的 v2 沙箱镜像里 web provider 插件是否生效（真机 e2b，dev 机可跑）。

检查三件事：
  1. /home/user/.hermes/plugins/web/serper/ 下三个插件文件是否在镜像里
  2. HermesHarness 会写的 config 里 plugins.enabled / web.backend（这里只验静态插件，
     config 是运行时写的，故直接看 hermes 能否发现插件）
  3. hermes tools 是否列出 web_search / web_extract / web_fetch 且可用

用法：
  source scripts/env/load_tencent_env.sh   # 或脚本内已 source
  /mnt/afs_toolcall/sunhao4/miniconda3/bin/python3 scripts/verify_web_tools.py
"""
from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))


def _load_env():
    """从 docker/sandbox/*.env 读 e2b + serper/jina 凭证进 os.environ。"""
    # 本脚本在 scripts/sandbox/ → 仓库根要上两级。
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for fn in ("tencent.env", "runtime.env", "image.env"):
        p = os.path.join(root, "docker", "sandbox", fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    _load_env()
    if not os.environ.get("E2B_API_KEY") or not os.environ.get("E2B_DOMAIN"):
        print("❌ 缺 E2B_API_KEY / E2B_DOMAIN", flush=True)
        return 2

    template = os.environ.get("SANDBOX_TEMPLATE", "agentic-cl-sandbox")
    print(f"起 e2b 沙箱 template={template} (当前 Tool 指向的镜像)...", flush=True)
    from e2b_code_interpreter import Sandbox

    sb = Sandbox.create(template=template, timeout=300)
    try:
        def run(cmd):
            r = sb.commands.run(cmd, timeout=90)
            return (r.stdout or "") + (r.stderr or "")

        print("\n=== 1. 插件文件是否在镜像里 ===", flush=True)
        out = run("ls -la /home/user/.hermes/plugins/web/serper/ 2>&1 || echo MISSING")
        print(out, flush=True)
        has_plugin = all(f in out for f in ("plugin.yaml", "__init__.py", "provider.py"))
        print(f"→ 三文件齐: {'✅' if has_plugin else '❌ 缺失'}", flush=True)

        print("\n=== 2. plugins/web/__init__.py 存在（缺它 import 失败，插件被静默跳过）===", flush=True)
        out_pkg = run("ls /home/user/.hermes/plugins/web/__init__.py 2>&1 && echo PKG_OK || echo PKG_MISSING")
        has_pkg = "PKG_OK" in out_pkg
        print(f"→ web/__init__.py: {'✅' if has_pkg else '❌ 缺失(插件包不可 import)'}", flush=True)

        print("\n=== 3. .hermes 是否 user 可写（chown 修复，缺则每 session Permission denied）===", flush=True)
        out_perm = run("ls -ld /home/user/.hermes; touch /home/user/.hermes/_wtest 2>&1 "
                       "&& rm -f /home/user/.hermes/_wtest && echo WRITABLE || echo READONLY")
        writable = "WRITABLE" in out_perm
        print(out_perm.strip(), flush=True)
        print(f"→ .hermes 可写: {'✅' if writable else '❌ 只读(hermes 建 cron 会崩)'}", flush=True)

        # 写真实 config.yaml（base64 避免转义）——模拟 HermesHarness 运行时注入的 plugins/web 段。
        cfg = ("plugins:\n  enabled: [web-serper]\n"
               "web:\n  search_backend: serper\n  extract_backend: serper\n")
        b64 = base64.b64encode(cfg.encode()).decode()
        run(f"mkdir -p /home/user/.hermes && echo {b64} | base64 -d > /home/user/.hermes/config.yaml")

        print("\n=== 4. ★ hermes 真的加载了 web-serper 插件吗（决定性检查）===", flush=True)
        pl = run("hermes plugins list 2>&1")
        loaded = "web-serper" in pl
        # 打印含 serper 的行（或提示没有）
        serper_lines = [ln for ln in pl.splitlines() if "serper" in ln.lower()]
        print("\n".join(serper_lines) if serper_lines else "(hermes plugins list 里无 web-serper)", flush=True)
        print(f"→ hermes 加载 web-serper: {'✅' if loaded else '❌ 未加载(import 失败/manifest 无效)'}", flush=True)

        print("\n=== 5. hermes tools list 里 web toolset 是否 enabled ===", flush=True)
        tl = run("hermes tools list 2>&1")
        web_tool = [ln for ln in tl.splitlines() if "web" in ln.lower() and ("enabled" in ln.lower() or "🔍" in ln)]
        web_ok = bool(web_tool)
        print("\n".join(web_tool) if web_tool else "(web toolset 未出现在 tools list)", flush=True)
        print(f"→ web toolset: {'✅ enabled' if web_ok else '❌ 未启用'}", flush=True)

        print("\n=== 6. serper/jina key 是否注入沙箱（评测/训练由 HermesHarness.env 注入 .hermes/.env）===", flush=True)
        out_key = run("env | grep -oE '^(SERPER_API_KEY|JINA_API_KEY)=' | sort -u")
        print(out_key.strip() or "(此裸实例未注入——正常，训练/评测时 HermesHarness 注入)", flush=True)

        all_ok = has_plugin and has_pkg and writable and loaded and web_ok
        print("\n=== 结论 ===", flush=True)
        if all_ok:
            print("✅ 全部通过：插件文件齐 + web/__init__.py 在 + .hermes 可写 + hermes 加载 web-serper "
                  "+ web toolset enabled。web_search/web_extract/web_fetch 对模型可用。", flush=True)
        else:
            fails = []
            if not has_plugin: fails.append("插件文件缺")
            if not has_pkg: fails.append("web/__init__.py 缺")
            if not writable: fails.append(".hermes 只读")
            if not loaded: fails.append("hermes 未加载插件")
            if not web_ok: fails.append("web toolset 未启用")
            print(f"❌ 未通过：{', '.join(fails)}。", flush=True)
            print("   处理：确认 git pull 最新 → BUILD_NO_CACHE=1 重建 → push → "
                  "UpdateSandboxTool 刷新 digest（见 doc/ops/sandbox/Sandbox_冒烟指南.md §7）。", flush=True)
        return 0 if all_ok else 1
    finally:
        try:
            sb.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
