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

        # ★ 决定性检查：直接 in-process 问 hermes 的 PluginManager + ToolRegistry。
        # ⚠️ 不能用 `hermes plugins list`——它只渲染 standalone 插件，所有 kind:backend 的
        #    web provider（bundled exa/tavily/... 和我们的 serper）都【不在该列表里】却已加载。
        #    2026-08-28 踩坑：曾据此误报 ❌"未加载"，实则插件正常工作。
        # 判定三件事（全 True 才算生效）：
        #   a. PluginManager 里 web/serper 已 load、enabled、无 error
        #   b. serper 在 agent.web_search_registry._providers（provider 真注册进 registry）
        #   c. web_search/web_extract/web_fetch 的 check_fn 在有 SERPER_API_KEY 时返回 True
        #      （= 对模型可见；HermesHarness 评测/训练时正是注入 SERPER_API_KEY 到 .hermes/.env）
        probe = (
            "import os\n"
            "from hermes_cli.plugins import get_plugin_manager\n"
            "pm=get_plugin_manager(); pm.discover_and_load(force=True)\n"
            "lp=pm._plugins.get('web/serper')\n"
            "loaded = lp is not None and getattr(lp,'enabled',False) and not getattr(lp,'error',None)\n"
            "import agent.web_search_registry as wsr\n"
            "in_reg = 'serper' in wsr._providers\n"
            "from tools.registry import ToolRegistry\n"
            "import tools.registry as R\n"
            "reg=None\n"
            "for a in dir(R):\n"
            "    o=getattr(R,a)\n"
            "    if isinstance(o,ToolRegistry): reg=o; break\n"
            "vis={}\n"
            "for w in ('web_search','web_extract','web_fetch'):\n"
            "    e=reg.get_entry(w) if reg else None\n"
            "    cf=getattr(e,'check_fn',None) if e else None\n"
            "    vis[w]=(e is not None) and (True if cf is None else bool(cf()))\n"
            "print('LOADED=',bool(loaded));print('INREG=',bool(in_reg))\n"
            "for w,v in vis.items(): print(f'VIS_{w}=',bool(v))\n"
        )
        pb64 = base64.b64encode(probe.encode()).decode()
        print("\n=== 4. ★ 决定性检查：PluginManager + ToolRegistry in-process ===", flush=True)
        # 带 SERPER/JINA key 跑（模拟 HermesHarness 注入），check_fn 才会 True
        pout = run(f"echo {pb64} | base64 -d > /tmp/_verify_web.py && "
                   f"SERPER_API_KEY=verify-probe JINA_API_KEY=verify-probe "
                   f"python3 /tmp/_verify_web.py 2>&1")
        print(pout.strip(), flush=True)
        loaded = "LOADED= True" in pout
        in_reg = "INREG= True" in pout
        vis_all = all(f"VIS_{w}= True" in pout
                      for w in ("web_search", "web_extract", "web_fetch"))
        web_ok = in_reg and vis_all
        print(f"→ web/serper 加载: {'✅' if loaded else '❌'} | "
              f"serper 进 registry: {'✅' if in_reg else '❌'} | "
              f"三工具对模型可见: {'✅' if vis_all else '❌'}", flush=True)

        print("\n=== 5. serper/jina key 是否注入沙箱（评测/训练由 HermesHarness.env 注入 .hermes/.env）===", flush=True)
        out_key = run("env | grep -oE '^(SERPER_API_KEY|JINA_API_KEY)=' | sort -u")
        print(out_key.strip() or "(此裸实例未注入——正常，训练/评测时 HermesHarness 注入)", flush=True)

        all_ok = has_plugin and has_pkg and writable and loaded and web_ok
        print("\n=== 结论 ===", flush=True)
        if all_ok:
            print("✅ 全部通过：插件文件齐 + web/__init__.py 在 + .hermes 可写 + web/serper 加载 "
                  "+ serper 进 provider registry + web_search/web_extract/web_fetch 对模型可见。", flush=True)
        else:
            fails = []
            if not has_plugin: fails.append("插件文件缺")
            if not has_pkg: fails.append("web/__init__.py 缺")
            if not writable: fails.append(".hermes 只读")
            if not loaded: fails.append("web/serper 未加载")
            if not web_ok: fails.append("工具对模型不可见/未进 registry")
            print(f"❌ 未通过：{', '.join(fails)}。", flush=True)
            print("   处理：确认 git pull 最新 → 重建 → push → "
                  "UpdateSandboxTool 刷新 digest（见 doc/ops/sandbox/Sandbox_冒烟指南.md §7）。", flush=True)
        return 0 if all_ok else 1
    finally:
        try:
            sb.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
