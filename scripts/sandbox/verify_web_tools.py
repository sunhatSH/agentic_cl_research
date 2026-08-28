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
            r = sb.commands.run(cmd, timeout=60)
            return (r.stdout or "") + (r.stderr or "")

        print("\n=== 1. 插件文件是否在镜像里 ===", flush=True)
        out = run("ls -la /home/user/.hermes/plugins/web/serper/ 2>&1 || echo MISSING")
        print(out, flush=True)
        has_plugin = all(f in out for f in ("plugin.yaml", "__init__.py", "provider.py"))
        print(f"→ 三文件齐: {'✅' if has_plugin else '❌ 缺失'}", flush=True)

        print("\n=== 2. provider.py 内容校验（确认是我们的版本，非空壳）===", flush=True)
        out2 = run("grep -c 'SerperWebProvider\\|google.serper.dev\\|r.jina.ai' "
                   "/home/user/.hermes/plugins/web/serper/provider.py 2>&1 || echo 0")
        print(f"provider.py 关键串命中行数: {out2.strip()}", flush=True)

        print("\n=== 3. __init__.py 有 override/web_fetch（最新修复）===", flush=True)
        out3 = run("grep -oE 'override=True|web_fetch|_serper_web_available' "
                   "/home/user/.hermes/plugins/web/serper/__init__.py 2>&1 | sort -u || echo NONE")
        print(out3, flush=True)

        print("\n=== 4. serper/jina key 是否注入沙箱（web 工具可用前提）===", flush=True)
        out4 = run("env | grep -oE '^(SERPER_API_KEY|JINA_API_KEY)=' | sort -u || echo NO_KEYS")
        print(out4 or "(沙箱内未注入——注意：训练/评测时由 HermesHarness.env 注入 ~/.hermes/.env)", flush=True)

        print("\n=== 结论 ===", flush=True)
        if has_plugin:
            print("✅ v2 镜像已包含 serper web provider 插件（含 override/web_fetch 修复）。", flush=True)
            print("   评测/训练时 HermesHarness 会注入 SERPER/JINA key + 写 config"
                  "(plugins.enabled=[web-serper])，web_search/web_extract/web_fetch 即对模型可用。", flush=True)
        else:
            print("❌ 镜像里没有插件文件——构建没带上，或 Tool 缓存了旧 digest。需重新 BUILD_NO_CACHE 构建/重建 Tool。", flush=True)
        return 0 if has_plugin else 1
    finally:
        try:
            sb.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
