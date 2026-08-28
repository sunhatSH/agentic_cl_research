"""Serper + Jina web provider plugin — bundled into the sandbox image.

把项目已有的两个 web 能力（google.serper.dev 搜索 + r.jina.ai 抓取）包成一个
hermes web provider，让模型能直接 call ``web_search`` / ``web_extract``（原来这两个
函数因为没配 backend provider 而 undefined，模型只能靠 terminal 调 /opt/tools 裸脚本）。

register() 做三件事：
  1. 注册 SerperWebProvider（search→serper、extract→jina），供 web_search/web_extract
     dispatch 时按 web.search_backend=serper 路由到它。
  2. ★ override=True 重注册 web_search / web_extract，把 check_fn 换成认 serper 的自定义
     检查。**必须 override**：hermes 内置 check_web_api_key（tools/web_tools.py:852）硬编码
     只认 exa/tavily/firecrawl/... 8 个 backend、不认 serper、无 registry fallback → 返回
     False → registry.py:417 据此把 web_search/web_extract 从模型可见工具列表剔除。即
     provider 注册了、dispatch 能跑（explicit config wins），但工具对模型【不可见】→ 仍
     undefined。override 换 check_fn 后工具才对模型可见。handler/schema/is_async 逐字照抄
     内置注册（web_search sync、web_extract async）。
  3. 注册 web_fetch 作为 web_extract 的别名（模型习惯叫 web_fetch，hermes 官方名是
     web_extract → 别名让两种叫法都命中）。

启用：configs/exps/hermes.config.yaml
    plugins:
      enabled: [web-serper]
    web:
      search_backend: serper
      extract_backend: serper

Env（沙箱镜像 runtime.env 注入 → HermesHarness 写进 ~/.hermes/.env）：
    SERPER_API_KEY   — 搜索（google.serper.dev）
    JINA_API_KEY     — 抓取（r.jina.ai）
"""

from __future__ import annotations

import os

from .provider import SerperWebProvider

# web_fetch 别名的 schema：镜像 hermes 的 WEB_EXTRACT_SCHEMA（tools/web_tools.py），
# 只把 name 换成 web_fetch，参数（urls / char_limit）保持一致。
_WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": (
        "Fetch content from web page URLs (alias of web_extract). Returns clean page "
        "content in markdown/text (no LLM summarization). Also works with PDF URLs. "
        "Pass a list of URLs (max 5). If a URL fails or times out, use the browser tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to fetch content from (max 5 URLs per call)",
                "maxItems": 5,
            },
            "char_limit": {
                "type": "integer",
                "description": "Optional per-page character budget (default 15000).",
                "minimum": 2000,
            },
        },
        "required": ["urls"],
    },
}


def _serper_web_available() -> bool:
    """web 工具可见性门槛（替换内置 check_web_api_key）：serper/jina 任一 key 存在即可用。

    ★ 必须走 hermes 的 get_env_value（先 os.environ、再 ~/.hermes/.env 文件）——
    HermesHarness 把 SERPER/JINA key 写进沙箱 ~/.hermes/.env，但【不】export 到进程
    os.environ；hermes 内置 check 也用 get_env_value 读两处。若只查 os.environ（旧实现），
    key 只在 .env 时 → 返回 False → check_fn 假 → web_search 对模型不可见 → undefined
    （2026-08-28 评测实测 web_search 10/11 次 undefined 的真根因）。get_env_value import
    失败时降级回 os.environ（本地无 hermes 的单测/离线场景）。廉价、不发网络请求。
    """
    def _get(key: str) -> str:
        try:
            from hermes_cli.config import get_env_value
            return (get_env_value(key) or "").strip()
        except Exception:  # noqa: BLE001 — 离线/无 hermes 时降级
            return os.environ.get(key, "").strip()

    return bool(_get("SERPER_API_KEY") or _get("JINA_API_KEY"))


def register(ctx) -> None:
    """Register provider + override web_search/web_extract (serper check_fn) + web_fetch 别名。"""
    ctx.register_web_search_provider(SerperWebProvider())

    # ── override web_search / web_extract，绕过不认 serper 的内置 check_web_api_key ──
    try:
        from tools.web_tools import (
            WEB_EXTRACT_SCHEMA,
            WEB_SEARCH_SCHEMA,
            web_extract_tool,
            web_search_tool,
        )

        # web_search：sync，handler 逐字照抄内置（tools/web_tools.py 注册处）。
        ctx.register_tool(
            name="web_search",
            toolset="web",
            schema=WEB_SEARCH_SCHEMA,
            handler=lambda args, **kw: web_search_tool(
                args.get("query", ""), limit=args.get("limit", 5)
            ),
            check_fn=_serper_web_available,   # ← 换成认 serper 的检查
            emoji="🔍",
            override=True,
        )
        # web_extract：async，handler + is_async 照抄内置。
        ctx.register_tool(
            name="web_extract",
            toolset="web",
            schema=WEB_EXTRACT_SCHEMA,
            handler=lambda args, **kw: web_extract_tool(
                args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [],
                "markdown",
                char_limit=args.get("char_limit"),
            ),
            check_fn=_serper_web_available,
            is_async=True,
            emoji="📄",
            override=True,
        )
    except Exception as exc:  # noqa: BLE001 — override 失败不该拖垮 provider 注册
        import logging

        logging.getLogger(__name__).warning(
            "web_search/web_extract override 失败（工具可能仍被 check_web_api_key gate）: %s", exc
        )

    # ── web_fetch 别名 → 转调 web_extract_tool（经 extract_backend=serper 走 jina）──
    try:
        from tools.web_tools import web_extract_tool

        def _web_fetch_handler(args, **kw):
            urls = args.get("urls", [])
            urls = urls[:5] if isinstance(urls, list) else []
            return web_extract_tool(urls, "markdown", char_limit=args.get("char_limit"))

        ctx.register_tool(
            name="web_fetch",
            toolset="web",
            schema=_WEB_FETCH_SCHEMA,
            handler=_web_fetch_handler,
            check_fn=_serper_web_available,   # 与 web_extract 同口径，保证可见
            is_async=True,       # web_extract_tool 是 async，与 web_extract 注册一致
            emoji="📄",
            description="Fetch web page content (alias of web_extract).",
        )
    except Exception as exc:  # noqa: BLE001 — 别名失败不该拖垮 provider 注册
        import logging

        logging.getLogger(__name__).warning("web_fetch alias 注册失败（忽略）: %s", exc)
