"""Serper + Jina web provider plugin — bundled into the sandbox image.

把项目已有的两个 web 能力（google.serper.dev 搜索 + r.jina.ai 抓取）包成一个
hermes web provider，让模型能直接 call ``web_search`` / ``web_extract``（原来这两个
函数因为没配 backend provider 而 undefined，模型只能靠 terminal 调 /opt/tools 裸脚本）。

另注册 ``web_fetch`` 作为 ``web_extract`` 的**别名**（评测日志里模型按通用习惯调过
``web_fetch``，但 hermes 抓取工具官方名叫 ``web_extract`` → undefined）。别名 handler
直接转调 hermes 自己的 ``web_extract_tool``（同样经 extract_backend=serper 走 jina），
schema 与 web_extract 一致，模型两种叫法都命中。

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

from plugins.web.serper.provider import SerperWebProvider

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


def register(ctx) -> None:
    """Register the Serper+Jina provider + web_fetch alias with the plugin context."""
    ctx.register_web_search_provider(SerperWebProvider())

    # web_fetch 别名 → 转调 hermes 自带的 web_extract_tool（经 extract_backend=serper 走 jina）。
    # override=False：web_fetch 是新名字，不与内置工具冲突。失败不阻断插件其余注册。
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
            is_async=True,       # web_extract_tool 是 async，与 web_extract 注册一致
            emoji="📄",
            description="Fetch web page content (alias of web_extract).",
        )
    except Exception as exc:  # noqa: BLE001 — 别名失败不该拖垮 provider 注册
        import logging

        logging.getLogger(__name__).warning("web_fetch alias 注册失败（忽略）: %s", exc)
