"""Serper + Jina web provider plugin — bundled into the sandbox image.

把项目已有的两个 web 能力（google.serper.dev 搜索 + r.jina.ai 抓取）包成一个
hermes web provider，让模型能直接 call ``web_search`` / ``web_extract``（原来这两个
函数因为没配 backend provider 而 undefined，模型只能靠 terminal 调 /opt/tools 裸脚本）。

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


def register(ctx) -> None:
    """Register the Serper+Jina provider with the plugin context."""
    ctx.register_web_search_provider(SerperWebProvider())
