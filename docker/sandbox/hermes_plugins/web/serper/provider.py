"""Serper (Google Search) + Jina Reader web provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

- ``search``  → POST https://google.serper.dev/search  (SERPER_API_KEY)
- ``extract`` → GET  https://r.jina.ai/{url}            (JINA_API_KEY)

复用 docker/sandbox/tools/{serper_search,jina_read}.py 的同一套 REST 调用（同端点、同 key、
同 header），只是包成 provider 契约，让 hermes 的 web_search / web_extract 直接可用。

响应契约（与 WebSearchProvider ABC 文档一致）：
    search:  {"success": True, "data": {"web": [{title,url,description,position}, ...]}}
    extract: [{"url","title","content","raw_content","metadata"}, ...]
    失败:    {"success": False, "error": str}   (search) / 每项带 error 字段 (extract)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"
_JINA_ENDPOINT = "https://r.jina.ai/"
_SEARCH_TIMEOUT = 15
_EXTRACT_TIMEOUT = 30


def _env(key: str) -> str:
    """读 key：先 os.environ、再 ~/.hermes/.env（hermes get_env_value 同口径）。

    ★ HermesHarness 把 SERPER/JINA 写进 ~/.hermes/.env 但不 export 到 os.environ，
    只查 os.environ 会漏 → is_available()=False（工具不可见）+ search/extract 报
    "not set"（2026-08-28 评测 web_search undefined 真根因）。降级回 os.environ。
    """
    try:
        from hermes_cli.config import get_env_value
        return (get_env_value(key) or "").strip()
    except Exception:  # noqa: BLE001 — 离线/无 hermes 时降级
        return os.environ.get(key, "").strip()


def _serper_key() -> str:
    return _env("SERPER_API_KEY")


def _jina_key() -> str:
    return _env("JINA_API_KEY")


class SerperWebProvider(WebSearchProvider):
    """google.serper.dev search + r.jina.ai extract."""

    @property
    def name(self) -> str:
        return "serper"

    @property
    def display_name(self) -> str:
        return "Serper + Jina"

    def is_available(self) -> bool:
        """可用当且仅当至少一个能力的 key 存在（search 需 SERPER，extract 需 JINA）。

        必须是廉价检查、不发网络请求（tool 注册时 / 每次 `hermes tools` 都会调）。
        """
        return bool(_serper_key() or _jina_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # ── search（serper，同步；dispatcher 会在需要时用线程包）───────────────────
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import requests

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001 — interrupt 模块缺失不该拦住搜索
            pass

        key = _serper_key()
        if not key:
            return {"success": False, "error": "SERPER_API_KEY not set"}

        logger.info("Serper search: '%s' (limit=%d)", query, limit)
        try:
            resp = requests.post(
                _SERPER_ENDPOINT,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": query, "num": limit},
                timeout=_SEARCH_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper search error: %s", exc)
            return {"success": False, "error": f"Serper search failed: {exc}"}

        # serper: {"organic": [{"title","link","snippet","position"}, ...]}
        organic = payload.get("organic") or []
        web_results: List[Dict[str, Any]] = []
        for i, item in enumerate(organic):
            web_results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("link") or "",
                    "description": item.get("snippet") or "",
                    "position": item.get("position") or (i + 1),
                }
            )
        logger.info("Serper: found %d search results", len(web_results))
        return {"success": True, "data": {"web": web_results}}

    # ── extract（jina，async；逐 URL 线程 + 超时）─────────────────────────────
    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]
        except Exception:  # noqa: BLE001
            pass

        key = _jina_key()
        if not key:
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": "JINA_API_KEY not set"}
                for u in urls
            ]

        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                content = await asyncio.wait_for(
                    asyncio.to_thread(self._jina_fetch, url, key),
                    timeout=_EXTRACT_TIMEOUT + 5,
                )
                results.append(
                    {
                        "url": url,
                        "title": "",  # jina reader 不单独返回 title，正文首行通常是标题
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url},
                    }
                )
            except asyncio.TimeoutError:
                logger.warning("Jina extract timed out for %s", url)
                results.append(
                    {"url": url, "title": "", "content": "", "raw_content": "",
                     "error": "Jina extract timed out"}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Jina extract error for %s: %s", url, exc)
                results.append(
                    {"url": url, "title": "", "content": "", "raw_content": "",
                     "error": f"Jina extract failed: {exc}"}
                )
        return results

    @staticmethod
    def _jina_fetch(url: str, key: str) -> str:
        """同步抓取单个 URL 的正文（Markdown），供 extract 在线程里调。"""
        import requests

        resp = requests.get(
            f"{_JINA_ENDPOINT}{url}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=_EXTRACT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Serper + Jina",
            "badge": "paid",
            "tag": "Google search (Serper) + URL content extraction (Jina Reader).",
            "env_vars": [
                {
                    "key": "SERPER_API_KEY",
                    "prompt": "Serper API key (Google Search)",
                    "url": "https://serper.dev",
                },
                {
                    "key": "JINA_API_KEY",
                    "prompt": "Jina Reader API key (URL extract)",
                    "url": "https://jina.ai",
                },
            ],
        }
