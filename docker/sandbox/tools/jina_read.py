#!/usr/bin/env python3
"""Jina Reader tool — fetch URL content as Markdown.

Usage:
    python3 /opt/tools/jina_read.py "https://example.com"
    echo '{"url":"https://example.com"}' | python3 /opt/tools/jina_read.py

Env: JINA_API_KEY (required)
"""

import json, os, sys
import requests


def jina_read(url: str, api_key: str = "") -> dict:
    """Fetch a URL through Jina Reader, returning Markdown."""
    key = api_key or os.environ.get("JINA_API_KEY", "")
    if not key:
        return {"error": "JINA_API_KEY not set"}
    try:
        r = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        r.raise_for_status()
        return {
            "url": url,
            "status_code": r.status_code,
            "content_length": len(r.text),
            "content": r.text,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        try:
            data = json.load(sys.stdin)
            url = data.get("url", "")
        except Exception:
            url = sys.stdin.read().strip()
    if not url:
        print('{"error":"no url"}')
        sys.exit(1)
    result = jina_read(url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
