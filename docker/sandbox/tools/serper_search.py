#!/usr/bin/env python3
"""Serper Google Search tool — callable from shell or as Python module.

Usage:
    python3 /opt/tools/serper_search.py "query"
    echo '{"q":"query","num":5}' | python3 /opt/tools/serper_search.py

Env: SERPER_API_KEY (required)
"""

import json, os, sys
import requests


def serper_search(query: str, num: int = 10, api_key: str = "") -> dict:
    """Call google.serper.dev/search and return parsed JSON."""
    key = api_key or os.environ.get("SERPER_API_KEY", "")
    if not key:
        return {"error": "SERPER_API_KEY not set"}
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            data = json.load(sys.stdin)
            query = data["q"]
        except Exception:
            query = sys.stdin.read().strip()
    if not query:
        print('{"error":"no query"}')
        sys.exit(1)
    result = serper_search(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
