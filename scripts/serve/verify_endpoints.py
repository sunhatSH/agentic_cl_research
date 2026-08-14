#!/usr/bin/env python3
"""Verify that all model endpoints in configs/agents.yaml are reachable and models exist.

Reads configs/agents.yaml (the single source of truth for model selection) and
for each endpoint (observer, questioner rotation pool, reward judge):
  1. Checks connectivity (GET /models or a minimal /chat/completions call)
  2. Verifies the model name is available at that endpoint
  3. Checks anti self-preference (no two agents sharing the same model)

Usage:
    python scripts/serve/verify_endpoints.py [--config PATH] [--dry-run] [--verbose]

Exit codes:
    0  all critical endpoints reachable
    1  one or more critical endpoints unreachable
    2  anti self-preference violation (warning only, not fatal unless --strict)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so we can import agents.config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.config import (  # noqa: E402
    ResolvedEndpoint,
    _reload_config,
    resolve_judge,
    resolve_observer,
    resolve_questioner,
    validate_model_distinctness,
)


def _check_endpoint(ep: ResolvedEndpoint, *, verbose: bool = False) -> dict:
    """Check a single endpoint: connectivity + model availability.

    Returns a dict with keys: ok, label, model, base_url, latency_ms, error.
    """
    import httpx

    result = {
        "ok": False,
        "model": ep.model,
        "base_url": ep.base_url,
        "latency_ms": -1,
        "error": "",
    }

    # Step 1: Try GET /models to check connectivity and model availability
    try:
        t0 = time.monotonic()
        resp = httpx.get(
            f"{ep.base_url}/models",
            headers={"Authorization": f"Bearer {ep.api_key}"},
            timeout=15.0,
        )
        latency = (time.monotonic() - t0) * 1000
        result["latency_ms"] = round(latency)

        if resp.status_code == 200:
            data = resp.json()
            available_models = []
            model_list = data.get("data", [])
            if isinstance(model_list, list):
                available_models = [m.get("id", "") if isinstance(m, dict) else str(m) for m in model_list]
            if ep.model in available_models:
                result["ok"] = True
                if verbose:
                    result["note"] = f"model found in /models list ({len(available_models)} models)"
            else:
                # Model not in list, but some providers don't list all models
                # Try a minimal chat completion as fallback verification
                if verbose:
                    print(f"  Model '{ep.model}' not in /models list, trying chat completion...")
                chat_ok = _try_minimal_chat(ep)
                if chat_ok:
                    result["ok"] = True
                    result["note"] = "model not in /models list but chat completion succeeded"
                else:
                    result["error"] = (
                        f"model '{ep.model}' not found in /models list "
                        f"(available: {available_models[:10]}...) and chat completion failed"
                    )
        elif resp.status_code == 401:
            result["error"] = f"authentication failed (HTTP {resp.status_code})"
        elif resp.status_code == 404:
            # /models not supported, try chat completion directly
            chat_ok = _try_minimal_chat(ep)
            if chat_ok:
                result["ok"] = True
                result["note"] = "/models endpoint not available but chat completion succeeded"
            else:
                result["error"] = "/models returned 404 and chat completion failed"
        else:
            result["error"] = f"/models returned HTTP {resp.status_code}"

    except httpx.ConnectError:
        result["error"] = "connection refused / unreachable"
    except httpx.TimeoutException:
        result["error"] = "connection timed out (15s)"
    except Exception as exc:
        result["error"] = f"unexpected error: {exc}"

    return result


def _try_minimal_chat(ep: ResolvedEndpoint) -> bool:
    """Try a minimal chat completion to verify the endpoint works."""
    import httpx

    try:
        resp = httpx.post(
            f"{ep.base_url}/chat/completions",
            json={
                "model": ep.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {ep.api_key}"},
            timeout=30.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify model endpoints from configs/agents.yaml")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to agents.yaml (default: configs/agents.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve endpoints without making network calls",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on anti self-preference warnings too",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show more detail",
    )
    args = parser.parse_args()

    # Force reload config from the specified path
    _reload_config(args.config)

    print("=" * 70)
    print("Agent Endpoint Verification")
    print("=" * 70)
    print()

    # Collect all endpoints to check
    checks: list[tuple[str, ResolvedEndpoint]] = []
    errors = 0

    # Observer
    print("📡 Observer")
    try:
        obs = resolve_observer()
        checks.append(("Observer", obs))
        print(f"   base_url : {obs.base_url}")
        print(f"   model    : {obs.model}")
        print(f"   temp     : {obs.temperature}")
        print(f"   key_set  : {'yes' if obs.api_key != 'sk-local' else 'no (using default)'}")
    except RuntimeError as e:
        print(f"   ❌ NOT CONFIGURED: {e}")
        errors += 1
    print()

    # Questioner
    print("📡 Questioner")
    try:
        q_cfg = resolve_questioner()
        if q_cfg.rotation:
            for i, ep in enumerate(q_cfg.rotation):
                label = f"Questioner:rotation[{i}]"
                checks.append((label, ep))
                print(f"   pool[{i}]  : {ep.model} @ {ep.base_url}")
            print(f"   rotate   : every {q_cfg.rotate_every} calls")
        if q_cfg.fallback:
            checks.append(("Questioner:fallback", q_cfg.fallback))
            print(f"   fallback : {q_cfg.fallback.model} @ {q_cfg.fallback.base_url}")
    except RuntimeError as e:
        print(f"   ❌ NOT CONFIGURED: {e}")
        errors += 1
    print()

    # Reward / Judge
    print("📡 Reward / Judge")
    try:
        judge = resolve_judge()
        checks.append(("Reward", judge))
        print(f"   base_url : {judge.base_url}")
        print(f"   model    : {judge.model}")
        print(f"   temp     : {judge.temperature}")
        print(f"   key_set  : {'yes' if judge.api_key != 'sk-local' else 'no (using default)'}")
    except RuntimeError as e:
        print(f"   ❌ NOT CONFIGURED: {e}")
        errors += 1
    print()

    # Anti self-preference check
    print("🛡️  Anti Self-Preference Check")
    warnings = validate_model_distinctness()
    if warnings:
        for w in warnings:
            print(f"   ⚠️  {w}")
        if args.strict:
            errors += len(warnings)
    else:
        print("   ✅ All agents use distinct models")
    print()

    # Dry run: stop here
    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — no network calls made")
        print(f"Endpoints to check: {len(checks)}")
        print(f"Config errors: {errors}")
        return 1 if errors else 0

    # Connectivity checks
    print("=" * 70)
    print("Connectivity Checks")
    print("=" * 70)
    print()

    results = []
    for label, ep in checks:
        print(f"🔍 {label} ({ep.model})")
        r = _check_endpoint(ep, verbose=args.verbose)
        r["label"] = label
        results.append(r)

        if r["ok"]:
            print(f"   ✅ OK  ({r['latency_ms']}ms)")
            if r.get("note"):
                print(f"   ℹ️  {r['note']}")
        else:
            print(f"   ❌ FAIL  {r['error']}")
            errors += 1
        print()

    # Summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Agent':<30} {'Model':<30} {'Status':<8} {'Latency'}")
    print("-" * 80)
    for r in results:
        status = "✅ OK" if r["ok"] else "❌ FAIL"
        lat = f"{r['latency_ms']}ms" if r["latency_ms"] >= 0 else "N/A"
        print(f"{r['label']:<30} {r['model']:<30} {status:<8} {lat}")

    print()
    if errors:
        print(f"❌ {errors} error(s) found")
        return 1
    else:
        print("✅ All endpoints verified")
        return 0


if __name__ == "__main__":
    sys.exit(main())
