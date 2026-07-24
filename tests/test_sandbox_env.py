"""Tests for sandbox Instance runtime env loading."""

from __future__ import annotations

import json
from pathlib import Path

from rollout.sandbox_env import load_sandbox_runtime_env


def test_runtime_env_template_has_expected_keys():
    """The runtime env template declares the keys the sandbox Instance needs.

    We only assert the two contract keys are PRESENT in the merged template
    (json + local). We do NOT assert all values are empty: the local
    ``docker/sandbox/runtime.env`` carries real secrets on dev machines, so
    merged values are non-empty there by design.
    """
    env = load_sandbox_runtime_env(include_empty=True)
    assert "AGENTIC_CL_WORKSPACE" in env
    assert "WEB_SEARCH_API_KEY" in env


def test_runtime_env_local_override(tmp_path: Path):
    json_path = tmp_path / "tpl.json"
    json_path.write_text(
        json.dumps({"env": {"AGENTIC_CL_WORKSPACE": "", "WEB_SEARCH_API_KEY": ""}}),
        encoding="utf-8",
    )
    local = tmp_path / "runtime.env"
    local.write_text("AGENTIC_CL_WORKSPACE=/workspace\nWEB_SEARCH_API_KEY=secret\n", encoding="utf-8")

    env = load_sandbox_runtime_env(json_path=json_path, local_path=local)
    assert env == {"AGENTIC_CL_WORKSPACE": "/workspace", "WEB_SEARCH_API_KEY": "secret"}

    empty_filtered = load_sandbox_runtime_env(json_path=json_path, local_path=tmp_path / "missing.env")
    assert empty_filtered == {}
