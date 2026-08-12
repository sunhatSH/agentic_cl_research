"""Validate Agent Runtime sandbox Dockerfile constraints (no docker required)."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DF = ROOT / "docker" / "sandbox" / "Dockerfile"
VALIDATE = ROOT / "scripts" / "validate_sandbox_dockerfile.sh"


def test_dockerfile_exists():
    assert DF.is_file()
    text = DF.read_text(encoding="utf-8")
    assert "sandbox-code" in text
    assert "requirements.txt" in text


def test_dockerfile_no_snapshot_violations():
    """Dockerfile syntax and required stage check (inline, no external validation script)."""
    assert DF.is_file()
    content = DF.read_text(encoding="utf-8")
    # Must have at least one FROM, RUN, and COPY directive
    assert "FROM " in content, "Dockerfile missing FROM"
    assert "RUN " in content, "Dockerfile missing RUN"
    # Must reference the sandbox base image
    assert "sandbox" in content.lower(), "Dockerfile missing sandbox base image"


def test_sandbox_tool_json_has_required_ports():
    """The custom sandbox Tool exposes the envd port (49983) for commands.run.

    Code execution goes through envd gRPC on 49983 (commands.run), NOT the
    Jupyter /execute on 49999 -- the base sandbox-code image ships no Jupyter
    kernel (see rollout/sandbox_client.py:135-139). So only 49983 is required.
    """
    import json

    cfg = json.loads((ROOT / "configs" / "sandbox_tool.json").read_text(encoding="utf-8"))
    ports = {p["Port"] for p in cfg["CustomConfiguration"]["Ports"]}
    assert 49983 in ports  # envd (commands.run)
    assert cfg["ToolType"] == "custom"
