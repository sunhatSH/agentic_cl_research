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
    assert VALIDATE.is_file()
    proc = subprocess.run(["bash", str(VALIDATE)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_sandbox_tool_json_has_required_ports():
    import json

    cfg = json.loads((ROOT / "configs" / "sandbox_tool.json").read_text(encoding="utf-8"))
    ports = {p["Port"] for p in cfg["CustomConfiguration"]["Ports"]}
    assert 49999 in ports  # run-code
    assert 49983 in ports  # envd
    assert cfg["ToolType"] == "custom"
