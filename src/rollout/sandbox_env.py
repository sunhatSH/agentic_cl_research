"""Load environment variables injected into Tencent sandbox Instances at create time."""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # src/rollout → repo root
_DEFAULT_JSON = _REPO_ROOT / "configs" / "sandbox_runtime_env.json"
_DEFAULT_LOCAL = _REPO_ROOT / "docker" / "sandbox" / "runtime.env"


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip().strip('"').strip("'")
    return out


def load_sandbox_runtime_env(
    *,
    json_path: Path | None = None,
    local_path: Path | None = None,
    include_empty: bool = False,
) -> dict[str, str]:
    """Merge runtime env template + optional local overrides.

    Order: ``configs/sandbox_runtime_env.json`` then ``docker/sandbox/runtime.env``
    (if present). By default drops keys whose final value is empty so CreateSandbox
    only sends filled configuration.
    """
    json_file = json_path or Path(os.environ.get("SANDBOX_RUNTIME_ENV_JSON", _DEFAULT_JSON))
    local_file = local_path or Path(os.environ.get("SANDBOX_RUNTIME_ENV_FILE", _DEFAULT_LOCAL))

    merged: dict[str, str] = {}
    if json_file.is_file():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        env_block = data.get("env", data)
        if isinstance(env_block, dict):
            for key, value in env_block.items():
                if key.startswith("_"):
                    continue
                merged[str(key)] = "" if value is None else str(value)

    if local_file.is_file():
        merged.update(_parse_env_file(local_file))

    if include_empty:
        return merged
    return {k: v for k, v in merged.items() if v != ""}
