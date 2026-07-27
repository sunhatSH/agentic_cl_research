#!/usr/bin/env bash
# Print merged sandbox Instance envVars as JSON (non-empty values only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 -c "
import json, sys
sys.path.insert(0, '${ROOT}')
from rollout.sandbox_env import load_sandbox_runtime_env
print(json.dumps(load_sandbox_runtime_env(), ensure_ascii=False))
"
