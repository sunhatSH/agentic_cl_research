#!/usr/bin/env bash
# Load local Tencent / sandbox credentials (gitignored env files only).
#
# Usage:
#   source scripts/env/load_tencent_env.sh
#   bash scripts/sandbox/create_sandbox_via_api.sh builtin

# Resolve repo root. Primary: git. Fallback: from this script's own location (BASH_SOURCE, not $0).
_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")"
_TENCENT="${_ROOT}/docker/sandbox/tencent.env"
_IMAGE="${_ROOT}/docker/sandbox/image.env"
_RUNTIME="${_ROOT}/docker/sandbox/runtime.env"

_load() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$f"
    set +a
    echo "[load_tencent_env] loaded $(basename "$f")"
  else
    echo "[load_tencent_env] skip missing $(basename "$f")"
  fi
}

_load "$_TENCENT"
_load "$_IMAGE"
_load "$_RUNTIME"

if [[ -f "$_RUNTIME" ]]; then
  echo "[load_tencent_env] sandbox runtime.env present (merged at Instance create)"
fi

# Derived RoleArn for custom sandbox Tool (configs/sandbox_tool.json)
if [[ -n "${TENCENT_UIN:-}" && -n "${AGS_ROLE_NAME:-}" ]]; then
  export AGS_ROLE_ARN="qcs::cam::uin/${TENCENT_UIN}:roleName/${AGS_ROLE_NAME}"
  echo "[load_tencent_env] AGS_ROLE_ARN=${AGS_ROLE_ARN}"
fi

# Derived full image tag for custom Tool
if [[ -n "${CCR_REGISTRY:-}" && -n "${CCR_NAMESPACE:-}" && -n "${IMAGE_NAME:-}" && -n "${IMAGE_TAG:-}" ]]; then
  export SANDBOX_IMAGE="${CCR_REGISTRY}/${CCR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"
  echo "[load_tencent_env] SANDBOX_IMAGE=${SANDBOX_IMAGE}"
fi

_missing=0
if [[ -z "${TENCENTCLOUD_SECRET_ID:-}" || -z "${TENCENTCLOUD_SECRET_KEY:-}" ]]; then
  echo "[load_tencent_env] WARN: TENCENTCLOUD_SECRET_ID/KEY empty — edit docker/sandbox/tencent.env"
  _missing=1
fi
if [[ $_missing -eq 0 ]]; then
  echo "[load_tencent_env] API credentials present (region=${TENCENTCLOUD_REGION:-unset})"
fi

# Tencent AGS issues E2B-compatible keys with an `ark_` prefix (not `e2b_`). The
# e2b SDK's validate_api_key hard-requires the `e2b_` prefix (regex \Ae2b_[0-9a-f]+\Z)
# and would raise AuthenticationException on every Sandbox.create. The SDK reads
# E2B_VALIDATE_API_KEY (connection_config.py:81) and skips the prefix check when
# it is "false" — this is the official switch, NOT a monkeypatch. Verified working
# 2026-06-30 on the new-cluster VPC sandbox (instance up 7.8s, run_code=372).
# Trains/rollouts pick this up by sourcing this script before launch.
export E2B_VALIDATE_API_KEY="${E2B_VALIDATE_API_KEY:-false}"
echo "[load_tencent_env] E2B_VALIDATE_API_KEY=${E2B_VALIDATE_API_KEY} (ark_ AGS key → skip e2b_ prefix check)"
