#!/usr/bin/env bash
# Load local Tencent / sandbox credentials (gitignored env files only).
#
# Usage:
#   source scripts/load_tencent_env.sh
#   bash scripts/create_sandbox_via_api.sh builtin

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
