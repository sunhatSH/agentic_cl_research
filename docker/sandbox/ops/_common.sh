# Shared helpers for docker/sandbox/ops/*.sh (source, do not execute directly).

_ops__dir="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%N}}")" && pwd)"
OPS_DIR="${_ops__dir}"
REPO_ROOT="$(cd "${OPS_DIR}/../../.." && pwd)"
SANDBOX_DIR="${REPO_ROOT}/docker/sandbox"

ops_load_env() {
  local f
  for f in "${SANDBOX_DIR}/tencent.env" "${SANDBOX_DIR}/image.env" "${SANDBOX_DIR}/runtime.env"; do
    if [[ -f "$f" ]]; then
      set -a
      # shellcheck source=/dev/null
      source "$f"
      set +a
    fi
  done
  : "${TENCENTCLOUD_REGION:=ap-beijing}"
  export TENCENTCLOUD_REGION
}

ops_require_credentials() {
  ops_load_env
  if [[ -z "${TENCENTCLOUD_SECRET_ID:-}" || -z "${TENCENTCLOUD_SECRET_KEY:-}" ]]; then
    echo "[ops] ERROR: fill ${SANDBOX_DIR}/tencent.env (CAM SecretId/SecretKey)" >&2
    exit 1
  fi
}

ops_require_tccli() {
  if ! command -v tccli >/dev/null 2>&1; then
    echo "[ops] installing tccli..."
    pip3 install -q tccli
  fi
}

ops_region() {
  ops_load_env
  echo "${TENCENTCLOUD_REGION:-ap-beijing}"
}
