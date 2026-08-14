#!/usr/bin/env bash
# Materialize a "user machine" filesystem into the agent workspace at INSTANCE
# START (not at image build) — see doc/sandbox/Sandbox_Agent架构.md §4.
#
# Determinism contract (CRITICAL): same AGENTIC_CL_PERSONA / AGENTIC_CL_FS_SEED
# => byte-identical workspace. All 8 slots of a GRPO group are launched with the
# SAME env, so they get the SAME starting filesystem (位级一致 / 组内公平).
#
# Persona selection precedence:
#   1. AGENTIC_CL_PERSONA  -- explicit persona id (matches a dir under fs-seeds/)
#   2. AGENTIC_CL_FS_SEED  -- arbitrary string; hashed -> stable pick from manifest
#   3. (neither)           -- no seeding; empty workspace
#
# Idempotent: re-running with the same persona is a no-op-ish refresh.

set -euo pipefail

SEED_ROOT="${AGENTIC_CL_FS_SEEDS_DIR:-/opt/agentic-cl/fs-seeds}"
MANIFEST="${SEED_ROOT}/manifest.json"
WORKSPACE="${AGENTIC_CL_WORKSPACE:-/root/workspace}"
MARKER="${WORKSPACE}/.agentic_cl_persona"

log() { echo "[seed_workspace] $*" >&2; }

mkdir -p "${WORKSPACE}"

# Resolve persona id deterministically.
persona="${AGENTIC_CL_PERSONA:-}"

if [[ -z "${persona}" && -n "${AGENTIC_CL_FS_SEED:-}" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    log "no manifest at ${MANIFEST}; cannot resolve FS_SEED"
    exit 0
  fi
  # List persona ids (one per line), pick by stable hash(seed) % N.
  mapfile -t personas < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
for p in m.get("personas", []):
    print(p["id"])
' "${MANIFEST}")
  n="${#personas[@]}"
  if [[ "${n}" -eq 0 ]]; then
    log "manifest has no personas; skipping"
    exit 0
  fi
  # Stable, language-agnostic hash via sha256 of the seed string.
  h="$(printf '%s' "${AGENTIC_CL_FS_SEED}" | sha256sum | cut -c1-8)"
  idx=$(( 16#${h} % n ))
  persona="${personas[${idx}]}"
  log "FS_SEED='${AGENTIC_CL_FS_SEED}' -> persona '${persona}' (idx ${idx}/${n})"
fi

if [[ -z "${persona}" ]]; then
  log "no persona / seed provided; leaving workspace empty"
  exit 0
fi

src="${SEED_ROOT}/${persona}"
if [[ ! -d "${src}" ]]; then
  log "ERROR: persona dir not found: ${src}"
  exit 1
fi

# cp -a preserves mode/mtime so the materialized tree is reproducible across slots.
log "materializing persona '${persona}' -> ${WORKSPACE}"
cp -a "${src}/." "${WORKSPACE}/"
printf '%s\n' "${persona}" > "${MARKER}"
log "done"
