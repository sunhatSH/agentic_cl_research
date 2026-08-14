#!/usr/bin/env bash
# Run ONE agent turn inside the sandbox (see doc/sandbox/Sandbox_Agent架构.md §3, W1).
#
# Responsibilities:
#   1. Ensure the persona filesystem is materialized (seed_workspace.sh).
#   2. Render OpenClaw config from the template, injecting the EXTERNAL vLLM
#      OpenAI-compatible endpoint from runtime env (NOT baked into the image).
#   3. Invoke OpenClaw headless on the query; the agent loop calls the external
#      model for each decision and executes tool actions in THIS sandbox.
#
# Inference is OUTSIDE the sandbox: OPENAI_API_BASE points at the vLLM server.
#
# Env (injected at instance create via envVars):
#   OPENAI_API_BASE   external vLLM OpenAI endpoint (e.g. http://10.x.x.x:8000/v1)
#   OPENAI_API_KEY    token for that endpoint (dummy ok for local vLLM)
#   OPENCLAW_MODEL    model name the endpoint serves (e.g. qwen3.6-27b)
#   AGENTIC_CL_WORKSPACE  agent working dir (default /root/workspace)
#   AGENTIC_CL_PERSONA / AGENTIC_CL_FS_SEED  filesystem persona (see seed script)
#
# Usage:
#   agent_entry.sh "<query text>"
#   echo "<query text>" | agent_entry.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${AGENTIC_CL_WORKSPACE:-/root/workspace}"
TEMPLATE="${OPENCLAW_CONFIG_TEMPLATE:-/opt/agentic-cl/openclaw.config.template.json}"
CONFIG_DIR="${HOME:-/root}/.openclaw"
CONFIG_OUT="${CONFIG_DIR}/openclaw.json"

log() { echo "[agent_entry] $*" >&2; }

query="${1:-}"
if [[ -z "${query}" && ! -t 0 ]]; then
  query="$(cat)"
fi
if [[ -z "${query}" ]]; then
  log "ERROR: no query provided (arg or stdin)"
  exit 2
fi

# 1. Materialize persona filesystem (idempotent).
bash "${HERE}/seed_workspace.sh"

# 2. Render OpenClaw config from template (envsubst over the known placeholders).
mkdir -p "${CONFIG_DIR}"
: "${OPENAI_API_BASE:?OPENAI_API_BASE must be set (external vLLM endpoint)}"
: "${OPENCLAW_MODEL:?OPENCLAW_MODEL must be set}"
export OPENAI_API_BASE OPENCLAW_MODEL
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"
export AGENTIC_CL_WORKSPACE="${WORKSPACE}"
envsubst < "${TEMPLATE}" > "${CONFIG_OUT}"
log "rendered config -> ${CONFIG_OUT} (endpoint=${OPENAI_API_BASE}, model=${OPENCLAW_MODEL})"

# 3. Run one headless agent turn. Flags depend on the installed OpenClaw version;
#    verify against `openclaw doctor` (see doc/sandbox/Sandbox_Agent架构.md §8).
log "running agent turn in ${WORKSPACE}"
cd "${WORKSPACE}"
openclaw agent --message "${query}"
