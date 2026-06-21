# Agentic CL custom sandbox image (Tencent Agent Runtime code-interpreter extension)
#
# Role (see doc/Sandbox_Agent架构.md):
#   - In-sandbox AGENT harness = OpenClaw (executes actions: bash/read/write/edit).
#   - Inference (Qwen3.6-27B via vLLM) lives OUTSIDE the sandbox; OpenClaw calls it
#     over an OpenAI-compatible endpoint passed via runtime env (NOT baked here).
#   - Filesystem "user machine" is materialized at INSTANCE START from a baked
#     persona corpus (fs-seeds/), never at build time -> 母版 stays bit-identical.
#
# Official guide:
#   https://cloud.tencent.com/document/product/1814/129691
#
# Rules (snapshot / pause-resume compatible — doc §快照启动约束, enforced by
# scripts/validate_sandbox_dockerfile.sh):
#   - Do NOT set USER (must stay root)
#   - Do NOT set WORKDIR (must stay /)
#   - Do NOT set ENV in Dockerfile (pass via API Env at instance creation)
#   - Do NOT override ENTRYPOINT (keep /init from base image)
#   - Only add dependencies via RUN/COPY; CMD/ENTRYPOINT inherit from base
#
# Build (after docker login to Tencent CCR and pull base):
#   bash scripts/build_sandbox_image.sh
#
# Platform: linux/amd64 only (required by Agent Runtime).
# NOTE: base image is assumed Debian/Ubuntu (apt). If the AGS code-interpreter
# base ships a different package manager, adjust the apt blocks accordingly.

ARG SANDBOX_BASE_IMAGE=ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest
FROM ${SANDBOX_BASE_IMAGE}

# OpenClaw runtime: Node 24 recommended (>=22.19 supported). Pin OpenClaw for
# reproducible母版; override at build time with --build-arg.
ARG NODE_MAJOR=24
ARG OPENCLAW_VERSION=latest

# 1. System tooling for agent ACTIONS (shell/file ops) + office/doc tasks.
#    No model/GPU/inference deps — those live outside the sandbox.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg \
        git jq ripgrep unzip zip less procps \
        poppler-utils pandoc fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 2. Node.js (OpenClaw is a TypeScript/Node project).
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npm --version

# 3. OpenClaw agent harness (the in-sandbox action executor).
RUN npm install -g "openclaw@${OPENCLAW_VERSION}" \
    && npm cache clean --force \
    && (openclaw --version || true)

# 4. Project Python rollout extras (edit requirements.txt as tasks evolve).
COPY requirements.txt /tmp/agentic-cl-sandbox-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agentic-cl-sandbox-requirements.txt \
    && rm -f /tmp/agentic-cl-sandbox-requirements.txt

# 5. Bake agent assets + user-filesystem persona corpus.
#    Materialization runs at instance start (bin/seed_workspace.sh), NOT here,
#    so every instance derived from this template is bit-identical (母版约束).
COPY bin/ /opt/agentic-cl/bin/
COPY openclaw.config.template.json /opt/agentic-cl/openclaw.config.template.json
COPY fs-seeds/ /opt/agentic-cl/fs-seeds/
RUN chmod +x /opt/agentic-cl/bin/*.sh

# 6. (OPTIONAL) Browser tool for ClawEval web tasks. The text-only 195-task
#    subset does NOT need this; it is heavy (~Chromium). Uncomment if needed.
# RUN npx -y playwright install --with-deps chromium
