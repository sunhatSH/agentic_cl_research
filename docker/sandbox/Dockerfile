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
# Build (after docker login to Tencent TCR and pull base):
#   bash scripts/build_sandbox_image.sh
#
# Platform: linux/amd64 only (required by Agent Runtime).
# NOTE: base image is assumed Debian/Ubuntu (apt). If the AGS code-interpreter
# base ships a different package manager, adjust the apt blocks accordingly.

ARG SANDBOX_BASE_IMAGE=tcr-rl.tencentcloudcr.com/agentos-cl-namespace/sandbox-code:latest
FROM ${SANDBOX_BASE_IMAGE}

# OpenClaw runtime: Node 24 recommended (>=22.19 supported). Pin OpenClaw for
# reproducible母版; override at build time with --build-arg.
ARG NODE_MAJOR=24
ARG OPENCLAW_VERSION=latest

# Hermes Agent runtime: Python 3.11 required (pyproject requires-python >=3.11,<3.14).
# Pinned to a release tag (matches the verl-hermes-remote-agent image family).
# Override at build time with --build-arg. See doc/Agent轨迹_Schema.md §0:
# training uses Hermes (spawn synchronous, child-trace separated), not OpenClaw.
# ★ 版本锁定（2026-08-17，勿升级）：v2026.7.7.2 的 `hermes chat -q` session 管理变化，
#   冷启动采集（sandbox_grpo_collect._hermes_chat）报 "Session not found"（242 条全废）。
#   v2026.6.5 是采集代码兼容的版本，升级需先同步改 _hermes_chat 的 session 调用。
ARG HERMES_VERSION=v2026.6.5

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

# 3a. OpenClaw agent harness (the in-sandbox action executor).
RUN npm install -g "openclaw@${OPENCLAW_VERSION}" \
    && npm cache clean --force \
    && (openclaw --version || true)

# 3b. Hermes Agent (the training-time agent framework — see doc/Agent轨迹_Schema.md §0).
#     Installed editable from a shallow clone of the NousResearch repo pinned to a
#     release tag (same recipe as verl-hermes-remote-agent, docker/verl-hermes-remote-agent
#     /Dockerfile.tencent-ags). --break-system-packages because the AGS base image uses
#     system Python (no venv); pip -e installs hermes_agent + its pinned deps into the
#     shared site-packages so `hermes` is on PATH at instance start.
#     Mirrors setup-hermes.sh but without venv isolation (snapshot images run as root).
RUN git clone --depth 1 --branch "${HERMES_VERSION}" \
        https://github.com/NousResearch/hermes-agent.git \
        /opt/hermes-agent \
    && pip install --no-cache-dir --break-system-packages -e /opt/hermes-agent \
    && (hermes --version || hermes --help >/dev/null 2>&1 || true)

# 4. Project Python rollout extras (edit requirements.txt as tasks evolve).
COPY requirements.txt /tmp/agentic-cl-sandbox-requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/agentic-cl-sandbox-requirements.txt \
    && rm -f /tmp/agentic-cl-sandbox-requirements.txt

# 5. Bake agent assets + user-filesystem persona corpus.
#    Materialization runs at instance start (bin/seed_workspace.sh), NOT here,
#    so every instance derived from this template is bit-identical (母版约束).
COPY bin/ /opt/agentic-cl/bin/
COPY openclaw.config.template.json /opt/agentic-cl/openclaw.config.template.json
COPY fs-seeds/ /opt/agentic-cl/fs-seeds/
RUN chmod +x /opt/agentic-cl/bin/*.sh

# 5b. Search tools (Serper + Jina) — callable scripts for hermes web search.
COPY tools/ /opt/tools/
RUN chmod +x /opt/tools/*.py

# 5c. Hermes web provider plugin (Serper 搜索 + Jina 抓取)：把 5b 的 web 能力包成
#     hermes 原生 web provider，让模型能直接 call web_search / web_extract（原来这两个
#     函数因无 backend provider 而 undefined，模型只能靠 terminal 调 5b 裸脚本）。
#     铺到 ~/.hermes/plugins/ 后由 configs/exps/hermes.config.yaml 的
#     plugins.enabled=[web-serper] + web.search_backend/extract_backend=serper 启用。
#     ★ HermesHarness.setup() 只 mkdir + 写 config.yaml/.env，不清空 .hermes，故插件保留。
#     ★★ COPY 会以 root 创建 /home/user/.hermes（Docker 建中间目录默认 root:root），但 hermes
#        以 user(uid1000) 运行，启动要写 ~/.hermes/cron 等 → Permission denied 每 session 崩
#        (2026-08-28 回归)。旧镜像 .hermes 是运行时 HermesHarness 以 user 身份 mkdir 建的(user
#        拥有)。故 COPY 后必须把整个 .hermes chown 回 user，恢复 user 所有权。
COPY hermes_plugins/ /home/user/.hermes/plugins/
RUN chown -R 1000:1000 /home/user/.hermes

# 6. (OPTIONAL) Browser tool for ClawEval web tasks. The text-only 195-task
#    subset does NOT need this; it is heavy (~Chromium). Uncomment if needed.
# RUN npx -y playwright install --with-deps chromium
