# Agentic CL custom sandbox image (Tencent Agent Runtime code-interpreter extension)
#
# Official guide:
#   https://cloud.tencent.com/document/product/1814/129691
#
# Rules (snapshot / pause-resume compatible — doc §快照启动约束):
#   - Do NOT set USER (must stay root)
#   - Do NOT set WORKDIR (must stay /)
#   - Do NOT set ENV in Dockerfile (pass via API Env at tool creation)
#   - Do NOT override ENTRYPOINT (keep /init from base image)
#   - Only add dependencies via RUN; CMD/ENTRYPOINT inherit from base
#
# Build (after docker login to Tencent CCR and pull base):
#   bash scripts/build_sandbox_image.sh
#
# Platform: linux/amd64 only (required by Agent Runtime).

ARG SANDBOX_BASE_IMAGE=ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest
FROM ${SANDBOX_BASE_IMAGE}

# Install project rollout extras (edit requirements.txt as tasks evolve).
COPY requirements.txt /tmp/agentic-cl-sandbox-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agentic-cl-sandbox-requirements.txt \
    && rm -f /tmp/agentic-cl-sandbox-requirements.txt
