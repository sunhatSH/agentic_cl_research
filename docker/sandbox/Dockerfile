# Agentic CL custom sandbox image (Tencent Agent Runtime code-interpreter extension)
#
# Role (see doc/Sandbox_Agent架构.md):
#   - In-sandbox AGENT harness = OpenClaw (executes actions: bash/read/write/edit).
#   - Inference (Qwen3.6-27B via vLLM) lives OUTSIDE the sandbox; OpenClaw calls it
#     over an OpenAI-compatible endpoint passed via runtime env (NOT baked here).
#   - Filesystem user
