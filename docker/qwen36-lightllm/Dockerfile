
ARG BASE_IMAGE=registry.cn-tj-01.sensecore.cn/ccr-devsfttj/verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211
FROM ${BASE_IMAGE}

LABEL maintainer="sunhao4" \
      description="吴泽寰 lightllm verl 基础镜像 + 固化运行时依赖（Qwen3.6-27B 训练）" \
      base="verl cu129 lightllm vllm0.13 torch2.9 fa3 te2.5" \
      route="lightllm-rollout"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- 固化吴泽寰启动脚本里的 10 个运行时 pip install ----
# 与他脚本里的版本严格对齐（这些是他已验证可跑 Qwen3.6 训练的组合）：
#   - flash-linear-attention==0.4.2  : Qwen3.6 GDN/linear-attention 层 kernel（训练硬依赖）
#   - transformers==5.8.0            : 真正认 qwen3_5 架构的 transformers 大版本
#   - TransferQueue==0.1.6 / tensordict / accelerate==1.13.0 : verl 训练链路依赖
#   - e2b==2.24.0                    : sandbox 采集
#   - nvidia-cutlass-dsl>=4.4.2 / mistral-common / rapidfuzz / langchain-openai
# 分三层 RUN，对应他脚本的三组命令，便于逐组定位失败。
RUN python -m pip install -U tensordict
RUN python -m pip install -U \
        "accelerate==1.13.0" \
        "nvidia-cutlass-dsl>=4.4.2" \
        "transformers==5.8.0"
RUN python -m pip install \
        "e2b==2.24.0" \
        "flash-linear-attention==0.4.2" \
        langchain-openai \
        "mistral-common==1.11.2" \
        rapidfuzz \
        "TransferQueue==0.1.6"

# ---- 构建期自检：关键训练依赖 import + 版本对齐 ----
# 任一不满足直接 build 失败，避免推出缺依赖的镜像。
RUN python - <<'PY'
import importlib.metadata as m
need = {
    "transformers": "5.8.0",
    "flash-linear-attention": "0.4.2",
    "TransferQueue": "0.1.6",
    "accelerate": "1.13.0",
    "e2b": "2.24.0",
}
for p, v in need.items():
    got = m.version(p)
    assert got == v, f"{p} {got} != expected {v}"
# transformers 5.8 必须认 qwen3_5（基础镜像 vllm0.13 不认，但训练用 lightllm+transformers 加载）
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
assert "qwen3_5" in M, "transformers 不认 qwen3_5 架构 — 无法加载 Qwen3.6"
# flash-linear-attention 可 import（GDN kernel）
import fla  # noqa: F401
print("[selfcheck] transformers", m.version("transformers"), "qwen3_5 OK; fla OK; deps pinned OK")
PY

WORKDIR /workspace
CMD ["/bin/bash"]
