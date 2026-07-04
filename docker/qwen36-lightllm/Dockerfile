
ARG BASE_IMAGE=registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211
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

# ---- 补齐 verl 0.8.0 训练链路依赖（base 镜像未必全有，对照 pyproject.toml）----
# 这些是 verl 训练 + 本项目代码硬依赖；泽寰 base 是 verl 训练镜像大概率已带，但
# 显式装一遍兜底，避免训练起不来才发现缺包。版本不 pin（用 base 已装的或最新兼容），
# 与泽寰栈冲突时以 base 为准（base 已装的 pip 会跳过）。
RUN python -m pip install \
        "omegaconf>=2.3" \
        "pyyaml>=6.0" \
        "pydantic>=2.0" \
        "uvicorn>=0.30" \
        "cachetools>=5.0" \
        "wandb>=0.16" \
        "swanlab>=0.3" \
        "aiohttp>=3.9" \
        "httpx>=0.27" \
        "tqdm>=4.66" \
        "numpy<2.0"
# numpy<2.0：verl setup.py / requirements.txt 硬要求 numpy<2.0.0（numpy 2.x 与旧 torch/
# verl 链路不兼容）。泽寰 base 装 verl 时已满足；显式 pin 兜底，防后续包把 numpy 拉到 2.x。

# ---- 构建期自检：关键训练依赖 import + 版本对齐 ----
# 任一不满足直接 build 失败，避免推出缺依赖的镜像。
# 注：verl 本体是定制版，在 /mnt/afs 挂载进容器，不在镜像里——自检跳过 verl 本体 import，
#     只校验它的训练链路依赖（第 4 项）。
RUN python - <<'PY'
import importlib.metadata as m

# --- 1. 泽寰栈：版本严格对齐（他已验证的组合）---
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
print("[selfcheck] 泽寰栈版本对齐 OK:", need)

# --- 2. transformers 5.8 必须认 qwen3_5 架构（训练加载 Qwen3.6 的前提）---
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
assert "qwen3_5" in M, "transformers 不认 qwen3_5 架构 — 无法加载 Qwen3.6"
print("[selfcheck] transformers qwen3_5 OK")

# --- 3. fla 可 import（Qwen3.6 GDN kernel）---
import fla  # noqa: F401
print("[selfcheck] fla import OK")

# --- 4. verl 训练链路依赖（对照 pyproject.toml + 训练代码 import）---
verl_deps = [
    "omegaconf", "yaml",           # 配置继承 (base.yaml)
    "pydantic",                    # verl schema
    "uvicorn", "cachetools",       # verl transitive (ray_trainer/engine_workers)
    "wandb", "swanlab",            # 实验追踪
    "aiohttp", "httpx",            # 网络 (judge / observer / rollout)
    "tensordict", "accelerate",    # verl 训练
    "ray",                         # verl 分布式
]
missing = []
for dep in verl_deps:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)
assert not missing, f"verl 训练链路缺依赖: {missing} — 上方 pip install 段没覆盖到，需补装"
print(f"[selfcheck] verl 训练链路 {len(verl_deps)} 依赖全部可 import OK")

# --- 5. verl 本体：定制版在 /mnt/afs 挂载进容器，不进镜像（build 期不查）---
# 训练时 AFS 盘挂载后 `import verl` 才可用，故镜像自检跳过 verl 本体。
# 上面第 4 项已确认 verl 的训练链路依赖（tensordict/accelerate/ray/...）齐全。

# --- 6. numpy 兼容性确认（不 pin 版本，只确认 import 不报错 + 版本可查）---
import numpy as np
print("[selfcheck] numpy", np.__version__, "OK")

print("[selfcheck] ALL PASS — lightllm 训练镜像就绪")
PY

WORKDIR /workspace
CMD ["/bin/bash"]
