# Qwen3.6-27B (qwen3_5 GDN 混合架构) 推理/训练镜像 —— vllm 0.19 + 全部 fix
#
# 背景：商汤 64 卡环境初始镜像 vllm 仅 0.13，不支持 qwen3_5 架构（缺 qwen3_5.py），
# 加载 Qwen3.6 直接报 architecture not supported。本镜像把「已验证能跑 Qwen3.6 的
# vllm 0.19 + 协同依赖 + GDN 死锁 fix 所需的版本组合」固化下来，供新环境开箱即用。
#
# 策略「只装运行时」：本镜像只固化系统 + CUDA runtime + Python + vllm 0.19 全栈依赖。
# verl 源码 / 项目代码 / 模型权重都走共享盘 /mnt/afs_toolcall 挂载，改代码不重打镜像。
#
# 为何 runtime 而非 devel：vllm 0.19 / torch 都是预编译 wheel（自带 .so），torch 自带
# cuda runtime lib，当前可跑通的环境本身就没有 nvcc —— 故 runtime 级 CUDA 镜像足够，
# 省数 GB。唯一权衡：若 vllm 在某路径触发 flashinfer JIT 编译会需要 nvcc；实测 GDN 走
# triton 绕过、flashinfer allreduce 失败可回退，故非硬依赖（见下方 ENV）。
#
# 构建（在带 docker 的机器上，本训练机无 docker daemon）：
#   docker build -t qwen36-vllm019:1.0 -f docker/qwen36-vllm019/Dockerfile .
# 推送（商汤天津 registry）：
#   docker login registry.cn-tj-01.sensecore.cn --username devsfttj-sunhao4
#   docker tag qwen36-vllm019:1.0 registry.cn-tj-01.sensecore.cn/<命名空间>/qwen36-vllm019:1.0
#   docker push registry.cn-tj-01.sensecore.cn/<命名空间>/qwen36-vllm019:1.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

LABEL maintainer="sunhao4" \
      description="vllm 0.19 + Qwen3.6-27B(qwen3_5/GDN) fixes, runtime-only" \
      vllm="0.19.0" torch="2.10.0+cu128" python="3.11"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ---- GDN 死锁 fix 的兜底默认值（核心 fix）----
    # Qwen3.6 GDN 层在 sm90 上走 FlashInfer kernel 会 FileLock 死锁；强制 triton/FLA 路径。
    # 启动 vllm 时仍建议显式带 --gdn-prefill-backend triton；此 env 作为双保险。
    VLLM_GDN_PREFILL_BACKEND=triton

# ---- 系统层 + Python 3.11（Ubuntu 24.04 自带 3.12，显式装 3.11 对齐已验证环境）----
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git tmux \
        openssh-server \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ---- uv（与本地 venv 同一套包管理器，pip 装到 system python）----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---- 核心：vllm 0.19 全栈 + 那组「踩坑修出来的」协同依赖 ----
# 关键：fastapi/starlette/instrumentator 必须是这组协同版本，否则 vllm server 所有
# 请求 500（'_IncludedRouter' object has no attribute 'path'，instrumentator8+starlette1.x 不兼容）。
# vllm 0.19 会自动带 torch 2.10+cu128 / triton 3.6 / flashinfer 0.6.6（均为预编译 wheel）。
RUN uv pip install --system --python /usr/bin/python3.11 \
        "vllm==0.19.0" \
    && uv pip install --system --python /usr/bin/python3.11 \
        "transformers==4.57.6" \
        "fastapi==0.115.6" \
        "starlette==0.41.3" \
        "prometheus-fastapi-instrumentator==7.0.0" \
        "httpx==0.28.1"

# ---- 构建期自检：import + 关键版本 + 协同依赖 + qwen3_5 模型文件在位 ----
# 任一不满足直接 build 失败，避免推出坏镜像。
RUN python3 - <<'PY'
import importlib.metadata as m
need = {
    "vllm": "0.19.0", "fastapi": "0.115.6",
    "starlette": "0.41.3", "prometheus-fastapi-instrumentator": "7.0.0",
}
for p, v in need.items():
    got = m.version(p)
    assert got == v, f"{p} {got} != expected {v}"
import vllm, os, glob
d = os.path.dirname(vllm.__file__)
hits = glob.glob(d + "/model_executor/models/qwen3_5*.py")
assert hits, "qwen3_5 model file MISSING in vllm — cannot load Qwen3.6"
print("[selfcheck] vllm", vllm.__version__, "qwen3_5 OK:", [os.path.basename(x) for x in hits])
PY

WORKDIR /workspace
CMD ["/bin/bash"]
