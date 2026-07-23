
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
#   - transformers==5.12.0           : verl ray_trainer.py import MistralForSequenceClassification
#                                       需要 5.12+（5.8.0 缺该类，verl import 崩）
#   - TransferQueue==0.1.6 / tensordict<=0.10.0 / accelerate==1.13.0 : verl 训练链路依赖
#   - e2b==2.34.0 + e2b-code-interpreter==2.8.1 : sandbox 采集（code-interpreter 需 e2b>=2.26）
#   - nvidia-cutlass-dsl==4.6.1 / mistral-common / rapidfuzz / langchain-openai
# 分三层 RUN，对应他脚本的三组命令，便于逐组定位失败。
RUN python -m pip install -U "tensordict>=0.8.0,<=0.10.0"
RUN python -m pip install -U \
        "accelerate==1.13.0" \
        "nvidia-cutlass-dsl==4.6.1" \
        "transformers==5.12.0"
RUN python -m pip install \
        "e2b==2.34.0" \
        "e2b-code-interpreter==2.8.1" \
        "flash-linear-attention==0.4.2" \
        langchain-openai \
        "mistral-common==1.11.2" \
        rapidfuzz \
        "TransferQueue==0.1.6"
# e2b-code-interpreter: 冷启动沙箱采集必需（rollout/sandbox_client.py 里
#   `from e2b_code_interpreter import Sandbox`）。base 镜像只带 e2b（2.24.0）不带
#   code-interpreter → 新容器起采集会 ModuleNotFoundError。2.8.1 与 e2b 2.24 实测兼容。

# ---- 补齐 verl 0.8.0 训练链路依赖（base 镜像未必全有，对照 pyproject.toml）----
# 这些是 verl 训练 + 本项目代码硬依赖；泽寰 base 是 verl 训练镜像大概率已带，但
# 显式装一遍兜底，避免训练起不来才发现缺包。版本不 pin（用 base 已装的或最新兼容），
# 与泽寰栈冲突时以 base 为准（base 已装的 pip 会跳过）。
#
# 2026-07-15 实机踩坑补齐（本机 py310_base 环境验证通过的完整组合）：
#   - ray>=2.41.0        : verl 分布式必需（base 镜像可能不带）
#   - msgpack            : ray 运行时硬依赖（ray._raylet import 时需要）
#   - torchdata          : verl ray_trainer.py:33 `from torchdata.stateful_dataloader import StatefulDataLoader`
#   - protobuf>=5.0      : swanlab 需要 protobuf.runtime_version（4.x 缺该 API）
#   - pyyaml>=6.0.2      : verl/transformers 版本比较需要 pyyaml>=5.1，6.0.1 的 dist-info 会损坏
#   - omegaconf==2.3.0   : oc.decode resolver（configs/base.yaml:195 靠它把 env 的
#                          "[console,swanlab]" 解析成 list）；2.3 前无此 resolver，会退回
#                          logger:[console] 丢 swanlab。与 scripts/check_train_env.sh 同款 pin。
#   - hydra-core==1.3.2  : verl 组装 config，与 omegaconf 配套 pin。
RUN python -m pip install \
        "omegaconf==2.3.0" \
        "hydra-core==1.3.2" \
        "pyyaml>=6.0.2" \
        "pydantic>=2.0" \
        "uvicorn>=0.30" \
        "cachetools>=5.0" \
        "wandb>=0.16" \
        "swanlab>=0.3" \
        "aiohttp>=3.9" \
        "httpx>=0.27" \
        "tqdm>=4.66" \
        "numpy<2.0" \
        "ray[default]>=2.41.0" \
        "msgpack" \
        "torchdata" \
        "protobuf>=5.0"
# numpy<2.0：verl setup.py / requirements.txt 硬要求 numpy<2.0.0（numpy 2.x 与旧 torch/
# verl 链路不兼容）。泽寰 base 装 verl 时已满足；显式 pin 兜底，防后续包把 numpy 拉到 2.x。

# ---- 构建期自检：关键训练依赖 import + 版本对齐 ----
# 任一不满足直接 build 失败，避免推出缺依赖的镜像。
# 注：verl 本体是定制版，在 /mnt/afs 挂载进容器，不在镜像里——自检跳过 verl 本体 import，
#     只校验它的训练链路依赖（第 4 项）。
RUN python - <<'PY'
import importlib.metadata as m

# --- 1. 泽寰栈：版本严格对齐（与 scripts/check_train_env.sh 的 PINNED 一致）---
need = {
    "transformers": "5.12.0",
    "flash-linear-attention": "0.4.2",
    "TransferQueue": "0.1.6",
    "accelerate": "1.13.0",
    "e2b": "2.34.0",
    "e2b-code-interpreter": "2.8.1",
    "nvidia-cutlass-dsl": "4.6.1",
    "mistral-common": "1.11.2",
    "omegaconf": "2.3.0",       # oc.decode/resolve；旧版 logger list 解析会挂
    "hydra-core": "1.3.2",
}
for p, v in need.items():
    got = m.version(p)
    assert got == v, f"{p} {got} != expected {v}"
print("[selfcheck] 泽寰栈版本对齐 OK:", need)

# --- 2. transformers 5.12 必须认 qwen3_5 架构 + MistralForSequenceClassification ---
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
assert "qwen3_5" in M, "transformers 不认 qwen3_5 架构 — 无法加载 Qwen3.6"
from transformers import MistralForSequenceClassification  # noqa: F401
print("[selfcheck] transformers qwen3_5 + Mistral OK")

# --- 3. fla 可 import（Qwen3.6 GDN kernel）---
import fla  # noqa: F401
print("[selfcheck] fla import OK")

# --- 3b. OmegaConf 能把 logger 字面 list 正常 resolve（复刻 verl_runner 流程）---
# configs/*.yaml 的 trainer.logger 是字面 [console, swanlab]；旧 omegaconf resolve 对
# ListConfig 报 UnsupportedValueType。校验镜像里的 omegaconf 能过这条流程。
from omegaconf import OmegaConf as _OC
_c = _OC.create({"trainer": {"logger": ["console", "swanlab"]}})
_c = _OC.create(_OC.to_container(_c, resolve=False))
_OC.resolve(_c)
assert list(_c.trainer.logger) == ["console", "swanlab"], _c.trainer.logger
print("[selfcheck] omegaconf logger list resolve OK")

# --- 4. verl 训练链路依赖（对照 pyproject.toml + 训练代码 import）---
verl_deps = [
    "omegaconf", "yaml",           # 配置继承 (base.yaml)
    "pydantic",                    # verl schema
    "uvicorn", "cachetools",       # verl transitive (ray_trainer/engine_workers)
    "wandb", "swanlab",            # 实验追踪
    "aiohttp", "httpx",            # 网络 (judge / observer / rollout)
    "tensordict", "accelerate",    # verl 训练
    "ray",                         # verl 分布式
    "msgpack",                     # ray 运行时
    "torchdata",                   # verl ray_trainer StatefulDataLoader
    "google.protobuf",             # swanlab runtime_version
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

# ---- Codex CLI（容器内 debug / 交互）----
# base 镜像无 Node.js，先通过 conda 安装
RUN conda install -y -c conda-forge nodejs=22
RUN npm install -g @openai/codex

# ---- flash_attn (CUDA 13.0 无预编译包，shim 兜底) ----
# PyTorch 2.13 sdpa 内置 unpad_input / pad_input / index_first_axis，
# 但 verl attention_utils.py 硬 import flash_attn.bert_padding。
# 先尝试 pip install flash-attn（编译耗时，可能因 CUDA 13.0 失败），
# 再落 shim 作为 fallback（torch sdpa 实现，同语义、0 依赖、无需编译）。
RUN python -m pip install flash-attn --no-build-isolation 2>&1 | grep -E "Successfully|error" || echo "flash-attn install failed (expected on CUDA 13.0), shim will cover"
COPY docker/qwen36-lightllm/flash_attn_shim/ /opt/flash_attn_shim/
ENV PYTHONPATH="/opt/flash_attn_shim:${PYTHONPATH}"

# ---- vllm 文件权限修复（base 镜像个别 .py 为 640 root:root，非 root 不可读）----
# routed_experts_capturer.py / qwen3_moe_mtp.py 出厂 o-x，import vllm 直接 PermissionError。
# 正式训练走 lightllm 不用 vllm，但开发/调试 import 时会崩，补 o+r 兜底。
RUN find /opt/conda/lib/python*/site-packages/vllm -name '*.py' ! -perm -o+r -exec chmod o+r {} + \
    && echo "[docker] vllm permissions fixed"


WORKDIR /workspace
CMD ["/bin/bash"]
