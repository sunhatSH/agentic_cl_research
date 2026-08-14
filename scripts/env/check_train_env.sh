#!/usr/bin/env bash
# 训练前环境自检 + 缺失依赖自动安装。
#
# 在真训练机上跑 —— 校验 /opt/conda 是否具备训练链路的全部依赖，版本不对/缺失
# 的自动 pip 装到正确版本，最后校验关键 import + GPU 可用。任一硬项失败即 exit≠0，
# 让调用方（train.sh / run.sh）在真正启动前就中止，而不是跑到一半才崩。
#
# 依赖清单权威来源：docker/qwen36-lightllm/Dockerfile 的 selfcheck 段（与镜像对齐）
# + 本轮采集/训练实测踩坑（e2b-code-interpreter、cutlass 4.6.1、qwen3_5 架构）。
#
# 用法：
#   bash scripts/env/check_train_env.sh              # 检查 + 自动装缺失
#   CHECK_ONLY=1 bash scripts/env/check_train_env.sh # 只检查不装（缺了就报错退出）
#   PY=/opt/conda/bin/python bash scripts/env/check_train_env.sh
set -uo pipefail

PY="${PY:-/opt/conda/bin/python}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
FAIL=0

command -v "$PY" >/dev/null 2>&1 || { echo "[env] ERROR: python 不存在: $PY"; exit 1; }
echo "[env] python = $PY ($($PY --version 2>&1))"

# ── 版本敏感的 pin 包：(pip名 期望版本)。版本不符或缺失 → 装到期望版本 ──
# 与 Dockerfile selfcheck 的 need{} 一致。
PINNED=(
  "transformers==5.12.0"
  "flash-linear-attention==0.4.2"
  "TransferQueue==0.1.6"
  "accelerate==1.13.0"
  "e2b==2.34.0"
  "e2b-code-interpreter==2.8.1"
  "nvidia-cutlass-dsl==4.3.4"
  "mistral-common==1.11.2"
  # omegaconf 2.3.0 才有 oc.decode resolver（configs/base.yaml:195 logger 靠它把
  # env 里的 "[console,swanlab]" 解析成 list）；旧版 import 得过但解析会挂 → 训练时
  # 只能退回 logger:[console]，丢 swanlab。hydra-core 组装 config 一起 pin 防连带。
  "omegaconf==2.3.0"
  "hydra-core==1.3.2"
)

# ── 只需存在、版本不 pin 的包（verl 训练链路 transitive）──
NEEDED=(
  pyyaml pydantic uvicorn cachetools wandb swanlab
  aiohttp httpx tensordict ray msgpack torchdata protobuf tqdm
)

_ver() { "$PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null; }

echo "[env] === 1. 版本敏感包 ==="
TO_INSTALL=()
for spec in "${PINNED[@]}"; do
  pkg="${spec%%==*}"; want="${spec##*==}"
  got="$(_ver "$pkg")"
  if [ "$got" = "$want" ]; then
    echo "  OK   $pkg $got"
  else
    echo "  DIFF $pkg 实际=${got:-未装} 期望=$want"
    TO_INSTALL+=("$spec")
  fi
done

echo "[env] === 2. 训练链路依赖（仅存在性）==="
for pkg in "${NEEDED[@]}"; do
  # import 名与 pip 名的少数差异
  imp="$pkg"; case "$pkg" in pyyaml) imp=yaml;; protobuf) imp=google.protobuf;; esac
  if "$PY" -c "import $imp" >/dev/null 2>&1; then
    echo "  OK   $pkg"
  else
    echo "  MISS $pkg"
    TO_INSTALL+=("$pkg")
  fi
done

# ── 安装缺失/版本不符 ──
# 逻辑：① 检查（上面）② 不符就装/改版本 ③ 训练。只有【第二步安装失败】才停止
# （快速失败，不带缺依赖硬跑白费算力/排队）。检查发现缺 → 进安装，不算失败。
if [ "${#TO_INSTALL[@]}" -gt 0 ]; then
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "[env] CHECK_ONLY=1：以下需安装但未装：${TO_INSTALL[*]}"
    FAIL=1
  else
    echo "[env] === 安装 ${#TO_INSTALL[@]} 个包 ==="
    echo "  ${TO_INSTALL[*]}"
    "$PY" -m pip install -i "$PIP_INDEX" "${TO_INSTALL[@]}" || {
      echo "[env] ERROR: pip 安装失败 —— 停止（环境修不好，快速失败好过带缺依赖硬跑）" >&2
      exit 4; }
  fi
fi

# ── flashinfer（lightllm rollout sampling_backend=flashinfer 运行时依赖）──
# 单列：pip 名(flashinfer-python) ≠ import 名(flashinfer)，塞进上面 NEEDED 会装错包。
# configs/run/*.yaml 的 rollout.engine_kwargs.lightllm.sampling_backend=flashinfer 靠它；
# 镜像层 Dockerfile 已兜底装，这里再兜一层（跑非我们镜像/旧镜像的机器）。有则跳过，
# 缺则装 flashinfer-python（不 pin 版本，用镜像/base 兼容的最新）。
if "$PY" -c "import flashinfer" >/dev/null 2>&1; then
  echo "  OK   flashinfer $(_ver flashinfer-python)"
elif [ "$CHECK_ONLY" = "1" ]; then
  echo "[env] CHECK_ONLY=1：flashinfer 缺失（sampling_backend=flashinfer 需要）"
  FAIL=1
else
  echo "[env] === 安装 flashinfer-python ==="
  "$PY" -m pip install -i "$PIP_INDEX" flashinfer-python || {
    echo "[env] ERROR: flashinfer-python 安装失败 —— 停止（sampling_backend=flashinfer 起不来）" >&2
    exit 4; }
fi

# ── flash-attn（verl/TE 硬 import flash_attn；真包优先，装不出由 shim 兜底）──
# 与 flashinfer 不同：flash-attn 在 CUDA 13.0 大概率编译失败，故【尽力装、不致命】——
# 装不出时 _train_impl.sh 会前置 flash_attn_shim 兜底(bert_padding 纯 torch + interface 转发
# 真 FA3)，训练照跑。这里只在【真包缺失】时试装一次，成功则运行时优先用真包(shim 让位)。
# 探测排除 shim 自身(shim 也叫 flash_attn),确保测的是 site-packages 里的真包。
if "$PY" -c "import flash_attn; assert 'flash_attn_shim' not in (getattr(flash_attn,'__file__','') or '')" >/dev/null 2>&1; then
  echo "  OK   flash_attn (真包) $(_ver flash-attn)"
elif [ "$CHECK_ONLY" = "1" ]; then
  echo "  INFO flash_attn 真包缺失（非致命，运行时 shim 兜底）"
else
  echo "[env] === 尝试安装 flash-attn（CUDA13 可能编译失败，失败不致命，shim 兜底）==="
  "$PY" -m pip install flash-attn --no-build-isolation 2>&1 | grep -E "Successfully|error" \
    || echo "  INFO flash-attn 编译失败（CUDA13 预期），运行时由 flash_attn_shim 兜底"
fi

# ── 3. 关键 import 校验（装完必须真能用）──
echo "[env] === 3. 关键能力校验 ==="

# transformers 认 qwen3_5（否则加载不了 Qwen3.5/3.6）
"$PY" -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M; assert 'qwen3_5' in M" \
  && echo "  OK   transformers 认 qwen3_5" \
  || { echo "  FAIL transformers 不认 qwen3_5"; FAIL=1; }

# e2b code-interpreter（冷启动采集）
"$PY" -c "from e2b_code_interpreter import Sandbox" >/dev/null 2>&1 \
  && echo "  OK   e2b_code_interpreter" \
  || { echo "  FAIL e2b_code_interpreter import"; FAIL=1; }

# verl（需 PYTHONPATH 指向 AFS 源码；训练脚本会设，这里若没设则跳过并提示）
if "$PY" -c "import verl" >/dev/null 2>&1; then
  echo "  OK   verl import"
else
  echo "  WARN verl 未 import（PYTHONPATH 未含 workspace/verl？训练脚本内会设，此处仅提示）"
fi

# fla（Qwen3.6 GDN kernel）
"$PY" -c "import fla" >/dev/null 2>&1 \
  && echo "  OK   fla" || { echo "  FAIL fla import"; FAIL=1; }

# flashinfer（lightllm sampling_backend=flashinfer；上面缺则已装，这里确认真能 import）
"$PY" -c "import flashinfer" >/dev/null 2>&1 \
  && echo "  OK   flashinfer" || { echo "  FAIL flashinfer import"; FAIL=1; }

# oc.decode resolver 能把 env 里的 "[console,swanlab]" 解析成 list（configs/base.yaml:195
# 的 logger 靠 oc.env→oc.decode 这层嵌套；旧 omegaconf 会解析失败 → 训练只能退回
# logger:[console] 丢 swanlab）。探针须与 base.yaml:195 同款嵌套（oc.env 先转字符串）。
VERL_LOGGER='[console,swanlab]' "$PY" -c "
from omegaconf import OmegaConf
c = OmegaConf.create({'logger': '\${oc.decode:\${oc.env:VERL_LOGGER,[console,swanlab]}}'})
r = OmegaConf.to_container(c, resolve=True)['logger']
assert r == ['console', 'swanlab'], r
" >/dev/null 2>&1 \
  && echo "  OK   oc.decode 解析 logger list" \
  || { echo "  FAIL oc.decode 无法解析 [console,swanlab]（omegaconf 版本过旧？）"; FAIL=1; }

# ── 4. GPU / torch CUDA ──
echo "[env] === 4. GPU ==="
"$PY" -c "
import torch, sys
if not torch.cuda.is_available():
    print('  FAIL torch.cuda 不可用'); sys.exit(1)
n = torch.cuda.device_count()
print(f'  OK   torch {torch.__version__} | CUDA available | {n} GPU')
" || FAIL=1

if [ "$FAIL" -ne 0 ]; then
  # 装完关键能力仍 FAIL（import 不了 / GPU 不可用）= 环境修不好，停止（快速失败）。
  # 检查阶段发现缺不算 FAIL —— 那些已进上面的安装步；只有装了还不行才到这。
  echo "[env] ✗ 环境自检未通过（见上方 FAIL）—— 停止" >&2
  exit 5
fi
echo "[env] ✓ 环境自检通过"
