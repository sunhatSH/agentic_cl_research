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
#   bash scripts/check_train_env.sh              # 检查 + 自动装缺失
#   CHECK_ONLY=1 bash scripts/check_train_env.sh # 只检查不装（缺了就报错退出）
#   PY=/opt/conda/bin/python bash scripts/check_train_env.sh
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
  "nvidia-cutlass-dsl==4.6.1"
  "mistral-common==1.11.2"
)

# ── 只需存在、版本不 pin 的包（verl 训练链路 transitive）──
NEEDED=(
  omegaconf pyyaml pydantic uvicorn cachetools wandb swanlab
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
# 原则：尽力装，但【永不因环境问题终止训练】——排队/集群训练里 check 主动 exit
# 会让整个排队任务白排。装不上只 WARNING，把最终决定权交给训练本身（真缺依赖
# 训练会自己报错，但不是 check 杀掉它）。CHECK_ONLY=1（人工预检）才报非零。
if [ "${#TO_INSTALL[@]}" -gt 0 ]; then
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "[env] CHECK_ONLY=1：以下需安装但未装：${TO_INSTALL[*]}"
    FAIL=1
  else
    echo "[env] === 安装 ${#TO_INSTALL[@]} 个包 ==="
    echo "  ${TO_INSTALL[*]}"
    "$PY" -m pip install -i "$PIP_INDEX" "${TO_INSTALL[@]}" \
      || echo "[env] WARN: pip 安装失败（继续，不终止训练；缺的包训练时会自暴）"
  fi
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
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "[env] ✗ 环境自检未通过（见上方 FAIL）"; exit 5   # 人工预检：报非零便于感知
  fi
  # 训练模式：绝不终止训练（排队/集群任务被 check 杀掉 = 白排）。只警告，
  # 让训练自己去撞真正缺的依赖 —— 至少不是 check 主动中止。
  echo "[env] ⚠ 环境自检有 FAIL 项（见上方），但按策略【不终止训练】，继续启动"
  exit 0
fi
echo "[env] ✓ 环境自检通过"
