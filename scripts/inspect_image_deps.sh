#!/usr/bin/env bash
# 查当前环境的关键依赖版本 —— 用于对比"能跑视觉的镜像" vs "当前环境",定位 cutlass/sgl_kernel 版本差异。
#
# 背景(§51):开视觉时 lightllm 起 visual 进程 import ViT flash-attn →
#   sgl_kernel → flash_attn_origin.cute → cutlass,报
#   `AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'`
#   → 环境里 cutlass/sgl_kernel/flash_attn_origin 版本不配套。纯文本不碰 ViT 所以能跑。
#
# 用法:
#   bash scripts/inspect_image_deps.sh                 # 用默认 python 查当前环境
#   PY=/path/to/python bash scripts/inspect_image_deps.sh   # 指定 python
#   # 在目标镜像(能跑视觉的那个)里同样跑一遍,对比两份输出。
#
# 输出:同时打到屏幕 + 存到 doc/debug/deps_<host>_<时间>.txt 便于对比。

set -uo pipefail
# ★ 与训练环境完全一致(见 _train_impl.sh):
#   - 解释器 PY = /opt/conda/bin/python(VENV=/opt/conda 的默认;你没传 --venv,一直用这个,
#     python 3.11)。cutlass/sgl_kernel 是 pip 装在 /opt/conda 的 site-packages。
#   - PYTHONPATH = LightLLM:verl:项目根(训练手动指定的,让 import 走 afs 源码而非 site-packages)。
#     查 cutlass/sgl_kernel 版本不需要它(在 site-packages),但带上=与训练一致,顺带能查 lightllm。
#   别用 afs 的 miniconda3(纯净 3.13,无训练依赖,查不到)。PY=... / PYTHONPATH=... 可覆盖。
PY="${PY:-/opt/conda/bin/python}"
LIGHTLLM_DIR="${LIGHTLLM_DIR:-/mnt/afs_toolcall/sunhao4/workspace/LightLLM}"
VERL_DIR="${VERL_DIR:-/mnt/afs_toolcall/sunhao4/dependencies/verl}"
ROOT="/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT:${PYTHONPATH:-}"
OUT_DIR="$ROOT/doc/debug"
mkdir -p "$OUT_DIR"
_stamp="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo now)"
_host="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)"
OUT="$OUT_DIR/deps_${_host}_${_stamp}.txt"

{
echo "################ 依赖版本快照 ################"
echo "host: $(hostname 2>/dev/null)"
echo "time: $_stamp"
echo "python: $PY"
"$PY" -c 'import sys; print("python version:", sys.version.split()[0])' 2>&1
echo

echo "======== 1. 关键包版本(pip show)========"
"$PY" - <<'PYEOF'
import importlib.metadata as md
# 关键嫌疑包 + 相关
pkgs = [
    "nvidia-cutlass-dsl", "cutlass", "nvidia-cutlass",
    "sgl-kernel", "sgl_kernel",
    "flash-attn-origin", "flash_attn_origin", "flash-attn", "flash_attn",
    "flashinfer", "flashinfer-python",
    "torch", "triton", "transformers", "lightllm", "vllm", "sglang",
    "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12",
]
for p in pkgs:
    try:
        v = md.version(p)
        print(f"  {p:32s} = {v}")
    except md.PackageNotFoundError:
        pass  # 没装的跳过,保持输出干净
PYEOF
echo

echo "======== 2. cutlass RoundingModeKind 是否存在(决定性判据)========"
"$PY" - <<'PYEOF'
# 崩溃根因:flash_attn_origin.cute.utils 调 cutlass._mlir.dialects.nvvm.RoundingModeKind
try:
    from cutlass._mlir.dialects import nvvm
    has = hasattr(nvvm, "RoundingModeKind")
    print(f"  cutlass._mlir.dialects.nvvm.RoundingModeKind: {'✅ 存在(视觉可跑)' if has else '❌ 缺失(视觉会崩)'}")
    # 顺带列 nvvm 里带 Rounding 的属性,便于看是不是改名了
    rk = [a for a in dir(nvvm) if 'round' in a.lower() or 'Rounding' in a]
    print(f"  nvvm 里 Rounding 相关属性: {rk}")
except Exception as e:
    print(f"  import cutlass._mlir.dialects.nvvm 失败: {type(e).__name__}: {e}")
try:
    import cutlass
    print(f"  cutlass.__version__ = {getattr(cutlass,'__version__','?')}")
    print(f"  cutlass path = {getattr(cutlass,'__file__','?')}")
except Exception as e:
    print(f"  import cutlass 失败: {type(e).__name__}: {e}")
PYEOF
echo

echo "======== 3. ViT flash-attn import 能否成功(直接试崩溃点)========"
"$PY" - <<'PYEOF'
# 直接复现 §51 的 import 链:sgl_kernel.flash_attn
try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func
    print("  from sgl_kernel.flash_attn import flash_attn_varlen_func: ✅ 成功")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")
PYEOF
echo

echo "======== 4. sgl_kernel 里 causal_conv1d(GDN 用,纯文本也依赖)========"
"$PY" - <<'PYEOF'
try:
    from sgl_kernel import causal_conv1d_fwd
    print("  from sgl_kernel import causal_conv1d_fwd: ✅ 成功")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")
PYEOF
echo

echo "======== 5. 全量 pip freeze(存档,便于 diff)========"
"$PY" -m pip freeze 2>/dev/null
} 2>&1 | tee "$OUT"

echo
echo "[inspect] 已保存: $OUT"
echo "[inspect] 对比方法: 在能跑视觉的镜像里同样跑本脚本,diff 两份 deps_*.txt(重点看 cutlass/sgl_kernel/flash_attn_origin)"
