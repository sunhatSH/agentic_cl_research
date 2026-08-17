#!/usr/bin/env bash
# Cold-start 数据质检 pipeline（替换旧 qc_trajectory）。
#
# 输入：hermes_structured 产出的 grpo_hermes.jsonl（OpenAI messages 格式）
# 流程：
#   (1) agent_data_tools validate-openai  → 规则层过滤（字段完整性、异常字符、token 注入等）
#   (2) LLMChecker                         → LLM 7 轮质量标注 → purify 提纯
#
# 用法：
#   bash scripts/pipeline/qc_cold_start.sh <grpo_hermes.jsonl> [output_dir]
#   bash scripts/pipeline/qc_cold_start.sh agentic_cl_rollouts/smoke/trajectory/gpt5/grpo_hermes.jsonl
set -euo pipefail

INPUT="${1:?Usage: $0 <grpo_hermes.jsonl> [output_dir]}"
OUT_DIR="${2:-$(dirname "$INPUT")/qc}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
QC_ROOT="/mnt/afs_toolcall/sunhao4/workspace/quality-check"
PY="${PYTHON:-python3}"
export PYTHONPATH="$QC_ROOT/agent_data_tools/src:$QC_ROOT/LLMChecker:${PYTHONPATH:-}"
CONCURRENCY="${CONCURRENCY:-4}"

DATASET_NAME="$(basename "$INPUT" .jsonl)"
mkdir -p "$OUT_DIR"

echo "=== QC pipeline ==="
echo "  input:       $INPUT"
echo "  dataset:     $DATASET_NAME"
echo "  output_dir:  $OUT_DIR"
echo ""

# ------------------------------------------------------------------ Stage 1: 规则过滤 (agent_data_tools)
echo "=== Stage 1: agent_data_tools validate-openai ==="
# Build meta.json pointing to the input file.
META="$OUT_DIR/meta_qc.json"
python3 -c "
import json
json.dump({'$DATASET_NAME': {'annotation': '$INPUT', 'length': 1}}, open('$META', 'w'))
"
echo "  meta: $META"

"$PY" -m agent_data_tools.cli validate-openai \
    -i "$META" -l "$OUT_DIR/logs_agent_tools" -w "$CONCURRENCY" -q
echo "  stage 1 done."

# ------------------------------------------------------------------ Stage 2: LLM 模型过滤 (LLMChecker)
echo ""
echo "=== Stage 2: LLMChecker ==="
# Runtime creds: reuse tokenhub key from repo .env / runtime.env
source "$REPO/scripts/env/load_tencent_env.sh" 2>/dev/null || true
TOKENHUB_KEY="${TOKENHUB_API_KEY:-}"
if [ -z "$TOKENHUB_KEY" ] && [ -f "$REPO/.env" ]; then
    TOKENHUB_KEY="$(grep TOKENHUB_API_KEY "$REPO/.env" 2>/dev/null | cut -d= -f2-)"
fi

# Build llmchecker config YAML.
LLM_CONFIG="$OUT_DIR/llmchecker_config.yaml"
cat > "$LLM_CONFIG" << YAML
input: "$INPUT"
output_dir: "$OUT_DIR/llmchecker"
data_format: openai
concurrency: $CONCURRENCY
endpoints:
  - base_url: https://tokenhub.sensetime.com/v1
    model: deepseek-v4-pro-202606
    rpm: 30
    tpm: 500000
    keys:
      - key: "$TOKENHUB_KEY"
max_retries: 2
YAML

echo "  config: $LLM_CONFIG"
cd "$QC_ROOT/LLMChecker"
"$PY" -m llmchecker --config "$LLM_CONFIG"
echo "  stage 2 done."

# ------------------------------------------------------------------ Stage 3: 提纯
echo ""
echo "=== Stage 3: purify ==="
"$PY" "$QC_ROOT/LLMChecker/scripts/postprocess/purify.py" \
    --output-dir "$OUT_DIR/llmchecker" \
    --input "$INPUT"
echo "  stage 3 done."

echo ""
echo "=== QC pipeline DONE ==="
echo "  agent_tools log:   $OUT_DIR/logs_agent_tools/"
echo "  llmchecker results: $OUT_DIR/llmchecker/"
echo "  purified:           $OUT_DIR/llmchecker/purified/"
