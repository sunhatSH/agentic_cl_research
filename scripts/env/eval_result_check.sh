#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 评测结果有效性判定（eval_all.sh / eval_model.sh 共用，防跳过逻辑漂移）。
#
# 坑（2026-08-27）：失败的评测 run 也会产出 per_task.json=[]（空 list，2 字节）。
# 旧的"文件存在就跳过"把空失败结果当"已评过" → 永远评不出来。此函数只认【非空】结果。
#
# _eval_result_nonempty <per_task.json>：
#   返回 0（真）当且仅当文件存在、是合法 JSON、且是【非空 list】（>0 条）。
#   文件不存在 / 空文件 / []（空 list）/ 非法 JSON → 返回 1（假），触发重评。
# ─────────────────────────────────────────────────────────────────────────────

_eval_result_nonempty() {
  local f="$1"
  [ -s "$f" ] || return 1   # 不存在或 0 字节 → 假
  # 用 Python 解析（比 jq 稳，环境必有 python）。非空 list → 退出码 0。
  "${PY:-python3}" - "$f" <<'PYEOF' 2>/dev/null
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(data, list) and len(data) > 0 else 1)
PYEOF
}
