#!/usr/bin/env bash
# [DEPRECATED] 请用统一入口：bash scripts/train.sh TOPO --config ...
# 保留向后兼容。等价于 scripts/train.sh TOPO "$@"。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train.sh" 64gpu "$@"
