# 冷采集数据管线 (Cold-Start Trajectory Collection)

> 从 taskspecs 到训练 parquet 的完整管线，含前/后清洗、LLM 打桶、人设预选、多轮采集。
> 权威文档：`doc/ops/冷采集数据管线.md`、`doc/ops/数据清洗.md`

## 一键启动

```bash
# 全量（打标+采集+parquet+warmup）
bash scripts/run_cold_pipeline.sh --all-collect

# 只采集（queries 已就绪）
bash scripts/run_cold_pipeline.sh --collect

# 只打标不采集
.venv/bin/python scripts/run_cold_start.py --generate --no-collect --classify-workers 32
```

## 四阶段

1. **打桶+人设**：task_family 静态映射 6 种 + LLM classify (sufy, 32并发) + LLM 选人设 (42选1)
2. **多轮采集**：hermes in sandbox + observer (沙箱diff) + questioner (人设追问)，无 reward/winner，incremental 增量
3. **后清洗**：`bin/strip_zw | bin/filter_garbled` (C++) — ZW 去字符 + garble>5% DROP
4. **parquet + warmup**：98:2 切 train/val，预热 buffer

## 清洗

| 位置 | 工具 | 功能 |
|------|------|------|
| 前（打标时） | Python `strip_zw` + `analyze_text` | seed_query 有脏字符→直接 DROP |
| 后（采集后） | C++ `bin/strip_zw` | 去 U+2060/U+FEFF/U+00AD |
| 后（采集后） | C++ `bin/filter_garbled` | garble>5%→DROP 轨迹 |

## 多轮架构

- **actor**：沙箱内 hermes (`--resume` 跨轮续接), sufy gpt-5
- **observer**：沙箱 before/after diff (ground truth), 不看 actor 轨迹
- **questioner**：人设驱动 (42 persona), 模型轮换池 (claude-sonnet/deepseek/qwen/kimi)
- **patience**：失败重做 `P_k = P0·r^k`, P≤0.1 斩杀, 每会话独立

## 关键文件

| 文件 | 角色 |
|------|------|
| `scripts/run_cold_start.py` | 主入口 (打标 + 采集) |
| `scripts/sandbox_grpo_collect.py` | 沙箱引擎 (hermes + observer + questioner) |
| `scripts/run_cold_pipeline.sh` | 一键链式脚本 |
| `bin/strip_zw` | C++ ZW 清洗 |
| `bin/filter_garbled` | C++ 脏字符检测+丢弃 |
| `agents/questioner.py` | Questioner + PatienceTracker |
| `agents/observer.py` | Observer (diff-driven) |
| `agents/personas.json` | 42 人设库 (P0/r/profile) |
