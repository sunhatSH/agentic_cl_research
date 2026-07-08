# 冷采集数据管线 (Cold-Start Trajectory Collection)

> 从 taskspecs 到训练 parquet 的完整管线，含前/后清洗、LLM 打桶+人设、多轮采集。
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

## 管线

```
S1  --generate   打桶+人设  taskspecs→seed_query→strip_zw→garbled>0?DROP→LLM(1call)→queries.jsonl
S1b --filter     不可跑     filter_unrunnable.py (3878→2849)
S2  --collect    多轮采集   hermes+observer+questioner, incremental, 32并发
S2b              后清洗     bin/strip_zw | bin/filter_garbled (C++)
S3  --parquet    转parquet  trajectory_to_parquet.py, 98:2切train/val
S4  --warmup     buffer     warmup_buffer.py → buffer.sqlite → train.sh
```

## 关键设计

- **打桶+人设合并**：一次 LLM 调用同时输出 `{bucket, persona_name}`，全覆盖
- **前清洗 threshold=0**：seed_query 有任何脏字符→直接丢弃该任务
- **后清洗 threshold=0.05**：轨迹 garble>5%→DROP
- **会话控制**：Questioner 全权决定结束（满意→end_session），不设预算
- **失败重做**：`P_k = P0·r^k`，P≤0.1 斩杀，每会话独立
- **hermes 上下文**：`--resume` 跨轮续接 + 沙箱文件持久

## 关键文件

| 文件 | 角色 |
|------|------|
| `scripts/run_cold_start.py` | 主入口 (打标+采集) |
| `scripts/sandbox_grpo_collect.py` | 沙箱引擎 |
| `scripts/run_cold_pipeline.sh` | 一键链式 |
| `bin/strip_zw` | C++ ZW 清洗 |
| `bin/filter_garbled` | C++ 脏字符检测+丢弃 |
| `agents/questioner.py` | Questioner + PatienceTracker |
| `agents/observer.py` | Observer (diff-driven) |
| `agents/personas.json` | 42 人设库 |
