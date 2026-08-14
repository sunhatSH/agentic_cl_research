# scripts/plot/ — 训练指标可视化

| 脚本 | 用途 |
|------|------|
| `plot_progress.py` | **主力**：每实验一文件夹，每指标 raw/EMA/SMA 三图 + config.json + 跨实验 compare/。旧图自动折叠 `{exp}_N/`。`python scripts/plot/plot_progress.py` |
| `show_progress.py` | 终端 Unicode 表格看 step metrics |
| `plot_metrics.py` | 单实验 metrics.jsonl → 分组曲线（pandas） |
| `plot_metrics_compare.py` | 多实验叠加对比 |
| `plot_metrics_nopandas.py` | 无 pandas 环境的 fallback |
| `extract_metrics.py` | 从 verl 日志抽扁平 metrics.jsonl |

数据源：`logs/metrics/qwen35_9b_<exp>/metrics.jsonl`（verl FileLogger）。
