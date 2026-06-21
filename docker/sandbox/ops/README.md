# Sandbox 快捷操作（`docker/sandbox/ops/`）

**以后所有沙箱相关一键脚本放这里。**

依赖：`docker/sandbox/tencent.env` 已填 CAM 密钥；自动读地域 `TENCENTCLOUD_REGION`。

## 入口

```bash
# 推荐：统一入口
bash docker/sandbox/ops/ops.sh query

# 或直接
bash docker/sandbox/ops/query.sh
```

## 当前命令

| 脚本 | 作用 |
|------|------|
| `query.sh` | 查询沙箱 Tool 列表 + Instance 数量/状态汇总 + RUNNING 明细 |
| `ops.sh` | 子命令入口（`query` / `status` / `ps` 同义） |

### query 常用参数

```bash
bash docker/sandbox/ops/query.sh --tool agentic-cl-code-interpreter
bash docker/sandbox/ops/query.sh --json
```

## 待添加（占位）

- `create-builtin.sh` — 建 builtin Tool
- `smoke.sh` — 冒烟
- `runtime-env.sh` — 预览 Instance 内 envVars
