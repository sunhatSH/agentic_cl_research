# 沙盒密钥与镜像配置

**完整操作（控制台 + 命令行 + 概念）：[`doc/Sandbox_腾讯云操作手册.md`](../../doc/sandbox/Sandbox_腾讯云操作手册.md)**

本目录只放密钥和 Docker 构建文件（`*.env` 已 gitignore）。

## 文件速查

| 文件 | 填什么 |
|------|--------|
| `tencent.env` | SecretId/Key、E2B_API_KEY、地域（默认北京 `ap-beijing`）、角色名 |
| `image.env` | CCR 命名空间、镜像 tag（仅 custom） |
| `runtime.env` | 沙箱 **Instance 内**环境变量（密钥填这里；模板见 `runtime.env.example`） |
| `configs/sandbox_runtime_env.json` | Instance 内环境变量键清单（默认全空） |
| `*.example` | 模板，勿改 |

## 快捷操作（一键脚本）

目录 **`ops/`** — 以后沙箱相关命令行操作都放这里：

```bash
bash docker/sandbox/ops/ops.sh query    # 查询 Tool / Instance
```

详见 [`ops/README.md`](ops/README.md)。

## 最快上手

```bash
vim docker/sandbox/tencent.env
source scripts/load_tencent_env.sh
bash scripts/create_sandbox_via_api.sh builtin
```
