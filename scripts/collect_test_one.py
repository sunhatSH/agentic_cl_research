#!/usr/bin/env python3
"""本机单线程采集测试（1 条 query）——验证沙箱 hermes CLI 能否采出 OpenAI 格式轨迹。

背景：recipe_custom 的 HermesHarness 绑定 gateway(sandbox_url/GatewayActor/SSH 隧道 + verl
model server)，本机无 GPU 起不了 gateway，跑不了完整链路。本脚本做一个【等价的本机版】：
手动模拟 HermesHarness 做的事（写 hermes config + 跑 `hermes -z '<prompt>' chat`），但模型
base_url 用外部端点(AGENT_MODEL_BASE)替代 gateway，验证核心——沙箱里 hermes CLI 采集 +
产出的轨迹能否解析成 OpenAI 格式(messages)。

单线程、1 条 query、1 个沙箱。用重建的 agentic-cl-sandbox Tool(hermes v0.18.2 + serper/jina)。

用法（先 source 凭证）：
  source scripts/env/load_tencent_env.sh
  python scripts/collect_test_one.py
"""
import json
import os
import shlex
import sys

sys.path.insert(0, ".")
from rollout.sandbox_client import make_sandbox

TEMPLATE = "agentic-cl-sandbox"


def main():
    model = os.environ.get("AGENT_MODEL_NAME", "openai/gpt-5.5")
    base = os.environ.get("AGENT_MODEL_BASE", "")
    key = os.environ.get("AGENT_MODEL_KEY", "sk-local")
    if not base:
        print("❌ 未设 AGENT_MODEL_BASE(source scripts/env/load_tencent_env.sh)", file=sys.stderr)
        return 1

    # 取一条真实训练 query（从预存文件读，避免主 conda python 缺 pandas；
    # 预存：py310_base python 从 datasets/train.parquet 提一条 user query 写 /tmp/one_query.txt）
    qpath = "/tmp/one_query.txt"
    if not os.path.exists(qpath):
        print(f"❌ 缺 {qpath}。先跑：/mnt/afs_toolcall/sunhao4/miniconda3/envs/py310_base/bin/python3 "
              "-c \"import pandas as pd; df=pd.read_parquet('datasets/train.parquet'); "
              "m=list(df.iloc[0]['prompt']); q=next(x['content'] for x in m if x.get('role')=='user'); "
              "open('/tmp/one_query.txt','w').write(q)\"", file=sys.stderr)
        return 1
    query = open(qpath).read().strip()
    print(f"[query] 用户任务(前200): {query[:200]}\n", flush=True)

    print("[1] 开沙箱(单线程)...", flush=True)
    sb = make_sandbox("e2b", template=TEMPLATE)

    # 模拟 HermesHarness.setup:写 hermes config(v0.18.2 新版 schema,model.provider=custom)
    # base_url 用外部端点(替代 gateway 的 sandbox_url)
    print("[2] 写 hermes config(model.provider=custom, 外部端点)", flush=True)
    cfg = (
        "import subprocess, shlex\n"
        "def run(c):\n"
        "    r=subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30)\n"
        "    return (r.stdout+r.stderr).strip()\n"
        f"B={base!r}; K={key!r}; M={model!r}\n"
        "print(run('hermes config set model.provider custom'))\n"
        "print(run('hermes config set model.base_url '+shlex.quote(B)))\n"
        "print(run('hermes config set model.default '+shlex.quote(M)))\n"
        "print(run('hermes config set model.api_key '+shlex.quote(K)))\n"
    )
    print(sb.run_code(cfg).stdout[:400], flush=True)

    # 模拟 HermesHarness.run:hermes -z '<prompt>' chat(单轮驱动,agent 内部多步)
    print("[3] 跑 hermes agent 采集(单线程 1 条)...", flush=True)
    runcode = (
        "import subprocess, shlex, json\n"
        f"q={query!r}; M={model!r}\n"
        "# -z 传 prompt, chat 子命令; --yolo 免确认; -m 用真实模型名\n"
        "cmd = 'hermes -z '+shlex.quote(q)+' -m '+shlex.quote(M)+' --provider custom chat --yolo > /tmp/h.log 2>&1; echo EXIT=$?'\n"
        "r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=280)\n"
        "print('run:', r.stdout[-200:], r.stderr[-200:])\n"
        # 抓 hermes 的会话记录(它存 ~/.hermes/sessions 或 log)
        "print('=== session files ===')\n"
        "print(subprocess.run('ls -t ~/.hermes/sessions/ 2>/dev/null | head; ls -t ~/.hermes/ 2>/dev/null', shell=True, capture_output=True, text=True).stdout)\n"
        "print('=== log tail ===')\n"
        "print(subprocess.run('tail -30 /tmp/h.log', shell=True, capture_output=True, text=True).stdout)\n"
    )
    res = sb.run_code(runcode)
    print(res.stdout or res.stderr, flush=True)

    print("[4] 尝试抓取 OpenAI 格式轨迹(hermes session 文件)", flush=True)
    grab = (
        "import subprocess, glob, json, os\n"
        # hermes v0.18 会话轨迹通常在 ~/.hermes/sessions/*.json 或 .jsonl
        "cands = subprocess.run('find ~/.hermes -name \"*.json*\" 2>/dev/null | head', shell=True, capture_output=True, text=True).stdout.split()\n"
        "print('候选轨迹文件:', cands[:5])\n"
        "for f in cands[:1]:\n"
        "    c = subprocess.run('cat '+f, shell=True, capture_output=True, text=True).stdout\n"
        "    print('--- '+f+' (前1500) ---'); print(c[:1500])\n"
    )
    print(sb.run_code(grab).stdout, flush=True)

    try:
        sb.kill()
    except Exception:
        pass
    print("\n[done] 沙箱已关", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
