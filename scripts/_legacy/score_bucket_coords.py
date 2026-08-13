#!/usr/bin/env python3
"""Score 9 capability buckets on 5 dimensions using an LLM judge.

Produces 9 coordinates in capability space. These are used in the
continual-learning replay weighting formula: buckets farther apart in
capability space get higher replay weights (higher forgetting risk).

Dimensions:
  - tool_intensity:     Reliance on tool calling (file I/O, terminal, browser, API).
  - reasoning_depth:    Depth of reasoning chain (multi-hop, synthesis, inference).
  - structure_rigidity: Strictness of format/rule/schema constraints.
  - knowledge_domain:   Domain-specific knowledge required.
  - multi_step:         Number of sequential steps (orchestration, pipeline).

Scores are NOT capped — some dimensions naturally spread wide, others cluster tight.

Usage:
    .venv/bin/python scripts/analysis/score_bucket_coords.py [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

PROMPT = r"""You are designing a capability-space coordinate system for 9 agent skill buckets. Each bucket represents a class of tasks an LLM agent performs.

Below are 5 REAL trajectories per bucket, randomly sampled. Each trajectory shows:
- The user query
- Message count, tool call count, and which tools were used
- Number of multi-turn rounds

Rate each bucket on these 5 dimensions. Do NOT cap scores — some dimensions naturally spread wide (e.g. tool_intensity varies greatly across buckets), others cluster tight. Let scores reflect the true distribution.

Dimensions:
- tool_intensity: Reliance on tool calling (file I/O, terminal, browser, API, MCP).
- reasoning_depth: Depth of reasoning chain (multi-hop, synthesis, inference, analysis).
- structure_rigidity: Strictness of format/rule/schema constraints (compliance, structured output, regulations).
- knowledge_domain: Domain-specific knowledge required (finance, law, engineering vs general).
- multi_step: Number of sequential steps (orchestration, pipeline, multi-stage workflow).

TRAJECTORIES:

workflow:
  [A] query: "根据需求完成业务流程图/状态机图，用draw.io绘制订单状态机、概率算法流程"
      msgs=8  tool_calls=3  tools=[todo, execute_code]  turns=1
  [B] query: "分析需求,总结生成需求测试点文档"
      msgs=80  tool_calls=23  tools=[read_file, clarify, patch, search_files, write_file]  turns=4
  [C] query: "建立运营总指挥的风禾体，制定配置方案，确认workspace/agents/soul/.learnings"
      msgs=36  tool_calls=13  tools=[todo, session_search, search_files, execute_code, skill_view]  turns=2
  [D] query: "制作框架图，对供应链数智化协同创新课题生成立诚框架图"
      msgs=64  tool_calls=27  tools=[todo, read_file, search_files, skills_list, execute_code, terminal]  turns=2
  [E] query: "网关重启好了，通知用户重启成功，继续处理未完成事项"
      msgs=6  tool_calls=2  tools=[session_search]  turns=1

ops:
  [A] query: "检测inkos进程，卡在思考状态；看是否有产出、是否还在工作"
      msgs=102  tool_calls=28  tools=[process, read_file, search_files, execute_code, terminal]  turns=3
  [B] query: "使用MCP工具查询Confluence页面，提取图片url，转换为markdown保存到prd/"
      msgs=26  tool_calls=9  tools=[execute_code, terminal, search_files, read_file]  turns=1
  [C] query: "按TASK_CONTEXT.md逐一修复10个post-gate问题（引用标记/章节/系统树/强制填充），建.stage_complete标记"
      msgs=100  tool_calls=45  tools=[todo, read_file, write_file, patch, search_files, skills_list, execute_code, terminal]  turns=1
  [D] query: "读取HEARTBEAT.md严格照执行，不推断不重复旧任务，没事回HEARTBEAT_OK"
      msgs=780  tool_calls=247  tools=[read_file, search_files, write_file, terminal, execute_code]  turns=7
  [E] query: "读取HEARTBEAT.md(准确路径)，严格照执行"
      msgs=21  tool_calls=6  tools=[terminal, search_files, read_file]  turns=2

qa:
  [A] query: "按系统JSON格式输出决策：龙门大堂中体力充沛但饥饿，向三人求助觅食并探询异象"
      msgs=6  tool_calls=0  tools=[]  turns=2
  [B] query: "查一下所有人的状态"
      msgs=62  tool_calls=16  tools=[clarify, session_search]  turns=4
  [C] query: "Dashboard是什么？什么作用？"
      msgs=2  tool_calls=0  tools=[]  turns=1
  [D] query: "comfyui里llama.cpp能否替代ollama做图片反推，API效果是否更好"
      msgs=6  tool_calls=0  tools=[]  turns=2
  [E] query: "任务执行情况"
      msgs=32  tool_calls=7  tools=[todo, terminal, process]  turns=3

finance:
  [A] query: "对Vietnam Top10选品清单全成本建模(采购/物流/佣金/广告/税费)，输出成本模型和毛利测算"
      msgs=138  tool_calls=54  tools=[execute_code, patch, search_files, read_file]  turns=4
  [B] query: "对Singapore Top10选品清单全成本建模，输出成本模型和毛利测算"
      msgs=70  tool_calls=27  tools=[execute_code, patch, search_files, read_file]  turns=3
  [C] query: "汇总工程数量/变压器容量/高压线路长度/低压线路长度/施工费用"
      msgs=34  tool_calls=10  tools=[clarify, execute_code, read_file]  turns=3
  [D] query: "融资融券数据延迟，尝试其他接口更新市场监控报告"
      msgs=277  tool_calls=106  tools=[todo, read_file, patch, session_search, search_files, execute_code, terminal]  turns=5
  [E] query: "1万USDT/3倍杠杆/无持仓，BB下轨0.9%/成交量1.4倍/RSI 38，分析多空观望，给杠杆仓位"
      msgs=12  tool_calls=3  tools=[execute_code]  turns=2

office:
  [A] query: "3份福袋MHTML整合成结构化Markdown PRD"
      msgs=90  tool_calls=40  tools=[todo, read_file, write_file, patch, search_files, execute_code]  turns=2
  [B] query: "风禾体数据周环比整理成表格"
      msgs=34  tool_calls=9  tools=[read_file, session_search, search_files, execute_code, terminal]  turns=2
  [C] query: "T3工作明细表数据被截断，用Python重新处理"
      msgs=272  tool_calls=117  tools=[read_file, write_file, session_search, search_files, execute_code, terminal]  turns=4
  [D] query: "上传6张25楼IP规划截图，整理进知识库"
      msgs=72  tool_calls=21  tools=[read_file, patch, memory, search_files, write_file, terminal]  turns=2
  [E] query: "结合5-6月工作计划，做6月工作汇报，简洁有创新"
      msgs=372  tool_calls=167  tools=[todo, read_file, write_file, search_files, skills_list, execute_code, terminal]  turns=4

communication:
  [A] query: "给框架图加上emoji图标"
      msgs=124  tool_calls=53  tools=[todo, read_file, patch, search_files, execute_code, terminal]  turns=2
  [B] query: "扮演林黛玉过日常生活，语气软萌害羞"
      msgs=2  tool_calls=0  tools=[]  turns=1
  [C] query: "写《登高》教学设计7部分(教材/学情/目标/重难点/方法/课时/过程)"
      msgs=40  tool_calls=17  tools=[todo, execute_code]  turns=2
  [D] query: "风禾体运营周报，别太AI套话，自然点"
      msgs=110  tool_calls=30  tools=[read_file, clarify, memory, session_search, search_files]  turns=4
  [E] query: "对比两份分析框架v3，PDF下载份数改回固定值，git commit统一-A参数"
      msgs=922  tool_calls=265  tools=[read_file, patch, session_search, search_files, execute_code, terminal]  turns=8

safety:
  [A] query: "创建'班主任手记'私有知识库(经验案例/家校沟通/活动资料/规章/学生记录)，代号替代真名防敏感"
      msgs=244  tool_calls=97  tools=[todo, execute_code, read_file, patch]  turns=3
  [B] query: "根据世界状态/焦点/记忆，严格按JSON格式输出角色决策"
      msgs=42  tool_calls=10  tools=[session_search, search_files, read_file]  turns=3
  [C] query: "抓包分析商品质保页面，找出隐藏的完整序列号CG2***WV7"
      msgs=12  tool_calls=5  tools=[execute_code, search_files, read_file]  turns=1
  [D] query: "安全巡检报告改'非标准连接'名称，去掉SFTP用户限制显示，配置不动"
      msgs=656  tool_calls=234  tools=[todo, read_file, clarify, memory, patch, session_search, search_files, execute_code]  turns=7
  [E] query: "每次心跳巡检云电脑CPU/内存/磁盘/负载/异常登录/可疑进程/外连，有问题提醒"
      msgs=18  tool_calls=6  tools=[execute_code, memory]  turns=2

coding:
  [A] query: "修复8个强化策略的KlineCalculator数据访问，统一方法签名"
      msgs=42  tool_calls=18  tools=[todo, read_file, patch, session_search, search_files, execute_code, terminal]  turns=1
  [B] query: "详细分析comfyui工作流JSON LTX2.3真无限时长流V13双采多人配音版"
      msgs=114  tool_calls=43  tools=[execute_code, search_files, read_file]  turns=3
  [C] query: "DOT从固定值改倍率伤害，ratio_source枚举对齐技能系统，先分析现有机制再给方案"
      msgs=725  tool_calls=313  tools=[todo, read_file, patch, search_files, execute_code, terminal]  turns=6
  [D] query: "生成CrmBaseOutputService#checkPush的入参JSON，入参和check要一致"
      msgs=54  tool_calls=13  tools=[search_files, read_file]  turns=2
  [E] query: "记录模块文件，代码级审查梳理整个项目，摸清每个模块作用及关联"
      msgs=84  tool_calls=28  tools=[todo, read_file, write_file, search_files, execute_code, terminal]  turns=2

research:
  [A] query: "分析Claude Code记忆插件完整架构(数据流/配置/hook/异步/MCP桥接/子代理/重试队列/状态线)，与kimi-openviking-memory对比"
      msgs=235  tool_calls=68  tools=[todo, terminal, search_files, read_file]  turns=3
  [B] query: "根据开源数据库更新评估5层记忆架构是否需要调整"
      msgs=50  tool_calls=19  tools=[read_file, session_search, search_files, execute_code, terminal]  turns=2
  [C] query: "整理projects.json中记忆和token节省相关的优秀开源项目，完善风禾体记忆框架"
      msgs=18  tool_calls=8  tools=[todo, execute_code, read_file]  turns=1
  [D] query: "对精卫智脑项目数据库查询和匹配机制全面审查"
      msgs=140  tool_calls=53  tools=[todo, read_file, write_file, session_search, search_files, execute_code, terminal]  turns=2
  [E] query: "两男生高考完7天西安洛阳武汉游：景区/线路/天气/火车/费用/完整规划"
      msgs=146  tool_calls=54  tools=[execute_code, terminal]  turns=4

Return ONLY a JSON object with this EXACT structure, no other text:
{
  "workflow": [tool_intensity, reasoning_depth, structure_rigidity, knowledge_domain, multi_step],
  "ops": [...],
  "qa": [...],
  "finance": [...],
  "office": [...],
  "communication": [...],
  "safety": [...],
  "coding": [...],
  "research": [...]
}"""


def _load_api_key() -> str:
    for varname in ("SUFY_API_KEY", "AGENT_MODEL_KEY"):
        key = os.environ.get(varname, "").strip()
        if key:
            return key
    for env_path in (_REPO / ".env", _REPO / "docker" / "sandbox" / "runtime.env"):
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() in ("SUFY_API_KEY", "AGENT_MODEL_KEY") and v.strip().strip('"').strip("'"):
                    return v.strip().strip('"').strip("'")
    raise SystemExit("ERROR: SUFY_API_KEY or AGENT_MODEL_KEY not found.")


def score(api_base: str = "https://openai.sufy.com/v1", model: str = "deepseek-v4-pro-202606") -> dict:
    import urllib.request

    api_key = _load_api_key()
    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.3,
            "max_tokens": 500,
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    raw = resp["choices"][0]["message"]["content"].strip()
    if "```" in raw:
        block = raw.split("```")[1]
        if block.startswith("json"):
            block = block[4:]
        raw = block.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", "-o", default=None, help="output JSON path (default: stdout)")
    ap.add_argument("--model", default="deepseek-v4-pro-202606", help="LLM model for scoring")
    args = ap.parse_args()

    coords = score(model=args.model)

    expected = {"workflow", "ops", "qa", "finance", "office", "communication", "safety", "coding", "research"}
    if set(coords.keys()) != expected:
        print(f"WARNING: got {set(coords.keys())}, expected {expected}", file=sys.stderr)

    for name, vec in coords.items():
        if len(vec) != 5:
            print(f"WARNING: {name} has {len(vec)} dims, expected 5", file=sys.stderr)

    result = {
        "dimensions": ["tool_intensity", "reasoning_depth", "structure_rigidity", "knowledge_domain", "multi_step"],
        "coordinates": coords,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output}")
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
