"""LLM-emitted task domain (= 9-bucket label) for replay-buffer routing.

The raw dataset carries no capability/domain label, but the buffer routes every
trajectory into one of 9 capability buckets. Rather than a separate classifier
pass, the agent emits the domain *while handling the task*: a prompt block
(``build_domain_instruction``) is injected into the rollout system prompt asking
the model to end its work with ``<task_domain>NAME</task_domain>``;
``parse_domain`` then recovers the label from the trajectory text.

Flow: rollout system prompt += build_domain_instruction()  ->  agent emits
``<task_domain>Finance</task_domain>``  ->  parse_domain(...)  ->  bucket
(consumed by trainer/trajectory_adapter.extract_trajectories_from_batch).

Granularity is PER-QUERY (= per trajectory), not per-session: the agent emits
one tag at the end of handling each query, so a multi-domain session simply
routes its queries into different buckets. Because the tag is produced WITH the
full conversation context, context-dependent follow-ups (e.g. "怎么样了") are
still labeled with the ongoing task's domain. See doc/SandboxRollout.md §5.5.

Pure + deterministic (no LLM call here) so it is unit-tested off-GPU.
"""

from __future__ import annotations

import re

# Canonical bucket names + short definitions (configs/base.yaml bucket_names,
# runs/_analysis/capability_buckets/buckets.json). Order is the canonical order.
# 桶体系 = ClawEval 官方 category 合并(去多模态)，单层无子桶(2026-07-04)。
DOMAIN_DEFINITIONS: list[tuple[str, str]] = [
    ("workflow", "多步骤工作流编排（拆解目标、串联动作、条件分支与流程协调）"),
    ("ops", "系统操作（文件读写、命令行/终端执行、系统运维、资源增删改查）"),
    ("qa", "问答检索（查事实、答问题、阅读理解、记忆检索；即查即答，不产出长报告）"),
    ("finance", "财务金融（贷款/税务/估值/ROI 计算、采购等按金融业务规则算账）"),
    ("office", "办公文档（办公问答、表格/报表处理、办公数据分析）"),
    ("communication", "沟通表达（邮件撰写/分类、内容创作、润色改写、翻译）"),
    ("safety", "安全合规（拒绝不安全请求、漏洞/威胁评估、合规审查、敏感操作把关）"),
    ("coding", "代码（编写、审查、调试、修复代码，代码正确性推理）"),
    ("research", "研究综合（查多源信息并综合成报告/简报/摘要；区别于 qa 的即查即答）"),
]

DEFAULT_BUCKETS: list[str] = [name for name, _ in DOMAIN_DEFINITIONS]

DOMAIN_TAG = "task_domain"

# Common surface forms / aliases the model may emit -> canonical bucket.
_ALIASES: dict[str, str] = {
    "workflow": "workflow",
    "productivity": "workflow",
    "organization": "workflow",
    "orchestration": "workflow",
    "ops": "ops",
    "sysops": "ops",
    "operations": "ops",
    "system operations": "ops",
    "terminal": "ops",
    "file_ops": "ops",
    "tooluse": "ops",
    "tool use": "ops",
    "qa": "qa",
    "what": "qa",
    "knowledge": "qa",
    "comprehension": "qa",
    "memory": "qa",
    "question answering": "qa",
    "finance": "finance",
    "financial": "finance",
    "procurement": "finance",
    "office": "office",
    "officeqa": "office",
    "office_qa": "office",
    "office qa": "office",
    "data_analysis": "office",
    "communication": "communication",
    "content": "communication",
    "rewriting": "communication",
    "safety": "safety",
    "security": "safety",
    "compliance": "safety",
    "coding": "coding",
    "code": "coding",
    "research": "research",
    "synthesis": "research",
}

_TAG_RE = re.compile(rf"<{DOMAIN_TAG}>\s*(.*?)\s*</{DOMAIN_TAG}>", re.IGNORECASE | re.DOTALL)


def build_domain_instruction(bucket_names: list[str] | None = None) -> str:
    """Prompt block to append to the agent's rollout system prompt.

    Asks the model to classify the task into exactly one canonical domain and
    emit it as the LAST line via ``<task_domain>NAME</task_domain>``. Pass
    ``bucket_names`` to restrict/relabel; descriptions fall back to the
    canonical set when a custom name has no known definition.
    """
    defs = dict(DOMAIN_DEFINITIONS)
    names = bucket_names or DEFAULT_BUCKETS
    lines = [f"  - {n}: {defs.get(n, '')}".rstrip() for n in names]
    options = "\n".join(lines)
    allowed = " | ".join(names)
    return (
        "## 任务领域标注\n"
        "完成本任务后，请判断该任务属于以下哪一个领域，并在你回复的【最后一行】"
        f"用如下精确格式输出（标签名必须是下列之一，原样英文）：<{DOMAIN_TAG}>NAME</{DOMAIN_TAG}>\n"
        "领域定义：\n"
        f"{options}\n"
        f"可选 NAME（必须精确匹配其一）：{allowed}\n"
        "只输出一个领域；若难以判断，选择最贴近主要意图的一个。"
    )


def parse_domain(text: str, valid: list[str] | None = None) -> str | None:
    """Extract the canonical domain from model output, or None if absent/unknown.

    - Reads the LAST ``<task_domain>...</task_domain>`` tag (the model may
      restate; the final emission is authoritative).
    - Normalizes case/aliases and validates against ``valid`` (default the 9
      canonical buckets). Returns None when no valid tag is found so callers can
      SKIP the trajectory rather than mislabel it (bug B12 discipline).
    """
    if not isinstance(text, str) or not text:
        return None
    valid = valid or DEFAULT_BUCKETS
    valid_lower = {v.lower(): v for v in valid}

    matches = _TAG_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    key = raw.lower()
    if key in valid_lower:
        return valid_lower[key]
    alias = _ALIASES.get(key)
    if alias and alias in valid:
        return alias
    return None
