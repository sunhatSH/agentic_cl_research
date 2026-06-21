"""LLM-emitted task domain (= 7-bucket label) for replay-buffer routing.

The raw dataset carries no capability/domain label, but the buffer routes every
trajectory into one of 7 capability buckets. Rather than a separate classifier
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
# doc/BucketDesign.md). Order is the canonical bucket order.
DOMAIN_DEFINITIONS: list[tuple[str, str]] = [
    ("Workflow", "多步骤任务的组织与编排（计划、串联多个动作完成一个目标）"),
    ("SysOps", "工具使用与系统操作（读写文件、执行命令、调用外部工具/接口）"),
    ("Dialogue", "多轮交互与状态跟踪（澄清、追问、依赖上下文的连续对话）"),
    ("Finance", "结构化业务规则（金融/财务/交易等有明确规则的业务计算）"),
    ("Communication", "表达与沟通（撰写、润色、翻译、面向人的表达）"),
    ("Knowledge", "检索与推理（知识问答、分析、基于资料的推断）"),
    ("OfficeQA", "办公语境问答（办公文档、表格、日常办公场景的问答）"),
]

DEFAULT_BUCKETS: list[str] = [name for name, _ in DOMAIN_DEFINITIONS]

DOMAIN_TAG = "task_domain"

# Common surface forms / aliases the model may emit -> canonical bucket.
_ALIASES: dict[str, str] = {
    "workflow": "Workflow",
    "sysops": "SysOps",
    "sys ops": "SysOps",
    "system operations": "SysOps",
    "system ops": "SysOps",
    "tooluse": "SysOps",
    "tool use": "SysOps",
    "dialogue": "Dialogue",
    "dialog": "Dialogue",
    "conversation": "Dialogue",
    "finance": "Finance",
    "financial": "Finance",
    "communication": "Communication",
    "knowledge": "Knowledge",
    "knowledge/analysis": "Knowledge",
    "knowledge / analysis": "Knowledge",
    "analysis": "Knowledge",
    "officeqa": "OfficeQA",
    "office qa": "OfficeQA",
    "office": "OfficeQA",
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
    - Normalizes case/aliases and validates against ``valid`` (default the 7
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
