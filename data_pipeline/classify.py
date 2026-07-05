"""Step 2 — LLM 把首 query 分到 9 桶之一，并精确到子桶（category）。

子桶清单取自 ``runs/_analysis/capability_buckets/buckets.json`` 各桶的官方 category
（ClawEval General split），是项目的既定设计——不另起炉灶::

    workflow      : workflow, productivity, organization
    ops           : ops, operations, terminal, file_ops
    qa            : what, knowledge, comprehension, memory
    finance       : finance, procurement
    office        : office_qa, data_analysis
    communication : communication, content, rewriting
    safety        : safety, security, compliance
    coding        : coding
    research      : research, synthesis

LLM 返回 JSON ``{"bucket": str, "sub_bucket": str, "rationale": str}``，``bucket``
必须是 9 桶之一（``trainer/domain_tagging.DEFAULT_BUCKETS``），``sub_bucket``
必须是该桶下的 category 之一；无法归类时 ``bucket="unknown"``、``sub_bucket=None``
（不入 buffer，与 ``trajectory_adapter`` 的 B12 跳过纪律一致）。

复用 ``agents/base.py`` 的 ``ChatClient`` Protocol + ``agents/config.py`` 的
``resolve_observer``（observer 端点：确定性、temp=0，适合分类）——分类是"客观
取证"性质，与 observer 同一温度语义。模型可经 env 覆盖
（``BUCKET_CLASSIFIER_API_BASE / _MODEL``），缺省用 observer 配置。

JSON 解析容错：截断/非 JSON 输出（thinking 模型常见）降级为 unknown，不静默
全 0——与 ``model_reward.parse_judge_output`` 同一纪律。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable

from trainer.domain_tagging import DEFAULT_BUCKETS

logger = logging.getLogger(__name__)

# 9 桶 → 各桶的子桶（category）清单。取自 runs/_analysis/capability_buckets/buckets.json（既定设计）。
SUB_BUCKETS: dict[str, list[str]] = {
    "workflow": ["workflow", "productivity", "organization"],
    "ops": ["ops", "operations", "terminal", "file_ops"],
    "qa": ["what", "knowledge", "comprehension", "memory"],
    "finance": ["finance", "procurement"],
    "office": ["office_qa", "data_analysis"],
    "communication": ["communication", "content", "rewriting"],
    "safety": ["safety", "security", "compliance"],
    "coding": ["coding"],
    "research": ["research", "synthesis"],
}

# 桶 + 一句话定义（给 LLM 的判据，取自 trainer/domain_tagging.DOMAIN_DEFINITIONS）。
from trainer.domain_tagging import DOMAIN_DEFINITIONS  # noqa: E402

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)

_CLASSIFY_SYSTEM = (
    "你是一个任务分类器。给定用户对一个 agentic 助手的首条 query，判断它属于以下"
    "9 个能力桶中的哪一个，并精确到该桶下的子桶（category）。"
)


def _build_options_block() -> str:
    """渲染 9 桶 + 各子桶的选项清单（供 prompt 注入）。"""
    lines = []
    for bucket, sub_buckets in SUB_BUCKETS.items():
        # DOMAIN_DEFINITIONS 是 [(name, def), ...]，取该桶定义
        definition = dict(DOMAIN_DEFINITIONS).get(bucket, "")
        subs = " / ".join(sub_buckets)
        lines.append(f"- {bucket}（{definition}）子桶: {subs}")
    return "\n".join(lines)


def build_classify_prompt(query: str) -> list[dict[str, str]]:
    """构造分类请求的 chat messages（system + user）。

    返回 OpenAI chat 格式，喂给 ``ChatClient.chat``。
    """
    options = _build_options_block()
    user = (
        f"## 桶与子桶定义\n{options}\n\n"
        f"## 待分类的首条 query\n{query}\n\n"
        "## 输出要求\n"
        "只输出一个 JSON 对象，不要任何额外文字：\n"
        '{"bucket": "九桶之一", "sub_bucket": "该桶子桶之一", "rationale": "一句话理由"}\n'
        "规则：\n"
        "1. bucket 必须是九桶之一（workflow/ops/qa/finance/office/communication/safety/coding/research），原样英文。\n"
        "2. sub_bucket 必须是你选的 bucket 下列出的子桶之一，原样。\n"
        "3. 若确实无法归入任何桶，返回 {\"bucket\": \"unknown\", \"sub_bucket\": null}。"
    )
    return [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_classify_output(text: str) -> dict:
    """容错解析 LLM 的分类 JSON。

    返回 ``{"bucket": str|None, "sub_bucket": str|None, "rationale": str}``。
    bucket 校验为 9 桶之一；sub_bucket 校验为该桶子桶之一；不合法降级为
    ``bucket="unknown"``（不入 buffer）。截断/非 JSON → unknown（不静默）。
    """
    result: dict = {"bucket": "unknown", "sub_bucket": None, "rationale": ""}
    if not isinstance(text, str) or not text.strip():
        return result
    obj: dict | None = None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                obj = None
    if not isinstance(obj, dict):
        # 非法/截断输出：明确标 unknown，调用方据此跳过（不静默全 0/误分桶）。
        result["rationale"] = f"parse_failed: {text[:120]!r}"
        return result

    bucket = str(obj.get("bucket", "")).strip()
    sub = obj.get("sub_bucket")
    if bucket.lower() in DEFAULT_BUCKETS:
        bucket = bucket.lower()  # normalize LLM output case
        result["bucket"] = bucket
        valid_subs = SUB_BUCKETS.get(bucket, [])
        if isinstance(sub, str) and sub.strip() in valid_subs:
            result["sub_bucket"] = sub.strip()
        else:
            # bucket 合法但 sub_bucket 不在该桶清单 → 标 None，保留 bucket。
            result["sub_bucket"] = None
            result["rationale"] = f"invalid_sub_bucket: {sub!r}"
    else:
        result["bucket"] = "unknown"
        result["sub_bucket"] = None
    if not result["rationale"]:
        rationale = obj.get("rationale", "")
        result["rationale"] = str(rationale)[:200] if rationale else ""
    return result


def classify_query(
    query: str,
    client,
    *,
    max_tokens: int = 256,
) -> dict:
    """用注入的 ``ChatClient`` 分类单条 query。

    ``client`` 需实现 ``agents.base.ChatClient.chat(messages, *, max_tokens)``。
    返回 ``{bucket, sub_bucket, rationale, model, ok}``；模型调用失败/截断时
    ``ok=False``、``bucket="unknown"``，调用方据此跳过或重试。
    """
    messages = build_classify_prompt(query)
    try:
        text = client.chat(messages, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — 网络层异常统一降级，不中断整批
        logger.warning("classify LLM call failed: %s", exc)
        return {
            "bucket": "unknown",
            "sub_bucket": None,
            "rationale": f"llm_error: {exc!r}",
            "model": getattr(client, "model", "unknown"),
            "ok": False,
        }
    parsed = parse_classify_output(text)
    parsed["model"] = getattr(client, "model", "unknown")
    parsed["ok"] = parsed["bucket"] != "unknown"
    return parsed


def classify_queries(
    records: Iterable[dict],
    client,
    *,
    max_tokens: int = 256,
) -> list[dict]:
    """批量分类：对每条首 query 记录调用 LLM，附上分类结果。

    输入是 extract 产出的首 query 记录（1:n 重写后为 query 列表）；输出每条加 ``bucket / sub_bucket /
    rationale / model / ok``。单条失败不影响其它（逐条 try）。
    """
    out: list[dict] = []
    for rec in records:
        verdict = classify_query(rec["first_query"], client, max_tokens=max_tokens)
        merged = {**rec, **verdict}
        out.append(merged)
    return out


def resolve_classifier_endpoint(config_path=None):
    """解析分类器端点：env 覆盖优先，缺省用 observer 配置（temp=0、确定性）。

    返回 ``agents.config.ResolvedEndpoint``。分类是客观取证性质，与 observer
    同一温度语义；想换模型用 ``BUCKET_CLASSIFIER_API_BASE / _MODEL`` env。
    """
    from agents.config import resolve_observer

    base = os.environ.get("BUCKET_CLASSIFIER_API_BASE", "").strip()
    model = os.environ.get("BUCKET_CLASSIFIER_MODEL", "").strip()
    if base and model:
        from agents.config import ResolvedEndpoint

        key = os.environ.get("SUFY_API_KEY", "").strip() or "sk-local"
        return ResolvedEndpoint(base_url=base, model=model, api_key=key, temperature=0.0)
    return resolve_observer(config_path)


def make_default_client(config_path=None):
    """构造默认分类器 client（OpenAIChatClient，端点取 resolve_classifier_endpoint）。

    供 CLI 使用；单测注入 mock client，不触网。
    """
    from agents.base import OpenAIChatClient

    ep = resolve_classifier_endpoint(config_path)
    return OpenAIChatClient(
        base_url=ep.base_url,
        model=ep.model,
        api_key=ep.api_key,
        temperature=ep.temperature,
    )
