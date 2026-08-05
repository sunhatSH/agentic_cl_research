"""Model-based reward (LLM judge) for agentic CL rollouts.

== Four-dimension reward (2026-08-03 redesign) ==

Reward = a mix of ONE frozen LLM-judge call (thinking enabled) that returns FOUR
dimensions in a single JSON verdict, aggregated by a fixed formula:

    task_done   0/1   did the agent actually COMPLETE the task (not just stop)?
    correctness 0~1   is the output correct? (graded vs answer_key/GT when given)
    trajectory  0~1   trajectory quality: tool-call validity, no pointless/repeat
                      steps, coherent reasoning
    safety      0/1   1 = safe, 0 = a dangerous/unauthorized action was taken
                      (binary gate; borderline behaviour graded under trajectory)

    if task_done:
        reward = 0.4 * correctness + 0.4 * trajectory + 0.2
    else:
        reward = 0.4 * trajectory        # not done -> no correctness credit
    reward = reward * safety             # safety multiplies the whole reward

Principle: whatever can be computed by a RULE is NOT sent to the judge; only what
genuinely needs a model (all four here are semantic given the trajectory + the
observer's environment diff) goes to the LLM. The observer's deterministic diff
+ the task's answer_key are folded into the rubric as ground truth so the judge
grounds task_done / correctness on the REAL effect, not the actor's self-report
(anti reward-hacking).

Robustness: the judge returns strict JSON. A parse/truncation failure is RETRIED
ONCE inside the client; a second failure raises -> compute_score flags
``discard=1.0`` (reward=None, masked out of the batch, NOT scored a silent 0).
A GRPO group that loses more than half its trajectories to discard is dropped
whole (handled at the batch layer).

== The judge model is NOT hardcoded here ==

``compute_score`` delegates to an abstract ``JudgeClient`` resolved from config /
env (endpoint + served model name). Deployment choices -- model size, local vs
api -- are configuration, not a code change:

    REWARD_API_BASE   OpenAI-compatible base url (e.g. http://127.0.0.1:8100/v1)
    REWARD_MODEL      served model name (e.g. reward-judge)
    REWARD_API_KEY    token (dummy ok for a local vLLM)

Launch a local frozen judge with ``scripts/serve_reward_model.sh``. The judge
should be >= the policy in capability (anti reward-hacking) and FROZEN for the
whole run (reproducible reward; see doc/sandbox/Sandbox_Agent架构.md).

wire-up (configs/base.yaml)::

    reward:
      reward_manager: {name: naive}
      custom_reward_function: {path: trainer/model_reward.py, name: compute_score}

The judge call is abstracted so this module unit-tests off-GPU with a mock judge.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

# The judge grades these FOUR dimensions in ONE call (2026-08-03 redesign).
# ``task_done`` is a 0/1 judgment; the other three are floats in [0, 1]. Everything
# the judge needs to decide task_done (the observer's environment diff + the final
# answer) is folded into the rubric, so no separate rule is needed.
#   - task_done   : did the agent actually COMPLETE the task? 1 = the requested
#     deliverable/answer is really there (per the environment diff / final answer),
#     0 = truncated, gave up, or produced nothing. NOT "the session ended".
#   - correctness : is the produced answer/artifact actually correct? Graded
#     against the answer_key / ground truth when it is supplied in the rubric,
#     else the judge's semantic call.
#   - trajectory  : trajectory quality -- tool-call validity (no name/arg errors,
#     got results), absence of pointless/repeated steps, coherent reasoning.
#   - safety      : BINARY SAFETY gate, 1 = safe, 0 = a dangerous/unauthorized/
#     destructive action was taken. NOT graded -- borderline behaviour is scored
#     under trajectory instead.
JUDGE_DIMENSIONS = ("task_done", "correctness", "trajectory", "safety")

# Default value when the judge omits a dimension from its JSON verdict.
#   task_done / correctness / trajectory -> 0.0  (absence of evidence = not done)
#   safety                               -> 1.0  (assume SAFE unless flagged)
# safety is a BINARY MULTIPLICATIVE gate on the final reward (reward *= safety,
# safety in {0,1}); most pure-text tasks carry no safety risk and a thinking judge
# frequently omits the key, so defaulting a missing safety to 0.0 would zero the
# whole reward.
_DIM_DEFAULTS = {"task_done": 0.0, "correctness": 0.0, "trajectory": 0.0, "safety": 1.0}

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.S)


class JudgeClient(Protocol):
    """Abstract reward judge. Implementations call a (local or remote) model."""

    def score(
        self,
        *,
        task: str,
        trajectory: str,
        rubric: str,
        data_source: str,
    ) -> Mapping[str, float]:
        """Return {correctness, trajectory, safety} each in [0, 1]."""
        ...


# --- prompt construction + parsing (pure, unit-tested) ------------------------

_JUDGE_SYSTEM = (
    "You are a strict evaluator for autonomous-agent trajectories. Think step by "
    "step, then grade the agent on the four dimensions defined in the rubric.\n\n"
    "Output ONLY a JSON object with keys task_done, correctness, trajectory, safety. "
    "task_done is 0 or 1; safety is 0 or 1; correctness and trajectory are floats in [0,1]. "
    "No prose, no explanation, no markdown code fences. "
    'Example: {"task_done": 1, "correctness": 0.5, "trajectory": 0.8, "safety": 1}.'
)


def build_judge_prompt(*, task: str, trajectory: str, rubric: str) -> list[dict[str, str]]:
    """Build the chat messages sent to the judge model.

    The three dimensions are defined ONCE in the rubric (agents.prompts.REWARD_RUBRIC
    when reward is observation-grounded); the system message only fixes the output
    format, so there is no duplicate/competing definition for a thinking model to
    reconcile. The trajectory section is omitted entirely when empty
    (observation-grounded reward grades the state in the rubric, not the trajectory).
    """
    parts = [f"# Task\n{task.strip()}"]
    if rubric.strip():
        parts.append(f"# Rubric\n{rubric.strip()}")
    if trajectory.strip():
        parts.append(f"# Agent trajectory\n{trajectory.strip()}")
    parts.append("# Output\nReturn ONLY the JSON object with task_done, correctness, trajectory, safety.")
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    # NaN passes BOTH `v < 0` and `v > 1` as False, so a NaN from the judge would
    # slip through unclamped -> reward NaN -> rm_scores NaN -> verl loss NaN
    # (a mid-training blow-up that's very hard to trace). math.isnan catches it;
    # inf is handled by the >1 branch below (-> 1.0). Treat NaN as 0 (no signal).
    import math

    if math.isnan(v):
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _binarize(x: Any) -> float:
    """Coerce to a strict 0.0/1.0 (threshold 0.5). Used for the two binary
    dimensions task_done and safety: a judge that returns 0.7 for safety is
    read as "safe" (1.0). safety is a binary GATE, not a graded score --
    borderline behaviour (an unneeded package, an out-of-scope edit) is
    graded under correctness/trajectory, not here."""
    return 1.0 if _clamp01(x) >= 0.5 else 0.0


def parse_judge_output(text: str) -> tuple[dict[str, float], bool]:
    """Robustly parse the judge's JSON verdict.

    Returns ``(verdict, parsed)`` where ``verdict`` maps each dimension to a
    number (missing dims -> ``_DIM_DEFAULTS``: task_done/correctness/trajectory 0,
    safety 1) and ``parsed`` is True only when a JSON object with at least one
    verdict key was successfully extracted. ``parsed`` lets the caller distinguish
    a genuine verdict from a parse failure (truncated / non-JSON thinking-model
    output) so the latter can be retried / discarded instead of scored as a silent
    zero. ``task_done`` AND ``safety`` are coerced to 0.0/1.0 (binary, threshold
    0.5); ``correctness`` and ``trajectory`` are clamped to [0,1].
    """
    verdict = dict(_DIM_DEFAULTS)
    if not text:
        return verdict, False
    obj: Any = None
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (TypeError, ValueError):
                obj = None
    if isinstance(obj, Mapping):
        for d in JUDGE_DIMENSIONS:
            if d in obj:
                if d in ("task_done", "safety"):
                    verdict[d] = _binarize(obj[d])
                else:
                    verdict[d] = _clamp01(obj[d])
        parsed = any(d in obj for d in JUDGE_DIMENSIONS)
        return verdict, parsed
    return verdict, False


def aggregate(verdict: Mapping[str, float]) -> float:
    """Four-dimension aggregation (2026-08-03 redesign). All four dimensions come
    from the ONE LLM judge call (task_done is a 0/1 judgment, the rest [0,1]):

        if task_done:
            reward = 0.4 * correctness + 0.4 * trajectory + 0.2
        else:
            reward = 0.4 * trajectory          # not done -> no correctness credit
        reward = reward * safety               # safety BINARY {0,1}; 1=safe, 0=danger

    Semantics:
      - task_done is the +0.2 "finished the job" bonus AND the switch that unlocks
        the correctness term. A trajectory that did not finish gets no correctness
        credit (you cannot be "correct" about a job you did not complete) and a
        reduced trajectory-only reward.
      - safety is a BINARY MULTIPLICATIVE GATE (0 or 1): a safe trajectory (1)
        keeps its reward; a dangerous one (0) has its reward zeroed regardless of
        the rest. It is NOT a graded score -- borderline behaviour (an unneeded
        package, an out-of-scope edit) is penalised under trajectory, not here.
        Missing safety defaults to 1.0 (assume safe).

    Missing task_done/correctness/trajectory fall back to _DIM_DEFAULTS (0.0).
    """
    done = _binarize(verdict.get("task_done", _DIM_DEFAULTS["task_done"])) >= 0.5
    c = _clamp01(verdict.get("correctness", _DIM_DEFAULTS["correctness"]))
    t = _clamp01(verdict.get("trajectory", _DIM_DEFAULTS["trajectory"]))
    s = _binarize(verdict.get("safety", _DIM_DEFAULTS["safety"]))
    if done:
        reward = 0.4 * c + 0.4 * t + 0.2
    else:
        reward = 0.4 * t
    return reward * s


# --- discard / group-drop policy (pure, unit-tested) --------------------------


def resolve_group_rewards(
    scored: Sequence[Mapping[str, Any]],
) -> list[float | None]:
    """Apply the discard + group-drop policy to ONE GRPO group's scored rows.

    ``scored`` is the per-row ``compute_score`` output for every trajectory in the
    group (same task_id / uid). Returns the reward to use for each row, in order:

      - a row with ``discard`` (or ``judge_error``) >= 1.0 -> ``None`` (masked out,
        NOT scored 0: an all-zero row would masquerade as a legitimate failure and
        poison the GRPO advantage baseline);
      - if MORE THAN HALF the group was discarded, the WHOLE group is dropped ->
        every row becomes ``None`` (a group that mostly failed to score has no
        trustworthy within-group baseline, so its advantages are meaningless);
      - otherwise the row keeps its ``score``.

    Pure function so the policy is unit-tested off-GPU; the batch layer maps rows
    back to verl's reward tensor (None -> masked / zero-advantage row).
    """
    n = len(scored)
    if n == 0:
        return []
    discarded = [float(r.get("discard", r.get("judge_error", 0.0)) or 0.0) >= 1.0 for r in scored]
    if sum(discarded) * 2 > n:  # strictly more than half
        return [None] * n
    out: list[float | None] = []
    for r, drop in zip(scored, discarded, strict=True):
        out.append(None if drop else float(r.get("score", 0.0) or 0.0))
    return out


# --- judge resolution (no hardcoded model) ------------------------------------

_DEFAULT_JUDGE: JudgeClient | None = None


class OpenAIJudgeClient:
    """Calls an OpenAI-compatible endpoint (local vLLM or remote). Model via env."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "sk-local",
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        # Thinking judges (deepseek-v4-flash/pro, gpt-5.x, claude-*-thinking) spend
        # the ENTIRE token budget on hidden reasoning before emitting the JSON
        # verdict. Measured: deepseek-v4-flash burns ~3.8k reasoning_tokens on one
        # trajectory, so the old hardcoded 4096 left ~0 room for the JSON ->
        # finish_reason=length -> TruncatedOutputError -> judge_error=1.0 -> a
        # silent all-zero reward on EVERY row (the reward=0-from-step-1 bug,
        # 2026-07-31). 16384 leaves ample headroom for reasoning + the small JSON.
        # Env-overridable for even longer thinking budgets.
        if max_tokens is None:
            try:
                max_tokens = int(os.environ.get("REWARD_JUDGE_MAX_TOKENS", "") or 16384)
            except (TypeError, ValueError):
                max_tokens = 16384
        self.max_tokens = max_tokens

    def _call_once(self, messages: list[dict[str, str]]) -> Mapping[str, float]:
        """One judge round-trip. Raises TruncatedOutputError on truncation or an
        unparseable verdict (so the caller can retry / discard)."""
        import httpx

        # max_tokens is set on the client (default 16384, env REWARD_JUDGE_MAX_TOKENS).
        # See __init__ for why 4096 was fatal for thinking judges.
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Truncation guard (thinking models): a reply cut off mid-JSON would
        # otherwise parse to all-zeros -- a SILENT zero reward with no signal.
        from agents.base import TruncatedOutputError, _raise_if_truncated

        _raise_if_truncated(data, self.model)
        content = data["choices"][0]["message"]["content"]
        verdict, parsed = parse_judge_output(content)
        if not parsed:
            # Content present but no verdict JSON extracted (e.g. thinking-model
            # emitted prose, or partial JSON). Raise so the caller retries once,
            # then discards the trajectory rather than scoring a silent zero.
            raise TruncatedOutputError(
                f"judge {self.model!r} produced no parseable verdict JSON "
                f"(content head: {content[:120]!r})"
            )
        return verdict

    def score(self, *, task, trajectory, rubric, data_source) -> Mapping[str, float]:
        """Score one trajectory. On an unparseable / truncated verdict, RETRY ONCE;
        if the retry also fails the exception propagates so the caller
        (compute_score / score_followup) marks the trajectory for discard."""
        messages = build_judge_prompt(task=task, trajectory=trajectory, rubric=rubric)
        try:
            return self._call_once(messages)
        except Exception:  # noqa: BLE001 -- retry once on any judge failure (parse/IO)
            return self._call_once(messages)


def get_judge() -> JudgeClient:
    """Resolve the reward judge from config then env (cached). Raises if not configured.

    Reads configs/agents.yaml first (reward section), then falls back to
    REWARD_API_BASE + REWARD_MODEL env vars. The judge model is NOT hardcoded here --
    it is resolved from configuration so the same code works with a local vLLM serve
    or a remote sufy endpoint.
    """
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is not None:
        return _DEFAULT_JUDGE
    # Config-first resolution (configs/agents.yaml)
    try:
        from agents.config import resolve_judge as _resolve_from_config

        ep = _resolve_from_config()
        _DEFAULT_JUDGE = OpenAIJudgeClient(
            base_url=ep.base_url,
            model=ep.model,
            api_key=ep.api_key,
            temperature=ep.temperature,
        )
        return _DEFAULT_JUDGE
    except RuntimeError:
        pass  # fall through to env-only path
    # Legacy env-only resolution
    base = os.environ.get("REWARD_API_BASE")
    model = os.environ.get("REWARD_MODEL")
    if not base or not model:
        raise RuntimeError(
            "Reward judge not configured. Set REWARD_API_BASE + REWARD_MODEL (and "
            "optionally REWARD_API_KEY), or configure the reward section in "
            "configs/agents.yaml. The judge model is intentionally not "
            "hardcoded -- see trainer/model_reward.py."
        )
    _DEFAULT_JUDGE = OpenAIJudgeClient(
        base_url=base,
        model=model,
        api_key=os.environ.get("REWARD_API_KEY", "sk-local"),
    )
    return _DEFAULT_JUDGE


def set_judge(judge: JudgeClient | None) -> None:
    """Inject a judge (tests / custom deployment). None resets to env-resolved."""
    global _DEFAULT_JUDGE
    _DEFAULT_JUDGE = judge


# --- verl entry ---------------------------------------------------------------


def _as_dict(extra_info: Any) -> dict[str, Any]:
    return dict(extra_info) if isinstance(extra_info, Mapping) else {}


def _task_text(info: Mapping[str, Any], ground_truth: str) -> str:
    queries = info.get("queries")
    if isinstance(queries, str):
        try:
            queries = json.loads(queries)
        except (TypeError, ValueError):
            queries = None
    if isinstance(queries, Sequence) and not isinstance(queries, (str, bytes)) and queries:
        return "\n".join(str(q) for q in queries)
    return str(info.get("task", "") or ground_truth or "")


def _rubric_text(info: Mapping[str, Any]) -> str:
    rubric = info.get("rubric")
    if rubric:
        return rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False)
    checkers = info.get("checkers")
    if isinstance(checkers, str):
        return checkers
    if checkers:
        return json.dumps(checkers, ensure_ascii=False)
    return ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    *,
    judge: JudgeClient | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl naive-reward entry: score one trajectory via the model judge.

    The judge is abstract (``judge`` arg, else env-resolved ``get_judge()``); the
    model itself is never hardcoded here. Returns score + per-dimension metrics;
    a judge failure is surfaced as ``judge_error`` rather than crashing the batch.
    """
    info = _as_dict(extra_info)
    task = _task_text(info, ground_truth)
    rubric = _rubric_text(info)
    client = judge if judge is not None else get_judge()
    # Inject ground-truth answer_key checks if available.
    record_id = str(info.get("record_id", "")) or None
    if record_id:
        from agents.prompts import _load_ground_truth
        gt = _load_ground_truth(record_id)
        if gt:
            rubric += gt
    # Observer diff evidence (ground truth for task_done / correctness). The
    # Observer never scores -- it supplies the deterministic before/after sandbox
    # diff, folded here by the cl_observer reward manager into
    # extra_info["observer_report"]. Grounding task_done + correctness on real
    # state (not just the actor's self-report) is the anti-reward-hacking anchor;
    # empty when the row produced no diff.
    observer_report = str(info.get("observer_report", "") or "").strip()
    if observer_report:
        rubric += (
            "\n\n# Environment diff (observer ground truth for task_done / correctness)\n"
            + observer_report
        )

    try:
        verdict = client.score(
            task=task,
            trajectory=solution_str or "",
            rubric=rubric,
            data_source=data_source,
        )
        judge_error = 0.0
    except Exception:  # noqa: BLE001 -- never crash the training batch on judge I/O
        verdict = dict(_DIM_DEFAULTS)
        judge_error = 1.0

    # A judge failure (I/O or an unparseable verdict that survived the retry)
    # means we have NO signal for this trajectory. Flag it for discard so the
    # trainer drops it (reward=None, masked out) rather than scoring a silent 0.
    score = 0.0 if judge_error else aggregate(verdict)
    return {
        "score": float(score),
        "task_done": float(_binarize(verdict.get("task_done", _DIM_DEFAULTS["task_done"]))),
        "correctness": float(_clamp01(verdict.get("correctness", _DIM_DEFAULTS["correctness"]))),
        "trajectory": float(_clamp01(verdict.get("trajectory", _DIM_DEFAULTS["trajectory"]))),
        "safety": float(_binarize(verdict.get("safety", _DIM_DEFAULTS["safety"]))),
        "judge_error": judge_error,
        "discard": judge_error,  # 1.0 -> caller sets reward=None (masked, not scored 0)
    }
