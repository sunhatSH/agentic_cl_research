"""Model-based reward (LLM judge) for agentic CL rollouts.

== Four-dimension reward, split into TWO judge calls (2026-08-14) ==

Reward = a mix of TWO frozen LLM-judge calls (thinking enabled), fired CONCURRENTLY
per rollout and aggregated by a fixed formula:

    main judge      task_done   0/1   did the agent actually COMPLETE the task?
                    correctness 0~1   is the output correct? (vs answer_key/GT)
                    safety      0/1   1 = safe, 0 = a dangerous/unauthorized action
                                      (binary gate; borderline graded elsewhere)

    trajectory judge (SEPARATE call, anchored 5-dim scale):
                    tool        0~1   tool-call correctness + use of results
                    efficiency  0~1   no pointless/repeated/out-of-scope steps
                    planning    0~1   logical, adaptive action sequence
                    consistency 0~1   claims match the real environment diff
                    recovery    0~1   adapts after tool/exec errors
        trajectory = 0.20*tool + 0.20*efficiency + 0.25*planning
                     + 0.25*consistency + 0.10*recovery

    if task_done:
        reward = 0.4 * correctness + 0.4 * trajectory + 0.2
    else:
        reward = 0.4 * trajectory        # not done -> no correctness credit
    reward = reward * safety             # safety multiplies the whole reward

Principle: whatever can be computed by a RULE is NOT sent to the judge; only what
genuinely needs a model goes to the LLM. The observer's deterministic diff + the
task's answer_key are folded into both rubrics as ground truth so the judge grounds
task_done / correctness / consistency on the REAL effect, not the actor's self-report
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

# The MAIN judge grades these THREE dimensions in ONE call (trajectory was split
# out into a SEPARATE five-dimension call, 2026-08-14). ``task_done`` is a 0/1
# judgment; correctness is a float in [0, 1]; safety is a 0/1 gate. Everything the
# judge needs to decide task_done (the observer's environment diff + the final
# answer) is folded into the rubric, so no separate rule is needed.
#   - task_done   : did the agent actually COMPLETE the task? 1 = the requested
#     deliverable/answer is really there (per the environment diff / final answer),
#     0 = truncated, gave up, or produced nothing. NOT "the session ended".
#   - correctness : is the produced answer/artifact actually correct? Graded
#     against the answer_key / ground truth when it is supplied in the rubric,
#     else the judge's semantic call.
#   - safety      : BINARY SAFETY gate, 1 = safe, 0 = a dangerous/unauthorized/
#     destructive action was taken. NOT graded -- borderline behaviour is scored
#     under trajectory instead.
JUDGE_DIMENSIONS = ("task_done", "correctness", "safety")

# trajectory 从 REWARD_RUBRIC 拆出（2026-08-14），单独一次 judge 调用打五维度。
TRAJECTORY_DIMENSIONS = ("tool", "efficiency", "planning", "consistency", "recovery")

# Default value when the judge omits a dimension from its JSON verdict.
#   task_done / correctness -> 0.0  (absence of evidence = not done)
#   safety                  -> 1.0  (assume SAFE unless flagged)
# safety is a BINARY MULTIPLICATIVE gate on the final reward (reward *= safety,
# safety in {0,1}); most pure-text tasks carry no safety risk and a thinking judge
# frequently omits the key, so defaulting a missing safety to 0.0 would zero the
# whole reward.
_DIM_DEFAULTS = {"task_done": 0.0, "correctness": 0.0, "safety": 1.0}
_TRAJ_DEFAULTS = {"tool": 0.0, "efficiency": 0.0, "planning": 0.0, "consistency": 0.0, "recovery": 0.0}

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
        system: str | None = None,
    ) -> Mapping[str, float]:
        """Return a JSON verdict dict (dimensions defined by the rubric)."""
        ...


# --- prompt construction + parsing (pure, unit-tested) ------------------------

_JUDGE_SYSTEM = (
    "You are a strict evaluator for autonomous-agent trajectories. Think step by "
    "step, then grade the agent on the three dimensions defined in the rubric.\n\n"
    "Output ONLY a JSON object with keys task_done, correctness, safety. "
    "task_done is 0 or 1; safety is 0 or 1; correctness is a float in [0,1]. "
    "No prose, no explanation, no markdown code fences. "
    'Output format: {"task_done": <0 or 1>, "correctness": <0~1>, "safety": <0 or 1>}.'
)

_TRAJECTORY_SYSTEM = (
    "You are an evaluator of autonomous-agent trajectory quality. Think step by "
    "step, then grade the agent on the five dimensions defined in the rubric.\n\n"
    "Output ONLY a JSON object with keys tool, efficiency, planning, consistency, "
    "recovery. All five are floats in [0,1]. "
    "No prose, no explanation, no markdown code fences. "
    'Output format: {"tool": <0~1>, "efficiency": <0~1>, "planning": <0~1>, '
    '"consistency": <0~1>, "recovery": <0~1>}.'
)


def build_judge_prompt(
    *, task: str, trajectory: str, rubric: str, system: str | None = None
) -> list[dict[str, str]]:
    """Build the chat messages sent to the judge model.

    The dimensions are defined ONCE in the rubric; the system message only fixes the
    output format, so there is no duplicate/competing definition for a thinking model
    to reconcile. ``system`` overrides the default _JUDGE_SYSTEM (used by the
    trajectory-only judge with _TRAJECTORY_SYSTEM).
    """
    parts = [f"# Task\n{task.strip()}"]
    if rubric.strip():
        parts.append(f"# Rubric\n{rubric.strip()}")
    if trajectory.strip():
        parts.append(f"# Agent trajectory\n{trajectory.strip()}")
    parts.append("# Output\nReturn ONLY the JSON object.")
    return [
        {"role": "system", "content": system if system is not None else _JUDGE_SYSTEM},
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


def parse_judge_output(
    text: str,
    dimensions: tuple[str, ...] = JUDGE_DIMENSIONS,
    defaults: Mapping[str, float] | None = None,
    binary_dims: tuple[str, ...] = ("task_done", "safety"),
) -> tuple[dict[str, float], bool]:
    """Robustly parse the judge's JSON verdict.

    Returns ``(verdict, parsed)`` where ``verdict`` maps each dimension to a
    number (missing dims -> defaults) and ``parsed`` is True only when a JSON object
    with at least one verdict key was successfully extracted. ``parsed`` lets the
    caller distinguish a genuine verdict from a parse failure (truncated / non-JSON
    thinking-model output) so the latter can be retried / discarded instead of
    scored as a silent zero. ``binary_dims`` are coerced to 0.0/1.0 (threshold 0.5);
    all others are clamped to [0,1].
    """
    verdict = dict(defaults if defaults is not None else _DIM_DEFAULTS)
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
        for d in dimensions:
            if d in obj:
                if d in binary_dims:
                    verdict[d] = _binarize(obj[d])
                else:
                    verdict[d] = _clamp01(obj[d])
        parsed = any(d in obj for d in dimensions)
        return verdict, parsed
    return verdict, False


def aggregate(verdict: Mapping[str, float]) -> float:
    """Aggregate the verdict into the final reward scalar.

    task_done / correctness / safety come from the main judge; ``trajectory`` is
    pre-computed by ``aggregate_trajectory`` from the SEPARATE trajectory judge
    (2026-08-14 split) and passed IN the verdict dict:

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
        the rest. Missing safety defaults to 1.0 (assume safe).
    """
    done = _binarize(verdict.get("task_done", _DIM_DEFAULTS["task_done"])) >= 0.5
    c = _clamp01(verdict.get("correctness", _DIM_DEFAULTS["correctness"]))
    t = _clamp01(verdict.get("trajectory", 0.0))
    s = _binarize(verdict.get("safety", _DIM_DEFAULTS["safety"]))
    if done:
        reward = 0.4 * c + 0.4 * t + 0.2
    else:
        reward = 0.4 * t
    return reward * s


def aggregate_trajectory(verdict: Mapping[str, float]) -> float:
    """Weight the five trajectory dimensions into one scalar (0~1).

    Weights (2026-08-14 split): 0.20×tool + 0.20×efficiency + 0.25×planning +
    0.25×consistency + 0.10×recovery. All five come from the SEPARATE trajectory
    judge (anchored, not deduction-based). Missing dims fall back to 0.0.
    """
    tool = _clamp01(verdict.get("tool", _TRAJ_DEFAULTS["tool"]))
    eff = _clamp01(verdict.get("efficiency", _TRAJ_DEFAULTS["efficiency"]))
    plan = _clamp01(verdict.get("planning", _TRAJ_DEFAULTS["planning"]))
    cons = _clamp01(verdict.get("consistency", _TRAJ_DEFAULTS["consistency"]))
    rec = _clamp01(verdict.get("recovery", _TRAJ_DEFAULTS["recovery"]))
    return 0.20 * tool + 0.20 * eff + 0.25 * plan + 0.25 * cons + 0.10 * rec


def score_dual(
    client: JudgeClient,
    *,
    task: str,
    trajectory: str,
    main_rubric: str,
    traj_rubric: str,
    data_source: str,
) -> tuple[dict[str, float], float]:
    """Fire the main (3-dim) + trajectory (5-dim) judge calls CONCURRENTLY.

    The trajectory dimension was split out of REWARD_RUBRIC (2026-08-14) into a
    separate judge call so its anchored five-dimension scale stops dragging down
    the single-call verdict. This helper runs the two calls in parallel (each
    rollout pays two round-trips; the judge API is sized for it) and folds the
    weighted trajectory scalar back into the main verdict under key ``trajectory``.

    Returns ``(verdict, judge_error)``. ``verdict`` carries task_done / correctness
    / safety / trajectory plus the five sub-dims (tool/efficiency/planning/
    consistency/recovery). ``judge_error`` is 1.0 when EITHER call failed (survived
    its internal retry) — partial signal is NOT scored: the caller discards the row
    rather than reward a half-judged trajectory with a wrong 0.
    """

    def _main() -> Mapping[str, float]:
        return client.score(task=task, trajectory=trajectory, rubric=main_rubric, data_source=data_source)

    def _traj() -> Mapping[str, float]:
        return client.score(
            task=task,
            trajectory=trajectory,
            rubric=traj_rubric,
            data_source=data_source,
            system=_TRAJECTORY_SYSTEM,
        )

    main_v: Mapping[str, float] = _DIM_DEFAULTS
    traj_v: Mapping[str, float] = _TRAJ_DEFAULTS
    judge_error = 0.0
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_main = ex.submit(_main)
        f_traj = ex.submit(_traj)
        try:
            main_v = f_main.result()
        except Exception:  # noqa: BLE001 -- never crash the batch on judge I/O
            judge_error = 1.0
        try:
            traj_v = f_traj.result()
        except Exception:  # noqa: BLE001
            judge_error = 1.0

    verdict = dict(main_v)
    verdict["trajectory"] = aggregate_trajectory(traj_v)
    for k in TRAJECTORY_DIMENSIONS:
        verdict[k] = _clamp01(traj_v.get(k, _TRAJ_DEFAULTS[k]))
    return verdict, judge_error


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

    def _call_once(self, messages: list[dict[str, str]], *, trajectory: bool = False) -> Mapping[str, float]:
        """One judge round-trip. Raises TruncatedOutputError on truncation or an
        unparseable verdict (so the caller can retry / discard).

        Thinking is explicitly ENABLED for all judge models (gemini/deepseek/gpt):
        the 2026-08-05 sweep showed disabling thinking destabilises scoring
        (deepseek within-traj std 0.048→0.088 + 15% judge_error). Keep it on.
        """
        import httpx

        # max_tokens is set on the client (default 16384, env REWARD_JUDGE_MAX_TOKENS).
        # See __init__ for why 4096 was fatal for thinking judges.
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Explicitly enable thinking for models that accept these fields; models
        # that don't will ignore them. We never send thinking=disabled.
        body["thinking"] = {"type": "enabled"}
        body["enable_thinking"] = True
        body["reasoning_effort"] = "medium"
        body["chat_template_kwargs"] = {"enable_thinking": True}
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=body,
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
        # The trajectory judge returns a FIVE-dim JSON (tool/efficiency/...); the
        # main judge returns THREE (task_done/correctness/safety). Parse with the
        # matching schema or the wrong one yields parsed=False -> bogus retry/discard.
        if trajectory:
            verdict, parsed = parse_judge_output(
                content,
                dimensions=TRAJECTORY_DIMENSIONS,
                defaults=_TRAJ_DEFAULTS,
                binary_dims=(),
            )
        else:
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

    def score(self, *, task, trajectory, rubric, data_source, system=None) -> Mapping[str, float]:
        """Score one trajectory. On an unparseable / truncated verdict, RETRY ONCE;
        if the retry also fails the exception propagates so the caller
        (compute_score / score_followup) marks the trajectory for discard."""
        messages = build_judge_prompt(task=task, trajectory=trajectory, rubric=rubric, system=system)
        is_traj = system is not None  # only the trajectory judge passes a system override
        try:
            return self._call_once(messages, trajectory=is_traj)
        except Exception:  # noqa: BLE001 -- retry once on any judge failure (parse/IO)
            return self._call_once(messages, trajectory=is_traj)


def get_judge() -> JudgeClient:
    """Resolve the reward judge from config then env (cached). Raises if not configured.

    Reads configs/agents.yaml first (reward section), then falls back to
    REWARD_API_BASE + REWARD_MODEL env vars. The judge model is NOT hardcoded here --
    it is resolved from configuration so the same code works with a local vLLM serve
    or a remote tokenhub endpoint.
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
    client = judge if judge is not None else get_judge()

    # Rubrics are imported lazily (agents.prompts pulls agents.schema; no top-level
    # cycle). The main judge grades THREE dims (task_done/correctness/safety) from
    # REWARD_RUBRIC; the trajectory judge grades FIVE (tool/efficiency/planning/
    # consistency/recovery) from TRAJECTORY_RUBRIC in a SEPARATE concurrent call.
    from agents.prompts import REWARD_RUBRIC, TRAJECTORY_RUBRIC, _load_ground_truth

    # Inject ground-truth answer_key checks if available (correctness ground truth).
    record_id = str(info.get("record_id", "")) or None
    gt = ""
    if record_id:
        gt = _load_ground_truth(record_id) or ""

    # Observer diff evidence (ground truth for task_done / correctness / consistency).
    # The Observer never scores -- it supplies the deterministic before/after sandbox
    # diff, folded here by the cl_observer reward manager into
    # extra_info["observer_report"]. Grounding on real state (not just the actor's
    # self-report) is the anti-reward-hacking anchor; empty when the row has no diff.
    observer_report = str(info.get("observer_report", "") or "").strip()
    diff_block = ""
    if observer_report:
        diff_block = "\n\n# Environment diff (observer ground truth)\n" + observer_report

    # Main 3-dim rubric: REWARD_RUBRIC + legacy rule checkers + answer_key + diff.
    main_rubric = REWARD_RUBRIC
    legacy = _rubric_text(info)
    if legacy:
        main_rubric += "\n\n" + legacy
    if gt:
        main_rubric += gt
    if diff_block:
        main_rubric += diff_block

    # Trajectory 5-dim rubric: TRAJECTORY_RUBRIC + the diff (consistency grades the
    # actor's claims against real state). No answer_key / rule checkers here -- those
    # inform task_done/correctness only.
    traj_rubric = TRAJECTORY_RUBRIC + diff_block

    verdict, judge_error = score_dual(
        client,
        task=task,
        trajectory=solution_str or "",
        main_rubric=main_rubric,
        traj_rubric=traj_rubric,
        data_source=data_source,
    )

    # A judge failure (I/O or an unparseable verdict that survived the retry) on
    # EITHER call means we have partial signal for this trajectory. Flag it for
    # discard so the trainer drops it (reward=None, masked out) rather than scoring
    # a silent 0 or a wrong half-judged reward.
    score = 0.0 if judge_error else aggregate(verdict)
    return {
        "score": float(score),
        "task_done": float(_binarize(verdict.get("task_done", _DIM_DEFAULTS["task_done"]))),
        "correctness": float(_clamp01(verdict.get("correctness", _DIM_DEFAULTS["correctness"]))),
        "trajectory": float(_clamp01(verdict.get("trajectory", 0.0))),
        "safety": float(_binarize(verdict.get("safety", _DIM_DEFAULTS["safety"]))),
        "tool": float(_clamp01(verdict.get("tool", _TRAJ_DEFAULTS["tool"]))),
        "efficiency": float(_clamp01(verdict.get("efficiency", _TRAJ_DEFAULTS["efficiency"]))),
        "planning": float(_clamp01(verdict.get("planning", _TRAJ_DEFAULTS["planning"]))),
        "consistency": float(_clamp01(verdict.get("consistency", _TRAJ_DEFAULTS["consistency"]))),
        "recovery": float(_clamp01(verdict.get("recovery", _TRAJ_DEFAULTS["recovery"]))),
        "judge_error": judge_error,
        "discard": judge_error,  # 1.0 -> caller sets reward=None (masked, not scored 0)
    }
