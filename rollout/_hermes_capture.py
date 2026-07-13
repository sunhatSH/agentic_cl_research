"""Sandbox-side hermes trajectory capture (方案3 P2).

This script runs INSIDE the sandbox (where hermes is pip-installed as an editable
package, so ``import run_agent`` works — pyproject py-modules lists run_agent).
It patches ``AIAgent.run_conversation`` to stash the STRUCTURED result (messages
with tool_calls, POST-compression = train/infer consistent) and ``delegate_tool.
_run_single_child`` to harvest each sub-agent (delegate_task) child trajectory
hermes spawns on its own. It reads {query, conversation_history} from a JSON input
file and writes {messages, children, ok, error} to an output file.

Contract (input file → stdout JSON, probe-style, so the actor reuses the same
read-back path as observer probes; no data crosses the dev process):
    python _hermes_capture.py <input.json>
    input.json  = {"query": str, "history": [msg...], "max_iterations": int}
    stdout      = a single line ``__CAPTURE__<json>`` where <json> =
                  {"messages": [...], "children": [{task_index,goal,messages}],
                   "ok": bool, "error": str}
    (the marker prefix lets the actor find the payload even if hermes prints noise.)

hermes model/base/key come from the sandbox's ~/.hermes/config.yaml (already written
by _write_hermes_config) + AGENT_MODEL_* env, so keys never leave the sandbox.

⚠️ Version note: the sandbox image pins HERMES_VERSION (e.g. v2026.6.5). The
AIAgent(...) kwargs and run_conversation signature below match the reference
source; if a pinned version differs, this script must be validated in-sandbox
(P2 smoke) and adjusted. It fails LOUD (writes error to output) rather than
silently producing a degraded trajectory.
"""

import json
import os
import sys
import threading

# ---- child (sub-agent) capture sink: run_token -> list[child cap] ----
_CHILD_SINK = {}
_CHILD_SINK_LOCK = threading.Lock()
_PATCHED = False


def _install_patches():
    """Wrap run_conversation (stash structured result) + _run_single_child
    (harvest each delegated child's trajectory). Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    import run_agent

    _orig_run = run_agent.AIAgent.run_conversation

    def _wrapped_run(self, *a, **k):
        result = _orig_run(self, *a, **k)
        try:
            self._cap = {
                "messages": (result or {}).get("messages") if isinstance(result, dict) else None,
                "completed": (result or {}).get("completed") if isinstance(result, dict) else None,
                "error": (result or {}).get("error") if isinstance(result, dict) else None,
                "ephemeral_system_prompt": getattr(self, "ephemeral_system_prompt", None),
                "model": getattr(self, "model", None),
            }
        except Exception:
            self._cap = None
        return result

    run_agent.AIAgent.run_conversation = _wrapped_run

    # Child harvest — only fires if hermes autonomously calls delegate_task.
    try:
        from tools import delegate_tool

        _orig_child = delegate_tool._run_single_child

        def _wrapped_child(task_index, goal, child=None, parent_agent=None, **kw):
            ret = _orig_child(task_index, goal, child, parent_agent, **kw)
            try:
                token = getattr(parent_agent, "_run_token", None)
                cap = getattr(child, "_cap", None)
                if token is not None and cap is not None:
                    with _CHILD_SINK_LOCK:
                        sink = _CHILD_SINK.get(token)
                    if sink is not None:
                        sink.append({"task_index": task_index, "goal": goal, "cap": cap})
            except Exception:
                pass
            return ret

        delegate_tool._run_single_child = _wrapped_child
    except Exception:
        # delegate tool absent in this hermes build -> single-agent only, fine.
        pass

    _PATCHED = True


def _child_messages(children_caps):
    """Flatten harvested child caps -> [{task_index, goal, messages}]."""
    out = []
    for c in children_caps or []:
        cap = c.get("cap") or {}
        out.append(
            {
                "task_index": c.get("task_index"),
                "goal": c.get("goal"),
                "messages": cap.get("messages") or [],
            }
        )
    return out


def _resolve_runtime(name: str, model: str) -> dict:
    """Resolve endpoint via hermes' OWN oneshot resolver (the working path).

    Returns a dict with base_url/api_key/provider/api_mode (whatever the resolver
    provides). Falls back to reading providers.<name> from ~/.hermes/config.yaml,
    then to {} (caller then uses env). Never raises."""
    # 1. hermes' own resolver — identical to `hermes chat -q` oneshot.
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested=name, target_model=model or None)
        if isinstance(rt, dict) and rt.get("base_url"):
            return rt
    except Exception:
        pass
    # 2. plain config.yaml providers.<name> read.
    try:
        import yaml

        cfg = yaml.safe_load(open(os.path.expanduser("~/.hermes/config.yaml"), encoding="utf-8")) or {}
        prov = ((cfg.get("providers") or {}).get(name)) or {}
        if prov.get("base_url"):
            return {"base_url": str(prov["base_url"]), "api_key": str(prov.get("api_key") or ""),
                    "provider": name}
    except Exception:
        pass
    return {}


def main() -> None:
    in_path = sys.argv[1]
    result_out = {"messages": [], "children": [], "ok": False, "error": ""}
    try:
        spec = json.load(open(in_path, encoding="utf-8"))
        query = spec["query"]
        history = spec.get("history") or []
        max_iter = int(spec.get("max_iterations", 30))

        _install_patches()
        import inspect

        import run_agent

        # Resolve the endpoint the SAME way the working `hermes chat -q` (oneshot)
        # path does: hermes_cli.runtime_provider.resolve_runtime_provider(requested=
        # "agent") returns {base_url, api_key, provider, api_mode, credential_pool}
        # from ~/.hermes/config.yaml providers.agent. Reusing hermes' own resolver
        # avoids reconstructing the URL/api_mode by hand (earlier hand-rolls hit
        # HTTP 404 then Connection error). Falls back to a plain config read, then env.
        model = os.environ.get("AGENT_MODEL_NAME", "")
        runtime = _resolve_runtime("agent", model)
        base = runtime.get("base_url") or os.environ.get("AGENT_MODEL_BASE", "")
        key = runtime.get("api_key") or os.environ.get("AGENT_MODEL_KEY", "")

        # Version-robust: only pass kwargs the sandbox's hermes AIAgent accepts.
        want = {
            "base_url": base or None,
            "api_key": key or None,
            "provider": runtime.get("provider") or "agent",
            "api_mode": runtime.get("api_mode"),
            "model": model,
            "max_iterations": max_iter,
            "save_trajectories": False,
            "quiet_mode": True,
            "persist_session": False,
            "skip_memory": True,
            "skip_context_files": True,
        }
        try:
            sig_params = set(inspect.signature(run_agent.AIAgent.__init__).parameters)
        except (TypeError, ValueError):
            sig_params = set(want)  # fall back to trying all if introspection fails
        kwargs = {k: v for k, v in want.items() if k in sig_params and v is not None}
        agent = run_agent.AIAgent(**kwargs)
        run_token = f"capture/{os.getpid()}"
        agent._run_token = run_token
        with _CHILD_SINK_LOCK:
            _CHILD_SINK[run_token] = []

        # Multi-turn continuation: pass prior structured messages as history.
        res = agent.run_conversation(
            query, conversation_history=history or None, task_id=run_token
        )

        cap = getattr(agent, "_cap", None) or {}
        with _CHILD_SINK_LOCK:
            child_caps = _CHILD_SINK.pop(run_token, [])

        result_out["messages"] = cap.get("messages") or (res or {}).get("messages") or []
        result_out["children"] = _child_messages(child_caps)
        result_out["ok"] = bool((res or {}).get("completed", True)) and not (res or {}).get("error")
        result_out["error"] = str((res or {}).get("error") or "")
    except Exception as exc:  # fail loud into the output payload
        result_out["ok"] = False
        result_out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Marker prefix so the actor can extract the payload past any hermes noise.
        sys.stdout.write("__CAPTURE__" + json.dumps(result_out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
