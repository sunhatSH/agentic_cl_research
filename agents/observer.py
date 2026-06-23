"""Observer agent (no persona, objective) -- doc §3.3 / §7.3 / O6.

DIFF-DRIVEN observation (2026-06-19). The observer's evidence is a DETERMINISTIC
before/after diff of the agent's environment -- computed by code, not by the
model reading what the actor *claimed*. Two evidence channels:

  - FS diff  : files created/modified/removed THIS turn, WITH content. Binary
               rich formats (xlsx/docx/pptx/pdf) are EXTRACTED to text (a), so
               "slide 3 says X" / "cell B2 = 12345" is checkable, not opaque bytes.
  - SYS diff : non-filesystem effects (b) -- packages installed, ports opened,
               processes started THIS turn (SysOps tasks whose effect is not a
               workspace file).

The actor trajectory is still passed, but only as CLAIMS to cross-check against
the real diff (it fills ``discrepancies``).

Why diff-driven instead of actor-claim-driven (the design's original §3.3 lead):
  1. Hallucination -- the actor's narrative can assert files/values that do not
     exist; only the environment diff is ground truth.
  2. Lost intermediates -- claims state *results* and omit intermediate artifacts
     the design most wants captured; a diff surfaces them regardless.
  3. Anti reward-hacking -- if reward grounds on claims, the policy learns to
     *say* it finished without finishing; grounding on real effect removes that.
  4. Omission, not just commission -- effects the actor never mentioned appear.
  5. Content-level verification -- file/extracted content can be checked.
  6. Determinism -- the diff is produced by code (reproducible).

Observer LLM is OPTIONAL (``use_llm``). Because the forensics layer already
produces TEXT evidence (FS text diff, binary-summarized-to-text, sys diff), the
default path builds the report DETERMINISTICALLY with NO model call -- this is
what avoids pushing raw evidence into the (expensive) reward model and asking it
to "observe". Set ``use_llm=True`` to additionally have a model summarize a large
diff / flag claim-vs-reality gaps. The deterministic FORENSICS layer always runs.

The probes use ONLY ``sandbox.run_code`` (the backend-agnostic interface), so the
observer works identically on the local / Tencent E2B / Alibaba AgentBay backends.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from agents.base import ChatClient, TruncatedOutputError, resolve_observer_client
from agents.prompts import build_observer_prompt
from agents.schema import ObservationReport


def flatten_trajectory(messages: Sequence[dict[str, Any]] | str) -> str:
    """Flatten winner messages into trajectory text (pass-through to reward).

    Accepts a message list or an already-flattened string. This is carried by the
    observer COMPONENT for reward; it is NEVER put into the observer LLM prompt.
    """
    if isinstance(messages, str):
        return messages
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join((c.get("text", "") if isinstance(c, dict) else str(c)) for c in content)
        lines.append(f"[{m.get('role', '?')}] {str(content).strip()}")
    return "\n".join(lines)


# Rendering caps for the prompt / report (#4): bound size regardless of workspace.
_MAX_RENDER_FILES = 50
_MAX_RENDER_CHARS = 1200
# Rich binary formats we extract to text in the sandbox (a). Libs are in the
# sandbox image (docker/sandbox §5: openpyxl / python-docx / python-pptx / pdfplumber).
_RICH_EXTS = (".xlsx", ".xlsm", ".docx", ".pptx", ".pdf")


class ReadOnlySandbox(Protocol):
    """Read-only surface the observer is allowed on the live winner instance."""

    def run_code(self, code: str, language: str = "python") -> Any:
        """Run a read-only probe; returns an object with .stdout / .stderr."""
        ...


# --------------------------------------------------------------------------- #
# FS snapshot probe -- cheap: os.stat + small text head; binaries only MARKED   #
# (kind=binary + ext). Extraction of changed binaries happens later, on demand. #
# --------------------------------------------------------------------------- #
_SNAPSHOT_PROBE = (
    "import os, json, hashlib\n"
    "ROOT='.'; MAX_TEXT=2048; MAX_FILES=200\n"
    "SKIP={'.git','__pycache__','node_modules','.cache','.ipynb_checkpoints','.venv'}\n"
    "out={}\n"
    "for root, dirs, files in os.walk(ROOT):\n"
    "    if root.count(os.sep) > 5:\n"
    "        dirs[:]=[]; continue\n"
    "    dirs[:]=[d for d in dirs if d not in SKIP]\n"
    "    for fn in files:\n"
    "        p=os.path.join(root, fn)\n"
    "        try:\n"
    "            st=os.stat(p)\n"
    "        except OSError:\n"
    "            continue\n"
    "        rec={'size': st.st_size, 'mtime': round(st.st_mtime, 3),\n"
    "             'ext': os.path.splitext(fn)[1].lower()}\n"
    "        try:\n"
    "            with open(p,'rb') as f:\n"
    "                raw=f.read(MAX_TEXT)\n"
    "            # content_hash for same-size change detection (≤4 KB)\n"
    "            if st.st_size <= 4096:\n"
    "                rec['chash']=hashlib.md5(raw).hexdigest()[:12]\n"
    "            try:\n"
    "                rec['text']=raw.decode('utf-8'); rec['truncated']=st.st_size>MAX_TEXT\n"
    "            except UnicodeDecodeError:\n"
    "                rec['binary']=True\n"
    "        except OSError:\n"
    "            pass\n"
    "        out[p]=rec\n"
    "        if len(out) >= MAX_FILES:\n"
    "            break\n"
    "    if len(out) >= MAX_FILES:\n"
    "        break\n"
    "print(json.dumps(out))\n"
)

# --------------------------------------------------------------------------- #
# System-state probe (b) -- non-FS effects: installed packages, listening TCP   #
# ports, process names. Each part is best-effort (try/except); on a non-Linux   #
# dev box /proc parts simply come back empty. NO env values (secret hygiene).   #
# --------------------------------------------------------------------------- #
_SYS_PROBE = (
    "import json\n"
    "out={'pip':{}, 'ports':[], 'procs':[]}\n"
    "try:\n"
    "    import importlib.metadata as m\n"
    "    out['pip']={d.metadata['Name'].lower(): d.version for d in m.distributions()}\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    ports=set()\n"
    "    with open('/proc/net/tcp') as f:\n"
    "        next(f)\n"
    "        for line in f:\n"
    "            parts=line.split()\n"
    "            if len(parts)>3 and parts[3]=='0A':\n"  # 0A = LISTEN
    "                ports.add(int(parts[1].split(':')[1], 16))\n"
    "    out['ports']=sorted(ports)\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    import os\n"
    "    names=set()\n"
    "    for pid in os.listdir('/proc'):\n"
    "        if not pid.isdigit():\n"
    "            continue\n"
    "        try:\n"
    "            with open('/proc/%s/comm' % pid) as f:\n"
    "                names.add(f.read().strip())\n"
    "        except OSError:\n"
    "            continue\n"
    "    out['procs']=sorted(names)\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps(out))\n"
)


def _extract_probe(paths: list[str]) -> str:
    """Build a probe that extracts text from the given rich-binary files (a).

    Runs in the sandbox; per file dispatches by extension to the office/pdf lib,
    bounding output. Any failure (missing lib, parse error) -> a short marker, so
    the snapshot/diff never crashes and degrades to "binary, not extracted".
    """
    return (
        "import os, json\n"
        "PATHS=" + json.dumps(paths) + "\n"
        "CAP=2000\n"
        "def x_xlsx(p):\n"
        "    import openpyxl\n"
        "    wb=openpyxl.load_workbook(p, read_only=True, data_only=True)\n"
        "    o=[]\n"
        "    for ws in wb.worksheets:\n"
        "        o.append('# sheet %s' % ws.title)\n"
        "        for i,row in enumerate(ws.iter_rows(values_only=True)):\n"
        "            if i>=30: o.append('  …'); break\n"
        "            o.append('  '+', '.join('' if c is None else str(c) for c in row))\n"
        "    return '\\n'.join(o)\n"
        "def x_docx(p):\n"
        "    import docx\n"
        "    return '\\n'.join(par.text for par in docx.Document(p).paragraphs if par.text)\n"
        "def x_pptx(p):\n"
        "    from pptx import Presentation\n"
        "    o=[]\n"
        "    for i,s in enumerate(Presentation(p).slides, 1):\n"
        "        o.append('# slide %d' % i)\n"
        "        for sh in s.shapes:\n"
        "            if getattr(sh,'has_text_frame',False) and sh.text_frame.text.strip():\n"
        "                o.append('  '+sh.text_frame.text.strip())\n"
        "    return '\\n'.join(o)\n"
        "def x_pdf(p):\n"
        "    import pdfplumber\n"
        "    o=[]\n"
        "    with pdfplumber.open(p) as pdf:\n"
        "        for i,pg in enumerate(pdf.pages):\n"
        "            if i>=3: o.append('…'); break\n"
        "            o.append(pg.extract_text() or '')\n"
        "    return '\\n'.join(o)\n"
        "DISP={'.xlsx':x_xlsx,'.xlsm':x_xlsx,'.docx':x_docx,'.pptx':x_pptx,'.pdf':x_pdf}\n"
        "out={}\n"
        "for p in PATHS:\n"
        "    ext=os.path.splitext(p)[1].lower()\n"
        "    fn=DISP.get(ext)\n"
        "    if fn is None:\n"
        "        continue\n"
        "    try:\n"
        "        t=fn(p)\n"
        "        out[p]=t[:CAP]+(' …[truncated]' if len(t)>CAP else '')\n"
        "    except Exception as e:\n"
        "        out[p]='[%s: extract failed: %s]' % (ext, type(e).__name__)\n"
        "print(json.dumps(out))\n"
    )


def _run_json_probe(sandbox: ReadOnlySandbox | None, probe: str) -> dict:
    """Run a probe that prints one JSON object; return it or {} on any failure."""
    if sandbox is None:
        return {}
    try:
        res = sandbox.run_code(probe)
        out = (getattr(res, "stdout", "") or "").strip()
        # Guard against oversized stdout (1 MB limit)
        if len(out) > 1_000_000:
            out = out[:1_000_000]
        data = json.loads(out) if out else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- observation must never crash the session
        return {}


def snapshot_workspace(sandbox: ReadOnlySandbox | None) -> dict[str, dict]:
    """Read-only FS snapshot: ``{path: {size, mtime, ext, text|binary}}``."""
    return _run_json_probe(sandbox, _SNAPSHOT_PROBE)


def snapshot_system(sandbox: ReadOnlySandbox | None) -> dict:
    """Read-only system-state snapshot: ``{pip:{name:ver}, ports:[...], procs:[...]}``."""
    return _run_json_probe(sandbox, _SYS_PROBE)


def extract_binaries(sandbox: ReadOnlySandbox | None, paths: list[str]) -> dict[str, str]:
    """Extract text from rich-binary files (a). Empty when no sandbox / nothing to do."""
    rich = [p for p in paths if p.lower().endswith(_RICH_EXTS)]
    if sandbox is None or not rich:
        return {}
    data = _run_json_probe(sandbox, _extract_probe(rich))
    return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _excerpt(rec: dict) -> dict:
    if rec.get("binary"):
        return {"kind": "binary", "size": rec.get("size", 0), "ext": rec.get("ext", "")}
    return {
        "kind": "text",
        "size": rec.get("size", 0),
        "content_excerpt": rec.get("text", ""),
        "truncated": bool(rec.get("truncated")),
    }


def _sig(rec: dict) -> tuple:
    """Cheap change signature: (size, mtime, chash). Any write updates mtime;
    chash catches same-size content changes for small files."""
    return (rec.get("size"), rec.get("mtime"), rec.get("chash"))


def diff_snapshots(pre: dict[str, dict] | None, post: dict[str, dict] | None) -> dict[str, list]:
    """Deterministic FS diff: added / modified / removed (change by size+mtime)."""
    pre = pre or {}
    post = post or {}
    added: list[dict] = []
    modified: list[dict] = []
    removed: list[dict] = []
    for path, rec in post.items():
        if path not in pre:
            added.append({"path": path, **_excerpt(rec)})
        elif _sig(pre[path]) != _sig(rec):
            entry = {"path": path, **_excerpt(rec)}
            entry["before_excerpt"] = pre[path].get("text", "")
            modified.append(entry)
    for path in pre:
        if path not in post:
            entry = {"path": path}
            excerpt = pre[path].get("text", "")
            if excerpt:
                entry["before_excerpt"] = excerpt[:200]
            removed.append(entry)
    return {"added": added, "modified": modified, "removed": removed}


def diff_system(pre: dict | None, post: dict | None) -> dict[str, list]:
    """Deterministic non-FS diff: packages installed, ports opened, procs started."""
    pre = pre or {}
    post = post or {}
    pre_pip, post_pip = pre.get("pip", {}) or {}, post.get("pip", {}) or {}
    installed = [f"{n}=={v}" for n, v in sorted(post_pip.items()) if n not in pre_pip]
    pre_ports, post_ports = set(pre.get("ports", [])), set(post.get("ports", []))
    pre_procs, post_procs = set(pre.get("procs", [])), set(post.get("procs", []))
    return {
        "installed_packages": installed,
        "opened_ports": sorted(post_ports - pre_ports),
        "started_procs": sorted(post_procs - pre_procs),
    }


def _merge_extracted(diff: dict[str, list], extracted: dict[str, str]) -> None:
    """Fold extracted binary text (a) back into the diff entries as content."""
    for group in ("added", "modified"):
        for f in diff[group]:
            if f["path"] in extracted:
                f["kind"] = "binary→text"
                f["content_excerpt"] = extracted[f["path"]]


def _fs_diff_empty(diff: dict[str, list]) -> bool:
    return not any(diff.get(k) for k in ("added", "modified", "removed"))


def _sys_diff_empty(sd: dict[str, list]) -> bool:
    return not any(sd.get(k) for k in ("installed_packages", "opened_ports", "started_procs"))


def _render_file(prefix: str, f: dict) -> str:
    head = f"{prefix} {f['path']} ({f.get('kind', '?')}, {f.get('size', 0)}B)"
    body = f.get("content_excerpt", "")
    if not body:
        return head
    if len(body) > _MAX_RENDER_CHARS:
        body = body[:_MAX_RENDER_CHARS] + " …[truncated]"
    indented = "\n".join("    " + ln for ln in body.splitlines())
    return f"{head}\n{indented}"


def _format_changes(diff: dict[str, list], sys_diff: dict[str, list]) -> str:
    """Render FS + SYS diff as the observer's ground-truth evidence (capped)."""
    fs_empty, sys_empty = _fs_diff_empty(diff), _sys_diff_empty(sys_diff)
    if fs_empty and sys_empty:
        return "mode: before/after diff -- (no filesystem or system changes this turn)"
    out = ["mode: before/after diff (what THIS turn changed in the environment)"]
    shown = 0
    for f in diff["added"]:
        if shown >= _MAX_RENDER_FILES:
            break
        out.append(_render_file("+ ADDED", f))
        shown += 1
    for f in diff["modified"]:
        if shown >= _MAX_RENDER_FILES:
            break
        out.append(_render_file("~ MODIFIED", f))
        # Show the before-content so the judge can see what changed FROM
        before = f.get("before_excerpt", "")
        if before:
            if len(before) > _MAX_RENDER_CHARS:
                before = before[:_MAX_RENDER_CHARS] + " …[truncated]"
            indented = "\n".join("  [was: " + ln for ln in before.splitlines())
            out.append(indented)
        shown += 1
    for f in diff["removed"]:
        out.append(f"- REMOVED {f['path']}")
        # Show a brief excerpt of what was removed (from baseline snapshot)
        was = f.get("before_excerpt", "")
        if was:
            snippet = was[:200]
            if len(was) > 200:
                snippet += "…"
            out.append(f"    [was: {snippet}]")
    total = len(diff["added"]) + len(diff["modified"])
    if total > _MAX_RENDER_FILES:
        out.append(f"… and {total - _MAX_RENDER_FILES} more changed files (truncated)")
    if not sys_empty:
        out.append("## system state changes (non-filesystem)")
        for pkg in sys_diff["installed_packages"]:
            out.append(f"+ INSTALLED {pkg}")
        for port in sys_diff["opened_ports"]:
            out.append(f"+ PORT LISTENING {port}")
        for proc in sys_diff["started_procs"]:
            out.append(f"+ PROCESS {proc}")
    return "\n".join(out)


def _format_state(post: dict[str, dict]) -> str:
    """Render the current workspace (no baseline) as content-level evidence (capped)."""
    if not post:
        return ""
    out = ["mode: current workspace snapshot (no baseline -- content-level evidence)"]
    for p, r in sorted(post.items())[:_MAX_RENDER_FILES]:
        out.append(_render_file("•", {"path": p, **_excerpt(r)}))
    if len(post) > _MAX_RENDER_FILES:
        out.append(f"… and {len(post) - _MAX_RENDER_FILES} more files (truncated)")
    return "\n".join(out)


def build_deterministic_report(
    *,
    diff: dict[str, list] | None,
    file_tree: str,
    state_diff: str,
) -> ObservationReport:
    """Build R_t from the diff with NO model call (observer LLM optional).

    Realized artifacts (added/modified files, incl. binary→text) go into ``final``.
    The report carries STATE only (no ``actor_claims`` -- the observer never sees
    the trajectory); the actor trajectory is given to the reward judge DIRECTLY,
    not via this report.
    """
    final: list[dict] = []
    intermediate: list[dict] = []
    discrepancies_parts: list[str] = []

    if diff is not None:
        # Track paths that were added then modified within the same turn → intermediate
        added_paths: dict[str, dict] = {}
        modified_paths: set[str] = set()
        for f in diff["added"]:
            added_paths[f["path"]] = f
        for f in diff["modified"]:
            modified_paths.add(f["path"])

        for f in diff["added"]:
            path = f["path"]
            entry = {
                "path": path,
                "kind": f.get("kind", "?"),
                "content_excerpt": f.get("content_excerpt", ""),
            }
            # If this file was later modified in the same turn, it's intermediate
            if path in modified_paths:
                intermediate.append(
                    {
                        "desc": "intermediate file (modified later in same turn)",
                        "source": path,
                        "value_excerpt": f.get("content_excerpt", "")[:200],
                    }
                )
            else:
                final.append(entry)

        for f in diff["modified"]:
            path = f["path"]
            if path not in added_paths:  # don't double-count
                final.append(
                    {
                        "path": path,
                        "kind": f.get("kind", "?"),
                        "content_excerpt": f.get("content_excerpt", ""),
                    }
                )

        # Structural discrepancy checks (no LLM needed)
        for f in diff["added"] + diff["modified"]:
            path = f["path"]
            kind = f.get("kind", "?")
            size = f.get("size", 0)
            content = f.get("content_excerpt", "")
            # Derive ext from path (the diff entry may not carry 'ext')
            ext = ""
            if "." in path.rsplit("/", 1)[-1]:
                ext = "." + path.rsplit(".", 1)[-1].lower()

            # Empty deliverable: file with office/data extension but size=0 or empty content
            if ext in (".xlsx", ".xlsm", ".docx", ".pptx", ".pdf", ".csv", ".json", ".xml"):
                if size == 0:
                    discrepancies_parts.append(f"{path}: empty file (size=0, ext={ext})")
                elif kind == "text" and (not content or content.strip() == ""):
                    discrepancies_parts.append(f"{path}: non-zero size but empty text content")
            if kind == "binary→text" and (not content or content.strip() == ""):
                discrepancies_parts.append(f"{path}: binary extracted but content is empty")

    # Cross-file value consistency check (simple: look for same key name with different values)
    # This is a best-effort heuristic — the LLM path does deeper semantic checks.
    _check_value_consistency(diff, discrepancies_parts)

    return ObservationReport(
        intermediate=intermediate,
        final=final,
        discrepancies="; ".join(discrepancies_parts) if discrepancies_parts else "",
        file_tree=file_tree,
        state_diff=state_diff,
    )


def _check_value_consistency(diff: dict[str, list] | None, discrepancies: list[str]) -> None:
    """Heuristic cross-file value consistency check.

    Scans added/modified CSV-like content for rows with the same key column
    but different values across files. Very conservative — only flags when
    the same header value appears in two files with a different second column.
    """
    if diff is None:
        return
    # Collect key→value pairs from CSV-like content across files
    file_values: dict[str, dict[str, str]] = {}
    for f in diff["added"] + diff["modified"]:
        content = f.get("content_excerpt", "")
        if not content or f.get("kind") == "binary":
            continue
        values: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip().lstrip(" ,")
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                values[parts[0]] = parts[1]
        if values:
            file_values[f["path"]] = values
    # Check for conflicts
    all_keys: dict[str, list[tuple[str, str]]] = {}
    for path, vals in file_values.items():
        for k, v in vals.items():
            all_keys.setdefault(k, []).append((path, v))
    for k, entries in all_keys.items():
        unique_vals = set(v for _, v in entries)
        if len(unique_vals) > 1 and len(entries) > 1:
            paths = ", ".join(f"{p}={v}" for p, v in entries)
            discrepancies.append(f"conflicting value for '{k}': {paths}")


# --------------------------------------------------------------------------- #
# Observer tools (OpenAI function-calling schema) for multi-turn tool-use.      #
# Each tool wraps a sandbox probe that is backend-agnostic (uses run_code only). #
# --------------------------------------------------------------------------- #


def _read_file_probe(path: str) -> str:
    """Build a probe to read a single file's content (capped at 4 KB)."""
    return (
        "import json\n"
        f"PATH={json.dumps(path)}\n"
        "try:\n"
        "    with open(PATH) as f:\n"
        "        data=f.read(4096)\n"
        "    print(json.dumps({'path': PATH, 'content': data, 'truncated': len(data)>=4096}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'path': PATH, 'error': str(e)}))\n"
    )


def _list_dir_probe(path: str) -> str:
    """Build a probe to list a directory."""
    return (
        "import os, json\n"
        f"PATH={json.dumps(path)}\n"
        "try:\n"
        "    entries=sorted(os.listdir(PATH))\n"
        "    print(json.dumps({'path': PATH, 'entries': entries}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'path': PATH, 'error': str(e)}))\n"
    )


OBSERVER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_diff",
            "description": (
                "Get the before/after environment diff that was auto-collected "
                "for this turn. Includes files added/modified/removed (with content) "
                "and system-state changes (packages, ports, processes)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_tree",
            "description": (
                "Get the full workspace file list with content excerpts. "
                "Use when you need to see ALL files (not just changed ones)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a specific file's content (up to 4 KB). Use to inspect a "
                "file in detail when the diff excerpt is truncated or suspicious."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to read (relative to workspace root, e.g. './report.txt').",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List contents of a directory. Use to discover files not shown " "in the diff or file tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (e.g. './output'). Defaults to workspace root.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_state",
            "description": (
                "Get the current system state: installed pip packages, "
                "listening TCP ports, running processes. Use when you suspect "
                "system-level changes not visible in the file diff."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _execute_observer_tool(
    tool_name: str,
    tool_args: dict,
    sandbox: ReadOnlySandbox | None,
    state_diff: str,
    file_tree: str,
    baseline: dict | None,
    post: dict | None,
) -> str:
    """Execute a single observer tool call and return the result as a string."""
    if sandbox is None:
        return json.dumps({"error": "no sandbox available"})

    if tool_name == "get_diff":
        return state_diff or "(no diff available)"

    if tool_name == "get_file_tree":
        return file_tree or "(empty workspace)"

    if tool_name == "read_file":
        path = tool_args.get("path", ".")
        result = _run_json_probe(sandbox, _read_file_probe(path))
        return json.dumps(result, ensure_ascii=False) if result else json.dumps({"error": "read failed"})

    if tool_name == "list_dir":
        path = tool_args.get("path", ".")
        result = _run_json_probe(sandbox, _list_dir_probe(path))
        return json.dumps(result, ensure_ascii=False) if result else json.dumps({"error": "list failed"})

    if tool_name == "get_system_state":
        sys_state = snapshot_system(sandbox)
        return json.dumps(sys_state, ensure_ascii=False, default=str)

    return json.dumps({"error": f"unknown tool: {tool_name}"})


class Observer:
    """Objective state observer. Persona-free (§3.3), diff-driven, LLM optional."""

    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        use_llm: bool = True,
        probe_system: bool = True,
        max_tokens: int = 1024,
        max_tool_rounds: int = 5,
    ):
        self._client = client
        self._use_llm = use_llm
        self._probe_system = probe_system
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = resolve_observer_client()
        return self._client

    def snapshot(self, sandbox: ReadOnlySandbox | None) -> dict[str, Any]:
        """Capture a {fs, sys} bundle; pass forward as next turn's ``baseline``/``post``."""
        return {
            "fs": snapshot_workspace(sandbox),
            "sys": snapshot_system(sandbox) if self._probe_system else {},
        }

    @staticmethod
    def _fs(bundle: dict | None) -> dict:
        """Accept a {fs,sys} bundle or a legacy plain-fs dict."""
        if not bundle:
            return {}
        return bundle.get("fs", bundle) if isinstance(bundle, dict) else {}

    @staticmethod
    def _sys(bundle: dict | None) -> dict:
        return bundle.get("sys", {}) if isinstance(bundle, dict) else {}

    def observe(
        self,
        sandbox: ReadOnlySandbox | None = None,
        *,
        actor_trajectory: Sequence[dict[str, Any]] | str = "",
        baseline: dict | None = None,
        post: dict | None = None,
    ) -> ObservationReport:
        """Diff-driven objective report with optional multi-turn tool-use.

        The observer MODEL sees STATE only; the trajectory is carried PASS-THROUGH.

        Flow:
          1. Deterministic forensics (always runs): snapshot → diff → extract → render.
          2. If use_llm=False → build deterministic report (degraded mode).
          3. If use_llm=True → LLM multi-turn tool-use loop:
             - Send diff evidence as the first user message.
             - LLM may call tools (read_file, list_dir, …) to investigate.
             - Loop until LLM outputs final JSON or max rounds reached.
             - On any failure → degrade to deterministic report.

        Args:
            sandbox: live (winner) instance; used for tool execution and binary
                extraction, and snapshotted for ``post`` if not supplied.
            actor_trajectory: winner messages (or text) -- carried through to reward,
                NOT given to the observer model.
            baseline: pre-turn {fs,sys} snapshot. With it -> before/after diff.
            post: post-turn {fs,sys} snapshot. Snapshotted here only if not supplied.
        """
        traj_text = flatten_trajectory(actor_trajectory)
        if post is None:
            post = self.snapshot(sandbox) if sandbox is not None else {"fs": {}, "sys": {}}
        post_fs = self._fs(post)
        file_tree = "\n".join(sorted(post_fs.keys()))

        diff: dict[str, list] | None = None
        if baseline is not None:
            diff = diff_snapshots(self._fs(baseline), post_fs)
            sys_diff = diff_system(self._sys(baseline), self._sys(post))
            # (a) extract changed rich-binary files to text, fold into the diff.
            changed = [f["path"] for f in diff["added"] + diff["modified"] if f.get("kind") == "binary"]
            if changed:
                _merge_extracted(diff, extract_binaries(sandbox, changed))
            if _fs_diff_empty(diff) and _sys_diff_empty(sys_diff):
                # nothing changed on disk OR in system state this turn -> gate
                return ObservationReport(
                    file_tree=file_tree,
                    actor_trajectory=traj_text,
                    state_diff="mode: before/after diff -- (no filesystem or system changes this turn)",
                    has_effect=False,
                )
            state_diff = _format_changes(diff, sys_diff)
        else:
            state_diff = _format_state(post_fs)

        # ------------------------------------------------------------------ #
        # Deterministic report (degraded mode when LLM unavailable)          #
        # ------------------------------------------------------------------ #
        if not self._use_llm:
            report = build_deterministic_report(diff=diff, file_tree=file_tree, state_diff=state_diff)
            report.actor_trajectory = traj_text
            return report

        # ------------------------------------------------------------------ #
        # LLM multi-turn tool-use loop                                       #
        # ------------------------------------------------------------------ #
        report = self._tool_use_loop(
            sandbox=sandbox,
            diff=diff,
            state_diff=state_diff,
            file_tree=file_tree,
            baseline=baseline,
            post=post,
        )
        report.actor_trajectory = traj_text
        return report

    def _tool_use_loop(
        self,
        sandbox: ReadOnlySandbox | None,
        diff: dict[str, list] | None,
        state_diff: str,
        file_tree: str,
        baseline: dict | None,
        post: dict | None,
    ) -> ObservationReport:
        """Run the LLM with tool-use until it outputs a final JSON report.

        Falls back to the deterministic report on any failure (LLM error, parse
        failure, max rounds exceeded). Never crashes the session.
        """
        try:
            messages = build_observer_prompt(state_diff=state_diff, file_tree=file_tree)
            # Check if the client supports tool-use
            client = self.client
            has_tool_support = hasattr(client, "chat_with_tools")

            for _round in range(self._max_tool_rounds):
                if has_tool_support:
                    msg = client.chat_with_tools(messages, tools=OBSERVER_TOOLS, max_tokens=self._max_tokens)
                else:
                    # Fallback: single-shot without tools
                    content = client.chat(messages, max_tokens=self._max_tokens)
                    msg = {"role": "assistant", "content": content}

                messages.append(msg)

                # Check if the LLM wants to call tools
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    # No tool calls — this should be the final JSON output
                    content = msg.get("content", "")
                    report = parse_observation_report(
                        content, fallback_tree=file_tree, fallback_diff=state_diff
                    )
                    report.file_tree = file_tree
                    report.state_diff = state_diff
                    return report

                # Process tool calls
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    try:
                        tool_args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                    tool_result = _execute_observer_tool(
                        tool_name,
                        tool_args,
                        sandbox,
                        state_diff,
                        file_tree,
                        baseline,
                        post,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": tool_result,
                        }
                    )

            # Max rounds exceeded — ask for the final report without tools
            messages.append(
                {
                    "role": "user",
                    "content": "You have used all available investigation rounds. "
                    "Output the JSON observation report now based on the evidence gathered.",
                }
            )
            if has_tool_support:
                final_msg = client.chat_with_tools(
                    messages, max_tokens=self._max_tokens
                )  # no tools → plain completion
            else:
                content = client.chat(messages, max_tokens=self._max_tokens)
                final_msg = {"role": "assistant", "content": content}
            content = final_msg.get("content", "")
            report = parse_observation_report(content, fallback_tree=file_tree, fallback_diff=state_diff)
            report.file_tree = file_tree
            report.state_diff = state_diff
            return report

        except TruncatedOutputError:
            # The model's reply was cut off (thinking models hitting max_tokens).
            # Do NOT parse the half-JSON as the final report -- degrade to the
            # deterministic forensics report (which is always correct/complete).
            return build_deterministic_report(diff=diff, file_tree=file_tree, state_diff=state_diff)
        except Exception:  # noqa: BLE001 — never crash the session
            # Degrade to deterministic report on any LLM failure
            return build_deterministic_report(diff=diff, file_tree=file_tree, state_diff=state_diff)


def parse_observation_report(
    text: str,
    *,
    fallback_tree: str = "",
    fallback_diff: str = "",
) -> ObservationReport:
    """Robustly parse the observer's JSON into an ObservationReport.

    Falls back to a minimal report (tree / diff) when the model output is not valid
    JSON. ``state_diff`` is the code-computed evidence and is always carried (the
    model does not emit it). ``actor_trajectory`` is set by the caller (pass-through).
    """
    obj: Any = None
    if text:
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    obj = json.loads(text[start : end + 1])
                except (TypeError, ValueError):
                    obj = None
    if not isinstance(obj, dict):
        return ObservationReport(file_tree=fallback_tree, state_diff=fallback_diff)

    def _as_list(v: Any) -> list[dict]:
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    return ObservationReport(
        intermediate=_as_list(obj.get("intermediate")),
        final=_as_list(obj.get("final")),
        discrepancies=str(obj.get("discrepancies") or ""),
        file_tree=str(obj.get("file_tree") or fallback_tree),
        state_diff=str(obj.get("state_diff") or fallback_diff),
    )
