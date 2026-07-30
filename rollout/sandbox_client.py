"""Vendor-isolated sandbox client + GRPO group sampling.

doc/SandboxRollout.md §7 calls for a ~50-line ``SandboxClient`` adapter so the
rollout loop does not couple to any one cloud vendor. The INTERFACE (the
``SandboxClient`` Protocol: ``run_code`` / ``kill``) is decoupled from the
IMPLEMENTATIONS, which register themselves in an open backend registry. The
rollout loop only ever selects a backend by NAME (``make_sandbox(backend=...)``),
so switching cloud vendor is a config change and adding a new vendor is one
``register_backend()`` call -- with NO edit to the rollout loop:

  - ``LocalSandbox``   -- runs Python in a local subprocess; no network/SDK,
    works on a dev box and in CI. Used to validate the execute+sample loop.
  - ``E2BSandbox``     -- E2B-compatible REST backend; on the cluster this is
    the Tencent Agent Runtime (needs E2B_API_KEY / E2B_DOMAIN + network).
  - ``AliyunSandbox``  -- Alibaba Cloud 无影 AgentBay (``wuying-agentbay-sdk``);
    needs AGENTBAY_API_KEY. Drop-in alternative to the Tencent backend.

Register an out-of-tree / future backend with ``register_backend(name, builder)``.

GRPO helpers (``grpo_advantages`` / ``select_winner``) implement the group
normalization + winner固化 selection from doc/SandboxRollout.md §3.3.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rollout.sandbox_env import load_sandbox_runtime_env

# =========================================================================== #
# INTERFACE -- the contract every backend implements. The rollout loop depends  #
# ONLY on this (run_code -> ExecResult, kill); it never imports a vendor class.  #
# =========================================================================== #


@dataclass
class ExecResult:
    """Outcome of one code execution in a sandbox (the return contract).

    ``stdout`` / ``stderr`` are the captured streams; ``ok`` is True iff the code
    ran to completion without error/timeout. Every backend's ``run_code`` returns
    this exact shape so the rollout loop is vendor-agnostic.
    """

    stdout: str
    stderr: str
    ok: bool  # process exited 0 and did not time out


class SandboxClient(Protocol):
    """The sandbox INTERFACE the rollout loop is written against.

    Any backend (local / Tencent E2B / Alibaba AgentBay / future) is swappable so
    long as it provides these two methods -- structural typing, no subclassing
    required. Concrete implementations live below; a backend is selected by NAME
    via ``make_sandbox`` / the registry, so swapping vendors touches no caller.

    Contract:
      - ``run_code(code, language)`` runs ``code`` in the live instance and returns
        an ``ExecResult``. Ordinary user-code failures come back as ``ok=False``
        with the message in ``stderr`` (do not raise for those).
      - ``kill()`` releases the instance; best-effort, must not raise on teardown.
    """

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        ...

    def kill(self) -> None:
        ...


# =========================================================================== #
# IMPLEMENTATIONS -- swappable backends behind the interface above. Local +     #
# E2B (Tencent) are real and in use; other vendors are intentionally left as    #
# stubs (留空) and filled in on the cluster with real SDK/credentials.           #
# =========================================================================== #


class LocalSandbox:
    """Subprocess-backed sandbox: executes Python locally (dev/CI backend).

    Deliberately NOT a security boundary -- it runs trusted, self-generated
    rollout code on a dev box where the real sandbox is unreachable.

    State is PERSISTENT across ``run_code`` calls: every call runs in the SAME
    working directory (created once per instance, removed on ``kill``). This is
    required so that (a) the agent's own multi-step ReAct loop can build on files
    it wrote in an earlier step, and (b) the observer's before/after workspace
    diff can actually see what changed. (Earlier this used a fresh temp dir per
    call, which silently dropped all state -- see CLAUDE.md TODO#5.)
    """

    def __init__(self, timeout: int = 30, workdir: str | None = None):
        self.timeout = timeout
        self._alive = True
        self._workdir = workdir or tempfile.mkdtemp(prefix="clsbx-")
        self._owns_workdir = workdir is None

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        if language != "python":
            return ExecResult("", f"LocalSandbox supports python only, got {language}", False)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self._workdir,
            )
            return ExecResult(proc.stdout, proc.stderr, proc.returncode == 0)
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timeout after {self.timeout}s", False)

    def kill(self) -> None:
        self._alive = False
        if self._owns_workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)


class E2BSandbox:
    """Tencent Agent Runtime / E2B-compatible sandbox (cluster backend).

    Uses the official ``e2b_code_interpreter`` SDK against the Tencent
    Agent Runtime E2B-compatible endpoint (``E2B_DOMAIN``). The SDK handles
    sandbox connect + envd/traffic token acquisition internally, so we just
    ``Sandbox.create(template=...)`` and drive execution.

    Code execution goes through ``commands.run`` (envd process gRPC on port
    49983), NOT ``run_code`` (Jupyter ``/execute`` on port 49999): the base
    ``sandbox-code`` image does not ship a Jupyter kernel, so
    ``POST 49999-{sid}/execute`` returns 500 (openresty). ``commands.run``
    only needs a shell (base image ships ``/bin/sh``) and is verified working
    on 2026-06-26 (see doc/sandbox/Sandbox_规格与run_code踩坑.md §四).
    """

    def __init__(self, template: str = "agentic-cl-sandbox", timeout: int = 600):
        import os

        api_key = os.environ.get("E2B_API_KEY")
        domain = os.environ.get("E2B_DOMAIN")
        if not api_key or not domain:
            raise RuntimeError("E2B_API_KEY and E2B_DOMAIN must be set for e2b backend")

        # 注:不动 e2b 连接池/HTTP2 设置 —— 用 SDK 原生默认(keepalive=20,复用长连接,
        # 短任务省资源/低延迟)。GOAWAY(入口网关单连接 ~1000 stream 后回收)只在【长任务】
        # 触发,短任务遇不到。长任务需要时再按需开:E2B_MAX_KEEPALIVE_CONNECTIONS 调大分散
        # stream,或对 RemoteProtocolError 重试一次。当前采集/训练以短任务为主,不改。

        # The e2b SDK reads E2B_API_KEY / E2B_DOMAIN from env itself; they are
        # already set (validated above), so create() picks the Tencent endpoint.
        from e2b_code_interpreter import Sandbox

        self._timeout = timeout
        runtime_env = load_sandbox_runtime_env() or None
        self._sb = Sandbox.create(
            template=template,
            timeout=timeout,
            envs=runtime_env,
        )
        self._sandbox_id = self._sb.sandbox_id

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        import shlex

        if language != "python":
            return ExecResult("", f"E2BSandbox supports python only, got {language}", False)
        cmd = f"python3 -c {shlex.quote(code)}"
        try:
            out = self._sb.commands.run(cmd, timeout=self._timeout, cwd="/tmp")
        except Exception as exc:
            return ExecResult("", str(exc), False)
        stdout = (out.stdout or "").strip()
        stderr = (out.stderr or "").strip()
        return ExecResult(stdout, stderr, out.exit_code == 0)

    def kill(self) -> None:
        # The SDK reclaims the instance via its own DELETE /sandboxes/<sandboxID>;
        # best-effort teardown (must not raise).
        try:
            self._sb.kill()
        except Exception:  # noqa: BLE001
            pass


class AliyunSandbox:
    """Alibaba Cloud 无影 AgentBay backend -- IMPLEMENTATION STUB (留空).

    This is the worked EXAMPLE of "another vendor": it shows exactly what a new
    backend must provide to satisfy the ``SandboxClient`` interface (``run_code``
    -> ``ExecResult``, ``kill``), but the body is intentionally left blank. The
    interface/registry split is the deliverable; the vendor wiring is filled in
    on the cluster with real SDK + credentials, not guessed here.

    To implement (回集群再做): back it with the official ``wuying-agentbay-sdk``
    (``AGENTBAY_API_KEY``), following the AgentBay CodeSpace lifecycle
    ``AgentBay(api_key) -> create(CreateSessionParams(image_id="code_latest"))
    -> session.code.run_code(code, language, timeout_s) -> agent_bay.delete(session)``.
    ``session.file_system`` / ``session.command.execute_command`` are also
    available for the observer's read-only state probing (see CLAUDE.md TODO#5).
    """

    def __init__(self, image_id: str = "code_latest", timeout: int = 600):
        raise NotImplementedError(
            "AliyunSandbox is a stub (留空): the SandboxClient interface + registry are "
            "ready, only this vendor body is unimplemented. Fill it in on the cluster "
            "with wuying-agentbay-sdk + AGENTBAY_API_KEY."
        )

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Backend registry: the SandboxClient INTERFACE is decoupled from the concrete #
# IMPLEMENTATIONS. A new vendor plugs in via ``register_backend`` WITHOUT       #
# touching the rollout loop; ``make_sandbox`` only resolves a name -> builder.  #
# --------------------------------------------------------------------------- #

SandboxBuilder = Callable[..., SandboxClient]
_BACKENDS: dict[str, SandboxBuilder] = {}


def register_backend(name: str, builder: SandboxBuilder) -> None:
    """Register (or override) a sandbox backend by name.

    ``builder(**kwargs)`` receives the keyword args passed to ``make_sandbox``
    (e.g. ``template`` / ``image_id`` / ``timeout``; unknown kwargs are ignored)
    and returns a SandboxClient. Out-of-tree backends can call this at import time.
    """
    _BACKENDS[name] = builder


def make_sandbox(backend: str = "local", **kwargs) -> SandboxClient:
    """Resolve a registered backend by name.

    Built-ins: ``local`` (dev/CI), ``e2b`` (Tencent Agent Runtime), ``aliyun``
    (Alibaba AgentBay). Unknown name -> ValueError listing what is registered.
    """
    try:
        builder = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown sandbox backend: {backend!r}; registered: {sorted(_BACKENDS)}") from None
    return builder(**kwargs)


register_backend("local", lambda **kw: LocalSandbox(timeout=kw.get("timeout", 30)))
register_backend(
    "e2b",
    lambda **kw: E2BSandbox(
        template=kw.get("template", "agentic-cl-sandbox"),
        timeout=kw.get("timeout", 300),
    ),
)
register_backend(
    "aliyun",
    lambda **kw: AliyunSandbox(
        image_id=kw.get("image_id", "code_latest"),
        timeout=kw.get("timeout", 300),
    ),
)


def grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Group-normalized advantages: ``(r - mean) / (std + eps)`` (doc §3.3)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var**0.5
    return [(r - mean) / (std + eps) for r in rewards]


def select_winner(rewards: list[float], trajectory_ids: list[str] | None = None) -> int:
    """Index of the winner = argmax(advantage) == argmax(reward).

    Ties broken by ``trajectory_id`` lexicographic order for deterministic,
    reproducible固化 (doc/SandboxRollout.md §3.3). Falls back to lowest index
    when no ids are given.
    """
    if not rewards:
        raise ValueError("no rewards to select a winner from")
    best = max(rewards)
    candidates = [i for i, r in enumerate(rewards) if r == best]
    if len(candidates) == 1 or trajectory_ids is None:
        return candidates[0]
    return min(candidates, key=lambda i: trajectory_ids[i])
