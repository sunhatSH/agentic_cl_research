"""Vendor-isolated sandbox client + GRPO group sampling.

doc/SandboxRollout.md §7 calls for a ~50-line ``SandboxClient`` adapter so the
rollout loop does not couple to the Tencent Agent Runtime / E2B SDK. Two
backends implement the SAME surface (``run_code`` / ``kill``):

  - ``LocalSandbox``  -- runs Python in a local subprocess; no network/SDK,
    works on a dev box and in CI. Used to validate the execute+sample loop.
  - ``E2BSandbox``    -- thin wrapper over ``e2b_code_interpreter.Sandbox``
    (needs E2B_API_KEY / E2B_DOMAIN + network); used on the cluster.

GRPO helpers (``grpo_advantages`` / ``select_winner``) implement the group
normalization + winner固化 selection from doc/SandboxRollout.md §3.3.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Protocol

from rollout.sandbox_env import load_sandbox_runtime_env


@dataclass
class ExecResult:
    """Outcome of one code execution in a sandbox."""

    stdout: str
    stderr: str
    ok: bool  # process exited 0 and did not time out


class SandboxClient(Protocol):
    """Minimal surface the rollout loop depends on (matches E2B Sandbox)."""

    def run_code(self, code: str, language: str = "python") -> ExecResult: ...

    def kill(self) -> None: ...


class LocalSandbox:
    """Subprocess-backed sandbox: executes Python locally (dev/CI backend).

    Deliberately NOT a security boundary -- it runs trusted, self-generated
    rollout code on a dev box where the real sandbox is unreachable. Each
    ``run_code`` runs in a fresh temp working directory with a timeout.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._alive = True

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        if language != "python":
            return ExecResult("", f"LocalSandbox supports python only, got {language}", False)
        try:
            with tempfile.TemporaryDirectory() as workdir:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                )
            return ExecResult(proc.stdout, proc.stderr, proc.returncode == 0)
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timeout after {self.timeout}s", False)

    def kill(self) -> None:
        self._alive = False


class E2BSandbox:
    """Tencent Agent Runtime / E2B-compatible sandbox (cluster backend).

    Uses the REST API directly: create returns ``envdAccessToken``; run-code
  host is ``49999-<sandboxID>.<E2B_DOMAIN>`` with header ``X-Access-Token``.
  (The pip ``e2b_code_interpreter`` SDK uses a different host/auth shape on
  Tencent and returns 401 on ``/execute`` without this path.)
    """

    _RUN_CODE_PORT = 49999

    def __init__(self, template: str = "agentic-cl-code-interpreter", timeout: int = 300):
        import os

        import httpx

        api_key = os.environ.get("E2B_API_KEY")
        domain = os.environ.get("E2B_DOMAIN")
        if not api_key or not domain:
            raise RuntimeError("E2B_API_KEY and E2B_DOMAIN must be set for e2b backend")

        self._api_key = api_key
        self._domain = domain
        self._api_url = f"https://api.{domain}"
        self._timeout = timeout
        self._client = httpx.Client()

        runtime_env = load_sandbox_runtime_env()
        resp = self._client.post(
            f"{self._api_url}/sandboxes",
            json={
                "templateID": template,
                "timeout": timeout,
                "metadata": {},
                "envVars": runtime_env,
            },
            headers={"X-API-KEY": api_key},
            timeout=60.0,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"sandbox create failed: {resp.status_code} {resp.text}")

        data = resp.json()
        self._sandbox_id = data["sandboxID"]
        self._full_id = f"{data['sandboxID']}-{data['clientID']}"
        self._envd_token = data["envdAccessToken"]
        self._run_url = f"https://{self._RUN_CODE_PORT}-{self._sandbox_id}.{domain}/execute"

    def run_code(self, code: str, language: str = "python") -> ExecResult:
        import json

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        try:
            with self._client.stream(
                "POST",
                self._run_url,
                json={"code": code, "language": language},
                headers={"X-Access-Token": self._envd_token},
                timeout=float(self._timeout),
            ) as resp:
                if resp.status_code >= 300:
                    body = resp.read().decode(errors="replace")
                    return ExecResult("", body, False)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = evt.get("type")
                    if kind == "stdout":
                        stdout_parts.append(evt.get("text", ""))
                    elif kind == "stderr":
                        stderr_parts.append(evt.get("text", ""))
                    elif kind == "error":
                        stderr_parts.append(str(evt))
        except Exception as exc:  # noqa: BLE001 — surface network/SDK errors to rollout
            return ExecResult("", str(exc), False)

        stdout = "".join(stdout_parts).strip()
        stderr = "".join(stderr_parts).strip()
        return ExecResult(stdout, stderr, not stderr)

    def kill(self) -> None:
        # Delete by the bare sandboxID (the run-code host id), NOT the
        # ``sandboxID-clientID`` form: the platform returns 404 for the latter
        # and the instance leaks. Verified 2026-06-12 against ap-beijing:
        # DELETE /sandboxes/<sandboxID> -> 204 (reclaimed), -<clientID> -> 404.
        try:
            self._client.delete(
                f"{self._api_url}/sandboxes/{self._sandbox_id}",
                headers={"X-API-KEY": self._api_key},
                timeout=30.0,
            )
        finally:
            self._client.close()


def make_sandbox(backend: str = "local", **kwargs) -> SandboxClient:
    """Factory: ``local`` (default, dev/CI) or ``e2b`` (cluster)."""
    if backend == "local":
        return LocalSandbox(timeout=kwargs.get("timeout", 30))
    if backend == "e2b":
        return E2BSandbox(
            template=kwargs.get("template", "agentic-cl-code-interpreter"),
            timeout=kwargs.get("timeout", 300),
        )
    raise ValueError(f"unknown sandbox backend: {backend!r}")


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
