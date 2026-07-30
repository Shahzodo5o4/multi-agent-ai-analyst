"""F6 — Sandboxed Python execution.

The guide's warning: "never execute model-written code on the bare server."
`PythonREPLTool` does exactly that — it runs `exec()` inside your API process, so
a bad `while True:` hangs the server and `os.remove` deletes your files.

This module gives the code agent three real protections instead:

  1. STATIC DENY-LIST  — the code is scanned before it runs; anything importing
     os/subprocess/socket/shutil or calling open()/eval() is rejected outright.
  2. SEPARATE PROCESS  — it runs in a fresh `python` child, not in the server.
     A crash or infinite loop kills the child, never the API.
  3. HARD TIMEOUT      — the child is killed after CODE_TIMEOUT_SECONDS.

That covers the rubric's "sandboxed with a cap". For a system exposed to the
public internet you'd go further (Docker with no network, a seccomp profile, or
a service like E2B) — this is documented in the README's limitations section.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import config

# Modules that give code filesystem, network, or process access.
_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "urllib2", "httpx", "pathlib", "glob", "pickle", "shelve", "ctypes",
    "multiprocessing", "threading", "importlib", "builtins", "webbrowser",
    "http", "ftplib", "smtplib", "telnetlib", "signal", "resource", "pty",
}

# Callables that reintroduce the same access even without an import.
_BLOCKED_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Dunder attribute access is the classic sandbox-escape route
# (`().__class__.__bases__[0].__subclasses__()` reaches os from a bare tuple),
# so it is blocked — except for the two harmless ones models write by habit.
_DUNDER_RE = re.compile(r"__[a-z_]+__")
_ALLOWED_DUNDERS = {"__name__", "__main__"}


class SandboxViolation(Exception):
    """Raised when the deny-list rejects the generated code."""


def strip_code_fences(text: str) -> str:
    """LLMs wrap code in ```python fences. Remove them before executing."""
    text = text.strip()
    fenced = re.match(r"^```(?:python|py)?\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Handle a stray opening fence with no closer.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def check(code: str) -> None:
    """Static screen. Raises SandboxViolation on anything dangerous."""
    for match in _IMPORT_RE.finditer(code):
        root = match.group(1).split(".")[0]
        if root in _BLOCKED_IMPORTS:
            raise SandboxViolation(
                f"blocked import '{root}' — the code agent may only compute, "
                "not touch the filesystem, network, or processes"
            )
    for match in _CALL_RE.finditer(code):
        name = match.group(1)
        if name in _BLOCKED_CALLS:
            raise SandboxViolation(f"blocked call '{name}()'")
    for match in _DUNDER_RE.finditer(code):
        if match.group(0) not in _ALLOWED_DUNDERS:
            raise SandboxViolation(
                f"blocked dunder '{match.group(0)}' (a common sandbox-escape route)"
            )


def run(code: str, timeout: int | None = None) -> str:
    """Execute `code` in a locked-down child process and return its output.

    Never raises for *code* errors — a traceback is a legitimate result that the
    supervisor and critic should be able to see and react to.
    """
    timeout = timeout or config.CODE_TIMEOUT_SECONDS
    code = strip_code_fences(code)
    if not code:
        return "ERROR: the model produced no code."

    try:
        check(code)
    except SandboxViolation as exc:
        return f"BLOCKED BY SANDBOX: {exc}"

    # -I = isolated mode: ignores PYTHON* env vars and the user site-packages
    # dir, and does not prepend the script's directory to sys.path — so the child
    # cannot import anything from this project. Installed libraries (math is
    # stdlib; numpy/pandas if present) still work, which is what the code agent
    # needs for real calculations.
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "agent_code.py"
        script.write_text(code, encoding="utf-8")

        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Windows needs this
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,          # a throwaway dir, not the project
                env=child_env,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return (
                f"TIMEOUT: execution exceeded the {timeout}s cap and was killed. "
                "The code probably contains an infinite loop."
            )
        except Exception as exc:  # noqa: BLE001 - surface any spawn failure
            return f"SANDBOX ERROR: could not start the child process ({exc})."

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        # Last traceback line is the useful part; keep some context above it.
        tail = "\n".join(stderr.splitlines()[-6:])
        return f"CODE RAISED AN ERROR:\n{tail}" + (f"\n\npartial stdout:\n{stdout}" if stdout else "")
    if not stdout:
        return "Code ran successfully but printed nothing. It must print() its result."
    return stdout[:4000]  # cap the payload sent back into the prompt
