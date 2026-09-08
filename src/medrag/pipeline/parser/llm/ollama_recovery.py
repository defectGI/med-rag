"""Recovery from a degenerate local Ollama server.

A model that starts repeating a single character forever ("??????...") is a
known local-inference failure mode (context overflow, a wedged KV cache, a
corrupted model instance) distinct from a normal transport error: the HTTP
call succeeds and the JSON parses fine, but the "answer" is garbage. Left
alone, a caller that just retries against the same server keeps getting the
same junk forever -- nothing recovers on its own until the Ollama server
itself is restarted.

Two pieces:
    is_garbage_output    -- cheap, provider-agnostic pattern check
    restart_local_ollama -- kill + relaunch `ollama serve`, LOCAL HOST ONLY

Only ever acts on localhost/127.0.0.1/::1: a remote Ollama (another machine,
e.g. a shared GPU box) is never killed or restarted from here -- same
boundary `pipeline/ensure_ollama_models.py` draws, just from the opposite
direction (that script starts a server that ISN'T running yet; this one
kills one that IS running but stuck).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urlparse

__all__ = ["is_garbage_output", "restart_local_ollama"]

# A reply shorter than this is never flagged -- a short legitimate answer
# ("---", "N/A") can be all-one-character without being degenerate.
_MIN_LEN = 20
# Share of the (whitespace-stripped) reply a single character must occupy
# to count as degenerate output rather than a coincidentally repetitive one.
_DOMINANT_SHARE = 0.8

# Time to wait for `ollama serve` to answer /api/version again after restart.
SERVE_RESTART_TIMEOUT = 20.0


def is_garbage_output(text: str) -> bool:
    """True if `text` looks like a model stuck repeating one character.

    Whitespace is ignored first (a real answer can be newline/space-heavy
    without being degenerate); only the non-whitespace characters are
    counted toward the dominant-character share.
    """
    stripped = "".join(text.split())
    if len(stripped) < _MIN_LEN:
        return False
    _, top_count = Counter(stripped).most_common(1)[0]
    return top_count / len(stripped) >= _DOMINANT_SHARE


def _ollama_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    root = root.removesuffix("/v1")
    return root


def _is_local_host(root: str) -> bool:
    host = (urlparse(root).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def _server_version(root: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{root}/api/version", timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001 -- liveness probe: any error means "the server is not answering", which is the restart flow's input
        return None


def _kill_ollama_process() -> None:
    """Best-effort: end whatever local process is serving Ollama.

    Nothing in this codebase tracks the server's PID (it may have been
    started by the user, a previous run, or `ensure_ollama_models.py`), so
    this targets the process BY NAME instead of a handle -- `taskkill` on
    Windows, `pkill` elsewhere. Never raises: "nothing to kill" and "killed
    successfully" both just fall through to the caller relaunching it.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/IM", "ollama.exe", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                check=False,  # best-effort: "no such process" counts as success too
            )
        else:
            subprocess.run(
                ["pkill", "-f", "ollama serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                check=False,  # best-effort: pkill returns 1 when nothing matched, which is fine
            )
    except Exception:  # noqa: BLE001, S110 -- the "Never raises" promise in the docstring: taskkill/pkill missing or no process to kill, both normal
        pass


def _start_ollama_process() -> bool:
    try:
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            popen_kwargs["start_new_session"] = True
        subprocess.Popen(["ollama", "serve"], **popen_kwargs)
        return True
    except FileNotFoundError:
        return False


def restart_local_ollama(base_url: str) -> bool:
    """Kill and relaunch a LOCAL Ollama server.

    Returns True once it answers `/api/version` again. Returns False without
    touching anything if `base_url` isn't localhost/127.0.0.1/::1 (a remote
    server is never killed from here), and False if `ollama` isn't on PATH
    or the server doesn't come back within `SERVE_RESTART_TIMEOUT`.
    """
    root = _ollama_root(base_url)
    if not _is_local_host(root):
        return False
    _kill_ollama_process()
    time.sleep(1.0)  # give the port a moment to free up before relaunching
    if not _start_ollama_process():
        return False
    deadline = time.monotonic() + SERVE_RESTART_TIMEOUT
    while time.monotonic() < deadline:
        if _server_version(root) is not None:
            return True
        time.sleep(1.0)
    return False
