"""Makes sure every LLM/VLM/VLM2 role that's configured to talk to a local
Ollama server actually has its model pulled and loaded before
`run_parse_pipeline.py` starts -- otherwise the first document in a batch
eats a multi-GB download or a cold-load delay.

Settings are read from exactly the same source as run_parse_pipeline.py:
first this script's own .env, then PARSER_DIR/.env (LLM_*/VLM_*/VLM2_* are
kept there -- see parser/llm/__init__.py). A role is considered "Ollama" if
its PROVIDER is "ollama", or if BASE_URL's port is 11434 or its host
contains "ollama" -- the model name is never hardcoded, whatever is
written in .env gets pulled (model-agnostic). Ollama is only ever talked to
through its stable HTTP API (/api/version, /api/tags, /api/pull,
/api/generate), CLI output is never parsed -- so this works no matter which
Ollama version is installed (version-agnostic).

If the server can't be reached:
  - if it's localhost/127.0.0.1: this script starts `ollama serve` itself
    (in the background, as a separate process). It does this ONLY if
    nothing is listening on that port -- i.e. it never touches, restarts,
    or reconfigures an Ollama instance that's already running (no matter
    who owns it).
  - if it's a different host (e.g. a remote GPU server): the failure to
    reach it is reported and that role is skipped; starting a service on
    another machine is not this script's job.

Running:
    python ensure_ollama_models.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

PARSER_DIR = os.environ.get("PARSER_DIR")
if PARSER_DIR:
    load_dotenv(Path(PARSER_DIR) / ".env")

# (prefix, fallback) -- VLM falls back to LLM per variable, VLM2 never falls
# back (see llm/__init__.py docstring). VLM_CLASSIFY (falls back to VLM) is
# handled separately in main() since it's optional and only appears when a
# profile deliberately overrides it -- same convention run_e2e_test.py uses.
ROLES = [("LLM", None), ("VLM", "LLM"), ("VLM2", None)]

_DEFAULT_BASE_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
_THINKING_TRUE = {"1", "true", "yes", "on"}

OLLAMA_ROOT_DEFAULT = "http://localhost:11434"

SERVE_START_TIMEOUT = 20.0  # time to wait for ollama serve to come up (sec)
PULL_TIMEOUT = 3600.0  # upper bound for large model downloads (sec)


def _env(prefix: str, name: str, fallback: str | None,
         cfg: Mapping[str, str] | None = None) -> str | None:
    source = cfg if cfg is not None else os.environ
    val = source.get(f"{prefix}_{name}")
    if val is None and fallback:
        val = source.get(f"{fallback}_{name}")
    return val or None


def resolve_role(prefix: str, fallback: str | None,
                  cfg: Mapping[str, str] | None = None) -> tuple[str, str | None, str | None]:
    """The same PROVIDER/MODEL/BASE_URL resolution as
    llm/__init__.py::_build_client, without importing parser (this script
    needs to work even without parser installed, before any model client
    exists). `cfg` defaults to the process's own os.environ (this script's
    normal self-use); pass an explicit dict to resolve against some other
    component's .env without mutating this process's environment (e.g. a
    preflight run from a different component's directory)."""
    provider = (_env(prefix, "PROVIDER", fallback, cfg) or "openai").strip().lower()
    model = _env(prefix, "MODEL", fallback, cfg)
    base_url = _env(prefix, "BASE_URL", fallback, cfg) or _DEFAULT_BASE_URL.get(provider)
    return provider, model, base_url


def is_ollama_target(provider: str, base_url: str | None) -> bool:
    if provider == "ollama":
        return True
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return parsed.port == 11434 or "ollama" in host


def ollama_root(base_url: str) -> str:
    """Strips the OpenAI-compatible '/v1' suffix and returns Ollama's own API root."""
    root = base_url.rstrip("/")
    root = root.removesuffix("/v1")
    return root


def is_local_host(root: str) -> bool:
    host = (urlparse(root).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_version(root: str) -> str | None:
    try:
        return _get_json(f"{root}/api/version").get("version")
    except Exception:  # noqa: BLE001 -- liveness probe: any error means "server not up"; the caller ensure_server_running prints the state
        return None


def ensure_server_running(root: str) -> bool:
    """Returns True once `root` responds to /api/version. Only starts
    `ollama serve` if nothing is listening on that port AND the host is
    local -- it never touches an Ollama instance (local or remote) that's
    already running."""
    version = server_version(root)
    if version is not None:
        print(f"  ollama server already running ({root}, version {version})")
        return True

    if not is_local_host(root):
        print(f"  WARNING: cannot reach remote Ollama server: {root} -- skipping")
        return False

    print(f"  local Ollama server at {root} is not responding, "
          f"starting `ollama serve`...")
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
    except FileNotFoundError:
        print("  ERROR: `ollama` command not found on PATH -- is Ollama installed?")
        return False

    deadline = time.monotonic() + SERVE_START_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(1.0)
        if server_version(root) is not None:
            print(f"  ollama serve came up ({root})")
            return True
    print(f"  ERROR: ollama serve did not come up within {SERVE_START_TIMEOUT:.0f}s")
    return False


def _norm(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def model_present(root: str, model: str) -> bool:
    try:
        tags = _get_json(f"{root}/api/tags").get("models", [])
    except Exception:  # noqa: BLE001 -- /api/tags unreadable -> assume "model absent"; the caller already goes to pull and prints the result
        return False
    names = {m.get("name") for m in tags}
    return model in names or _norm(model) in names


def pull_model(root: str, model: str) -> bool:
    print(f"  pulling: {model} ...")
    req = urllib.request.Request(
        f"{root}/api/pull",
        data=json.dumps({"name": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PULL_TIMEOUT) as resp:
            last_status = None
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("error"):
                    print(f"  ERROR: {model}: {evt['error']}")
                    return False
                status = evt.get("status")
                if status and status != last_status:
                    print(f"    {model}: {status}")
                    last_status = status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        print(f"  ERROR: {model} pull failed (HTTP {exc.code}): {detail}")
        return False
    except urllib.error.URLError as exc:
        print(f"  ERROR: {model} pull failed: {exc}")
        return False
    print(f"  done: {model}")
    return True


def warm_model(root: str, model: str) -> None:
    """Loads the model into memory (calling generate without a prompt is
    the official Ollama idiom for loading only, without producing output)."""
    req = urllib.request.Request(
        f"{root}/api/generate",
        data=json.dumps({"model": model, "stream": False, "keep_alive": "10m"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            resp.read()
        print(f"  loaded into memory: {model}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: failed to load {model} into memory: {exc}")


# ---------------------------------------------------------------------------
# thinking-really-off smoke test (shared with run_e2e_test.py / run_full_corpus.py --
# moved here 2026-07-17 so both entrypoints check with the exact same request
# shape instead of maintaining two copies).
# ---------------------------------------------------------------------------


def chat_once(root: str, model: str, prompt: str, thinking_on: bool) -> str:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0,
    }
    # Send exactly the toggle llm/__init__.py sends for an ollama provider, so
    # the smoke test exercises the same request shape the real run will --
    # reasoning_effort="none" only when thinking is meant to be OFF.
    if not thinking_on:
        payload["reasoning_effort"] = "none"
    req = urllib.request.Request(
        f"{root}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


def check_thinking_disabled(models: dict[str, str], thinking: dict[str, bool],
                             root: str = OLLAMA_ROOT_DEFAULT) -> bool:
    """Live call through each model -- not a trust-the-flag check. For a role
    whose thinking is meant to be OFF, a model that ignores the toggle either
    leaks a <think>...</think> block into content or burns the whole
    max_tokens budget and returns empty content; either one means every
    downstream stage (OCR, table description) silently produces garbage or
    nothing across the whole run, so callers abort before that happens. A
    role deliberately run with thinking ON is only checked for a non-error
    reply (empty content is expected there under a tiny budget)."""
    print("verifying reasoning/thinking behaves as configured (live smoke test)...")
    api_root = ollama_root(root)
    ok = True
    for role, model in models.items():
        want_thinking = thinking.get(role, False)
        try:
            content = chat_once(api_root, model, "Reply with exactly one word: OK", want_thinking)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{role}] {model}: FAILED to call model: {exc}")
            ok = False
            continue

        if want_thinking:
            print(f"  [{role}] {model}: OK -- thinking ON as configured (content not asserted)")
            continue

        stripped = content.strip()
        leaked = "<think" in content.lower()
        if leaked or not stripped:
            print(f"  [{role}] {model}: THINKING NOT SUPPRESSED "
                  f"(reasoning_effort='none' had no effect) -- raw content: {content!r}")
            ok = False
        else:
            print(f"  [{role}] {model}: OK -- thinking off, reply={stripped!r}")
    return ok


def resolve_thinking_models(cfg: Mapping[str, str],
                             roles: tuple[str, ...] = ("LLM", "VLM", "VLM_CLASSIFY"),
                             ) -> tuple[dict[str, str], dict[str, bool]]:
    """Picks out, from `cfg` (a component's .env values, e.g. via
    dotenv_values()), only the roles that resolve to an Ollama target -- the
    live smoke test above only makes sense for Ollama (it POSTs
    /v1/chat/completions with reasoning_effort, Ollama's own toggle shape).
    A role pointing at a cloud provider is silently skipped, same as
    ensure_ollama_models's own "not Ollama, skipping" behavior. VLM falls
    back to LLM, VLM_CLASSIFY falls back to VLM (same chain as
    llm/__init__.py); VLM_CLASSIFY is only included when the config actually
    overrides one of its own fields -- otherwise it just reuses the VLM
    client and there's nothing extra to check."""
    fallback_of = {"LLM": None, "VLM": "LLM", "VLM_CLASSIFY": "VLM"}
    models: dict[str, str] = {}
    thinking: dict[str, bool] = {}
    for role in roles:
        fb = fallback_of.get(role)
        if role == "VLM_CLASSIFY" and not any(
                cfg.get(f"VLM_CLASSIFY_{k}") for k in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY")):
            continue
        provider, model, base_url = resolve_role(role, fb, cfg)
        if not model or not is_ollama_target(provider, base_url):
            continue
        models[role] = model
        thinking[role] = (_env(role, "THINKING_ON", fb, cfg) or "").strip().lower() in _THINKING_TRUE
    return models, thinking


def _ensure_role(prefix: str, fallback: str | None, seen: dict[tuple[str, str], str]) -> bool:
    """Pulls+warms one role if it points at Ollama. Returns True if it did
    (i.e. this role IS an Ollama target), False otherwise -- used by main()
    to compute `any_ollama`."""
    provider, model, base_url = resolve_role(prefix, fallback)
    if not model or not base_url or not is_ollama_target(provider, base_url):
        print(f"[{prefix}] provider={provider!r} -> not Ollama, skipping")
        return False

    root = ollama_root(base_url)
    key = (root, model)
    if key in seen:
        print(f"[{prefix}] {model} @ {root} -> same as {seen[key]}, not reprocessing")
        return True

    print(f"[{prefix}] model={model!r} @ {root}")
    seen[key] = prefix
    if not ensure_server_running(root):
        return True

    if model_present(root, model):
        print(f"  already present: {model}")
    elif not pull_model(root, model):
        return True

    warm_model(root, model)
    return True


def main() -> None:
    seen: dict[tuple[str, str], str] = {}  # (root, model) -> previous role, to avoid reprocessing
    any_ollama = False

    for prefix, fallback in ROLES:
        if prefix == "VLM2" and not (os.getenv("VLM2_PROVIDER") or os.getenv("VLM2_MODEL")):
            continue  # optional role, may be deliberately left unset
        any_ollama = _ensure_role(prefix, fallback, seen) or any_ollama

    # VLM_CLASSIFY: optional, only pulled/warmed when a profile actually
    # overrides one of its own fields (otherwise it reuses the VLM client
    # above, nothing extra to fetch) -- same condition run_e2e_test.py uses.
    if any(os.getenv(f"VLM_CLASSIFY_{k}") for k in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY")):
        any_ollama = _ensure_role("VLM_CLASSIFY", "VLM", seen) or any_ollama

    if not any_ollama:
        print("No LLM/VLM/VLM2/VLM_CLASSIFY role points to Ollama -- nothing to do.")


if __name__ == "__main__":
    main()
