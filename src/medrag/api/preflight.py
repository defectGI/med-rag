"""Startup preflight: at boot, fires a known-answer probe at every LLM role
and prints health + whether thinking is at the desired setting to the
terminal.

Why: a reachable endpoint (HTTP 200) does NOT mean a usable model -- a
hybrid-reasoning model can show "up" yet spend its whole token budget on the
hidden ``<think>`` block and return empty/unparseable content (proven in
parser/benchmark work). Since the four roles can be separate
endpoints/models (answering, intent, linking, sql), each is checked
individually.

Scope:
- For the **answering** (chatbot) and **intent** (retrieval) roles, thinking
  IS controllable from this repo -- the probe is fired with the same
  suppression body (thinking.py) the role's real calls use, and "is it at the
  desired setting" is verified.
- For the **linking** / **sql** roles (external text2sql-engine), thinking is
  NOT controllable (the package's provider has no extra_body hook) -- the
  probe only DETECTS ``<think>`` LEAKAGE and warns (detect-only).

Behavior on failure: default is WARN + CONTINUE (a long-lived web server
should boot even against a momentarily down model). If `CHATBOT_PREFLIGHT_STRICT`
is truthy, abort instead (SystemExit) -- for batch/CI that wants fail-fast.

No REAL calls run on this machine (GPU rule): probes go to remote endpoints,
and only `main()` (actual server boot) calls this -- `create_app()` never
touches the network, so tests import it safely.

Scope addition: besides LLM roles, the Qdrant that top_n connects to is also
checked as a one-line probe -- if Qdrant is unreachable or the collection is
missing, the top_n retriever explodes on the first real query, and we want to
see that at boot (same early-warning rationale as the LLM roles'
"known-answer probe"). If `QDRANT_*` (see `.env.example`) is unconfigured
(top_n unused in this deployment), it is skipped like the LLM roles'
"skip" pattern in `roles_from_env`.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

from medrag.api.answering_model import ProviderError, _post_json
from medrag.api.reply_contract import has_reasoning_marker
from medrag.api.thinking import is_suppressing, thinking_off_body

_PROBE_SYSTEM = "You are a health check. Follow the instruction exactly."
_PROBE_USER = "Reply with exactly: PONG"
_TRUE = ("1", "true", "yes", "on")


@dataclass
class RoleProbe:
    """A single LLM role checked by preflight."""

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    # Is thinking controllable for this role (answering/intent) or is this
    # leak-detection only (linking/sql, external text2sql)?
    controllable: bool = True
    want_thinking: bool = False
    off_body: dict = field(default_factory=dict)
    # When `provider == "ollama" and num_ctx` are set, the probe goes to native
    # `/api/chat` -- the same distinction as the real runtime branching of
    # answering/reconciler/continuation (see `_call_native` in
    # answering_model.py/reconciler.py/continuation.py); if the probe here
    # didn't mirror that branching (its old form), then with a BASE_URL whose
    # `/v1` was stripped the real call would succeed via native while preflight
    # mistakenly printed 404. `intent` (retrieval's own classifier) and
    # `linking`/`sql` (external text2sql-engine) NEVER FILL these fields --
    # those clients' own code doesn't implement the native distinction yet, so
    # the `/v1` assumption is still correct for them.
    provider: str | None = None
    num_ctx: int | None = None


@dataclass
class RoleResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class QdrantProbe:
    """The Qdrant connection checked by preflight (top_n's target)."""

    url: str
    collection_name: str
    api_key: str | None = None


def qdrant_probe_from_env(env: Mapping[str, str]) -> QdrantProbe | None:
    """`QDRANT_URL`/`QDRANT_COLLECTION` (see .env.example,
    factory.py::build_top_n_retriever_from_env); None if unconfigured
    -- top_n isn't used in this deployment, skipped like the roles."""
    url = (env.get("QDRANT_URL") or "").strip()
    collection = (env.get("QDRANT_COLLECTION") or "").strip()
    if not url or not collection:
        return None
    return QdrantProbe(url=url, collection_name=collection,
                       api_key=(env.get("QDRANT_API_KEY") or None) or None)


def _qdrant_get_collection(url: str, collection_name: str, api_key: str | None, timeout: float):
    """The real Qdrant call -- tests monkeypatch this (without touching the
    transport at all); same lazy-import pattern as
    `QdrantVectorStore.from_env`."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
    return client.get_collection(collection_name)


def probe_qdrant(probe: QdrantProbe, *, timeout: float = 30.0) -> RoleResult:
    """Connects to Qdrant and verifies the collection exists (not just a
    reachable endpoint -- a wrong/deleted collection name is caught here too)."""
    try:
        info = _qdrant_get_collection(probe.url, probe.collection_name, probe.api_key, timeout)
    except Exception as exc:  # noqa: BLE001 - qdrant-client raises different types for connection/404/etc.
        return RoleResult("qdrant", False, f"transport: {exc}")

    points_count = getattr(info, "points_count", None)
    detail = f"collection={probe.collection_name!r}"
    if points_count is not None:
        detail += f", points={points_count}"
    return RoleResult("qdrant", True, detail)


def roles_from_env(env: Mapping[str, str]) -> list[RoleProbe]:
    """Collects configured roles from env. A role without base_url/model is
    skipped (it means the role isn't used in this deployment)."""
    roles: list[RoleProbe] = []

    def add(name: str, base: str, model: str, key: str, *, controllable: bool, prefix: str | None) -> None:
        base_v = (env.get(base) or "").strip()
        model_v = (env.get(model) or "").strip()
        if not base_v or not model_v:
            return
        # Suppression explicitly requested (prefix_THINKING=off): a leak is a
        # failure; otherwise thinking is allowed (leak isn't flagged).
        # linking/sql (prefix=None) are detect-only: no suppression, but a
        # leak still warns.
        suppressing = bool(prefix) and is_suppressing(env, prefix)
        off = thinking_off_body(env, prefix) if prefix else {}
        want = (not suppressing) if prefix else False
        roles.append(RoleProbe(
            name=name, base_url=base_v, model=model_v, api_key=(env.get(key) or None),
            controllable=controllable, want_thinking=want, off_body=off,
        ))

    # answering: chatbot's own client (answering_model.py) supports native
    # `/api/chat` -- the probe uses the SAME provider/num_ctx resolution
    # (answering_model._resolve_num_ctx) so it mirrors exactly which path the
    # real call will take.
    answering_base = (env.get("CHATBOT_LLM_BASE_URL") or "").strip()
    answering_model = (env.get("CHATBOT_LLM_MODEL") or "").strip()
    if answering_base and answering_model:
        from medrag.api.answering_model import _resolve_num_ctx

        answering_suppressing = is_suppressing(env, "CHATBOT_LLM")
        answering_provider = (env.get("CHATBOT_LLM_PROVIDER") or "openai").strip().lower()
        roles.append(RoleProbe(
            name="answering", base_url=answering_base, model=answering_model,
            api_key=(env.get("CHATBOT_LLM_API_KEY") or None), controllable=True,
            want_thinking=not answering_suppressing,
            off_body=thinking_off_body(env, "CHATBOT_LLM"),
            provider=answering_provider,
            num_ctx=_resolve_num_ctx(env, "CHATBOT_LLM", answering_provider),
        ))
    # intent: retrieval package's own classifier -- this client's code (not
    # chatbot's) doesn't implement the native `/api/chat` distinction yet, so
    # `provider`/`num_ctx` are DELIBERATELY left unset (always assumes `/v1`).
    add("intent", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
        controllable=True, prefix="LLM")
    # reconciler (L0+L1): RECONCILER_* override or LLM_* fallback. Endpoint/
    # provider/num_ctx resolution lives in one place in reconciler.py (same
    # RECONCILER_*->LLM_* pattern as BASE_URL/MODEL/API_KEY).
    from medrag.api.reconciler import (
        _reconciler_endpoint,
        _reconciler_num_ctx,
        _reconciler_provider,
    )

    rec_base, rec_model, rec_key = _reconciler_endpoint(env)
    if rec_base and rec_model:
        rec_suppressing = is_suppressing(env, "RECONCILER")
        rec_provider = _reconciler_provider(env)
        roles.append(RoleProbe(
            name="reconciler", base_url=rec_base, model=rec_model, api_key=rec_key,
            controllable=True, want_thinking=not rec_suppressing,
            off_body=thinking_off_body(env, "RECONCILER"),
            provider=rec_provider, num_ctx=_reconciler_num_ctx(env, rec_provider),
        ))
    # continuation: CONTINUATION_* override or LLM_* fallback -- the SAME
    # pattern as the `reconciler` role; endpoint/provider/num_ctx resolution
    # lives in one place in continuation.py (see that module).
    from medrag.api.continuation import (
        _continuation_endpoint,
        _continuation_num_ctx,
        _continuation_provider,
    )

    cont_base, cont_model, cont_key = _continuation_endpoint(env)
    if cont_base and cont_model:
        cont_suppressing = is_suppressing(env, "CONTINUATION")
        cont_provider = _continuation_provider(env)
        roles.append(RoleProbe(
            name="continuation", base_url=cont_base, model=cont_model, api_key=cont_key,
            controllable=True, want_thinking=not cont_suppressing,
            off_body=thinking_off_body(env, "CONTINUATION"),
            provider=cont_provider, num_ctx=_continuation_num_ctx(env, cont_provider),
        ))
    # linking/sql: external text2sql-engine, its own unprefixed env +
    # OPENAI_API_KEY; thinking uncontrollable -> detect-only (prefix=None,
    # off_body empty).
    add("linking", "LINKING_BASE_URL", "LINKING_MODEL", "OPENAI_API_KEY",
        controllable=False, prefix=None)
    add("sql", "SQL_BASE_URL", "SQL_MODEL", "OPENAI_API_KEY",
        controllable=False, prefix=None)
    return roles


def probe_role(role: RoleProbe, *, timeout: float = 30.0) -> RoleResult:
    """Fires a known-answer probe at a single role: health (PONG) + thinking
    behavior.

    Merges the SAME suppression body (off_body) the role's real calls use --
    so "is thinking at the desired setting" is genuinely tested, not
    hypothetical."""
    messages = [
        {"role": "system", "content": _PROBE_SYSTEM},
        {"role": "user", "content": _PROBE_USER},
    ]
    if role.provider == "anthropic":
        from medrag.core.llm.anthropic import call_anthropic

        try:
            content = call_anthropic(
                messages, model=role.model, base_url=role.base_url,
                api_key=role.api_key, timeout=timeout,
            )
        except ProviderError as exc:
            return RoleResult(role.name, False, f"transport: {exc}")
    elif role.provider == "ollama" and role.num_ctx:
        # The same distinction as the real runtime branching of
        # answering/reconciler/continuation (see `_call_native` in
        # answering_model.py/reconciler.py/continuation.py) -- if the probe
        # didn't also go to native `/api/chat`, a BASE_URL with `/v1` stripped
        # would print a bogus 404 (even when the real call works fine via
        # native).
        payload: dict = {
            "model": role.model, "messages": messages, "stream": False,
            "options": {"num_ctx": role.num_ctx},
        }
        if "reasoning_effort" in role.off_body:
            payload["think"] = False
        url = f"{role.base_url.removesuffix('/v1')}/api/chat"
        try:
            response = _post_json(url, payload, api_key=role.api_key, timeout=timeout)
        except ProviderError as exc:
            return RoleResult(role.name, False, f"transport: {exc}")
        try:
            content = str(response["message"]["content"] or "")
        except (KeyError, TypeError):
            return RoleResult(role.name, False, f"beklenmeyen yanıt: {str(response)[:200]}")
    else:
        base_payload = {"model": role.model, "messages": messages}
        url = f"{role.base_url.rstrip('/')}/chat/completions"
        try:
            response = _post_json(url, {**base_payload, **role.off_body},
                                  api_key=role.api_key, timeout=timeout)
        except ProviderError as exc:
            # If off_body was rejected: the model doesn't know this thinking
            # mode -> retry without a body (same as the answering model's
            # runtime behavior).
            if role.off_body and any(str(k).lower() in str(exc).lower() for k in role.off_body):
                try:
                    response = _post_json(url, base_payload, api_key=role.api_key, timeout=timeout)
                except ProviderError as exc2:
                    return RoleResult(role.name, False, f"transport: {exc2}")
            else:
                return RoleResult(role.name, False, f"transport: {exc}")

        try:
            content = str(response["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            return RoleResult(role.name, False, f"beklenmeyen yanıt: {str(response)[:200]}")

    lowered = content.lower()
    stripped = content.strip()

    # Health: it should return PONG.
    if "pong" not in lowered:
        # Empty content + thinking off = most likely thinking ate the budget.
        if not stripped and not role.want_thinking:
            return RoleResult(role.name, False,
                              "thinking bastırılamadı (boş content -- <think> bütçeyi yemiş olabilir)")
        return RoleResult(role.name, False, f"kullanılamaz yanıt: {stripped[:120]!r}")

    # Thinking behavior.
    if role.want_thinking:
        return RoleResult(role.name, True, "thinking on")
    # This spot used to search ONLY for the literal `<think` -- it couldn't see
    # the harmony/`<|channel|>` forms observed in live testing, so preflight
    # printed "OK (thinking off)" while real turns leaked. Detection is now
    # centralized in `reply_contract.has_reasoning_marker` (the SAME marker
    # list as the answering role's output contract -- the two can't diverge).
    leak = has_reasoning_marker(content)
    if leak is not None:
        note = "sızıntı" if not role.controllable else "bastırılamadı"
        return RoleResult(role.name, False, f"muhakeme {note} ({leak}): {stripped[:120]!r}")

    return RoleResult(role.name, True, "thinking off" if role.controllable else "detect-only, temiz")


def run_startup_preflight(env: Mapping[str, str], *, timeout: float = 30.0,
                          out=sys.stderr) -> bool:
    """Probe every role, print to the terminal, return True when all are OK.

    Called by `main()`: if it returns False and `CHATBOT_PREFLIGHT_STRICT` is
    truthy, boot is aborted; otherwise a warning is printed and startup
    continues."""
    roles = roles_from_env(env)
    qdrant = qdrant_probe_from_env(env)
    print("model preflight:", file=out)
    if not roles and qdrant is None:
        print("  (konfigüre edilmiş rol yok -- .env boş mu?)", file=out)
        return True

    names = [r.name for r in roles] + (["qdrant"] if qdrant is not None else [])
    width = max(len(n) for n in names)
    all_ok = True
    for role in roles:
        result = probe_role(role, timeout=timeout)
        status = "OK" if result.ok else "FAILED"
        detail = f" ({result.detail})" if result.detail else ""
        print(f"  [{role.name.ljust(width)}] {status}{detail}", file=out)
        all_ok = all_ok and result.ok

    if qdrant is not None:
        result = probe_qdrant(qdrant, timeout=timeout)
        status = "OK" if result.ok else "FAILED"
        detail = f" ({result.detail})" if result.detail else ""
        print(f"  [{'qdrant'.ljust(width)}] {status}{detail}", file=out)
        all_ok = all_ok and result.ok

    if not all_ok:
        strict = (env.get("CHATBOT_PREFLIGHT_STRICT") or "").strip().lower() in _TRUE
        if strict:
            print("preflight başarısız (CHATBOT_PREFLIGHT_STRICT=1) -- kalkış iptal.", file=out)
        else:
            # ASCII marker is deliberate: a Windows cp1254 terminal dies with
            # UnicodeEncodeError on a symbol like '⚠' (U+26A0) -- preflight's
            # only job is printing to the terminal, so its output must be
            # strictly codepage-safe.
            print("[!] preflight uyardı ama sunucu başlıyor "
                  "(fail-fast için CHATBOT_PREFLIGHT_STRICT=1).", file=out)
    return all_ok


__all__ = [
    "QdrantProbe",
    "RoleProbe",
    "RoleResult",
    "probe_qdrant",
    "probe_role",
    "qdrant_probe_from_env",
    "roles_from_env",
    "run_startup_preflight",
]
