"""Thinking/reasoning toggle: shared env -> request-body logic.

When a hybrid-reasoning model is left with thinking ON, it can spend its
entire token budget on the hidden ``<think>`` block for short-answer tasks and
return empty/unparseable content (a problem proven in the parser and the
benchmark tooling). Therefore every LLM role's thinking must be
suppressible via env.

Convention: suppression is **opt-in**. A suppression field is merged into the
request body only when ``<PREFIX>_THINKING`` is explicitly ``off``; when unset
or ``on``, NOTHING is injected (neutral -- so a model without thinking can
never reject it, and legacy behavior is preserved). This is deliberate: the
default ``{"reasoning_effort": "none"}`` can return 400 on some servers (e.g.
some Ollama /v1); the answering model handles that with a graceful
"retry without body if rejected", but intent (langchain) has no such hook --
so injection must be an explicit opt-in, never a silent default.

When ``off``, the suppression body is overridden by ``<PREFIX>_THINKING_OFF_BODY``
(JSON); the default is ``{"reasoning_effort": "none"}`` (the parser's proven
choice). Preflight (preflight.py) probes with the same body at startup and
prints any leakage to the terminal, so a wrong body is caught at boot rather
than at request time.

Because `retrieval`'s intent_classification is a separate component (not
imported), it carries its own tiny equivalent -- this module only feeds the
roles owned by THIS component (answering + preflight).
"""

from __future__ import annotations

import json
from collections.abc import Mapping

_DEFAULT_OFF_BODY: dict = {"reasoning_effort": "none"}


def is_suppressing(env: Mapping[str, str], prefix: str) -> bool:
    """Is ``<prefix>_THINKING`` EXPLICITLY ``off``? Only then is suppression
    applied; unset/on -> False (nothing injected, neutral)."""
    return (env.get(f"{prefix}_THINKING") or "").strip().lower() == "off"


def thinking_off_body(env: Mapping[str, str], prefix: str) -> dict:
    """The dict to merge when suppression is EXPLICITLY requested
    (``<prefix>_THINKING=off``); empty otherwise (neutral).

    When suppressing: the ``<prefix>_THINKING_OFF_BODY`` JSON (if present) or
    the default ``{"reasoning_effort": "none"}``. Invalid JSON -> ValueError
    (fail loudly instead of swallowing silently; the root CONFIG.md principle
    is "missing/broken config fails loudly")."""
    if not is_suppressing(env, prefix):
        return {}
    raw = (env.get(f"{prefix}_THINKING_OFF_BODY") or "").strip()
    if not raw:
        return dict(_DEFAULT_OFF_BODY)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{prefix}_THINKING_OFF_BODY geçerli bir JSON nesnesi değil: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise ValueError(  # noqa: TRY004 -- not a type error but a CONFIG error; the docstring's "fail loudly" contract is built on ValueError
            f"{prefix}_THINKING_OFF_BODY bir JSON NESNESİ olmalı, {type(body).__name__} değil"
        )
    return body


__all__ = ["is_suppressing", "thinking_off_body"]
