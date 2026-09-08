"""Shared Anthropic Messages API call + response parsing.

`answering_model.py`/`reconciler.py`/`continuation.py`/`slot_selection.py`/
`preflight.py` (see each one's `_call_openai_compat`) all speak the
OpenAI-compatible `/v1/chat/completions` contract -- Anthropic's Messages API
(`/v1/messages`) is DIFFERENT in three places:
  1. the `system` message is not INSIDE the list, it is a SEPARATE top-level
     field in the body.
  2. `max_tokens` is optional in OpenAI, MANDATORY in Anthropic.
  3. Auth uses the `x-api-key` + `anthropic-version` headers instead of
     `Authorization: Bearer`; the answer is the FIRST `type=="text"` block in
     the `content[]` list, not `choices[0].message.content` (on a model with
     extended thinking enabled `content[0]` can be `type=="thinking"` -- a
     failure actually hit in production, see the note inside `call_anthropic`).

These three differences are translated in ONE place -- each of the five
call sites passes its own `messages` list (already built in the
OpenAI-compatible `{"role":.., "content":..}` format) in here, the rest happens
here (over the SAME shared HTTP/error-handling layer as `core/llm/http.py::
post_json` -- no second urllib copy was opened, keeping the consolidation
principle of that layer).
"""

from __future__ import annotations

from medrag.core.llm.errors import ProviderError
from medrag.core.llm.http import post_json

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


def call_anthropic(
    messages: list[dict[str, str]], *, model: str, base_url: str,
    api_key: str | None, timeout: float, max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
) -> str:
    """`messages` is expected in the OpenAI-compatible format
    (`{"role": "system"|"user"|"assistant", "content": str}`) -- the SAME one
    the other providers' callers use. `system`-role messages are split out and
    moved into Anthropic's separate `system` field (joined if there is more
    than one); the remaining messages go into Anthropic's `messages` in order."""
    if not api_key:
        raise ProviderError("Anthropic provider icin API_KEY zorunlu (bkz. .env.example)")
    system_parts = [str(m["content"]) for m in messages if m.get("role") == "system"]
    rest = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
    payload: dict = {"model": model, "messages": rest, "max_tokens": max_tokens}
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    # `base_url` may have been entered with the SAME habit as the other
    # providers (`.../v1`) -- that is how it is used everywhere in the repo
    # (e.g. the `/v1` roots of ollama/openai/openrouter). Anthropic's real
    # endpoint is `/v1/messages`; a trailing `/v1` is stripped once and added
    # BACK, so both `.../v1` and the bare root (`ANTHROPIC_DEFAULT_URL`) work
    # (a real failure: `.../v1/v1/messages` -- 404).
    url = f"{base_url.rstrip('/').removesuffix('/v1')}/v1/messages"
    response = post_json(
        url, payload, api_key=None, timeout=timeout,
        extra_headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
    )
    # A real failure: `content[0]` cannot be taken BLINDLY -- on a model with
    # extended thinking enabled the first block is `type=="thinking"` (NOT the
    # answer text, it has no `text` field at all, `KeyError`). Search by type
    # the way `text2sql-native/providers/anthropic_provider.py` already does:
    # SKIP thinking/redacted_thinking blocks, take the first real `text` block.
    try:
        blocks = response["content"]
        for block in blocks:
            if block.get("type") == "text":
                return str(block["text"])
    except (KeyError, TypeError) as exc:
        raise ProviderError(
            f"unexpected Anthropic response shape: {str(response)[:500]}"
        ) from exc
    raise ProviderError(
        f"Anthropic yanıtında text bloğu yok (thinking-only?): {str(response)[:500]}"
    )


__all__ = ["ANTHROPIC_DEFAULT_MAX_TOKENS", "ANTHROPIC_VERSION", "call_anthropic"]
