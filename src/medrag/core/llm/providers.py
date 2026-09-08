"""Shared OpenAI-compatible provider constants.

`_COMPAT_DEFAULT_URL` (default `/v1` root per provider) and
`_COMPAT_PROVIDERS` were byte-identical across `urun/api/answering_model.py`,
`urun/api/retrieval/modules/top_n/embedder.py`, and
`urun/pipeline/vectorize/embedder/openai_compat.py`. Consolidated here.

**Correction to the original consolidation plan:** the plan anticipated
needing *two* named URL constants -- one OpenAI-compatible `/v1` root and one
bare Ollama-native root -- because
`urun/pipeline/parser/llm/openai_compat.py` routes `num_ctx`-bearing Ollama
calls through Ollama's native `/api/chat` instead of `/v1`
(`github.com/ollama/ollama/issues/5356`: `/v1` can't carry `num_ctx`
per-request). On inspection that native-root derivation is NOT a second
constant anywhere -- parser computes it at call time as
`self.base_url.removesuffix("/v1")` inside `OpenAICompatClient._chat`, and
that whole routing decision lives in parser's client class, which is out of
this batch's scope (parser itself isn't an installed `medrag` dependent yet --
see the http.py docstring). Within THIS consolidation's actual scope
(chatbot/retrieval/vectorize), there was only ever one dict, three identical
copies -- no duality to preserve here. `COMPAT_DEFAULT_URL` below is exactly
that: the OpenAI-compatible `/v1` default root per provider, nothing else.

Also NOT folded in here (deliberately, still parser-only, still a second copy
after this batch): `urun/pipeline/parser/llm/__init__.py`'s own local
`_COMPAT_DEFAULT_URL` inside `_build_client` differs in one real way --
it has no `"openai"` entry (parser's provider list requires an explicit
`{prefix}_BASE_URL` for the "openai" provider rather than defaulting to
`api.openai.com/v1`). That's a genuine behavioral difference, not
incidental drift; silently adding an `"openai"` default there would
change parser's behavior. Left alone, kept as its own separate,
independently-revertable commit target for whenever parser joins `medrag`.
"""

from __future__ import annotations

COMPAT_DEFAULT_URL: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}

COMPAT_PROVIDERS: tuple[str, ...] = ("ollama", "openai", "openrouter", "local")

# Anthropic's Messages API does NOT speak the OpenAI-compatible
# `/v1/chat/completions` contract (separate `system` field, mandatory
# `max_tokens`, different auth headers, different response shape) -- that's why
# it was NOT added to `COMPAT_PROVIDERS` and is kept as a separate constant.
# The translation lives in `core/llm/anthropic.py`.
ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"

# Ollama's OpenAI-compatible `/v1` endpoint cannot carry `num_ctx` per-request
# (github.com/ollama/ollama/issues/5356) -- an over-budget prompt is silently
# truncated. When `{prefix}_NUM_CTX` is unset AND the provider is "ollama",
# callers fall back to this instead of silently running unenforced. Mirrors
# `parser/llm/__init__.py::_DEFAULT_OLLAMA_NUM_CTX` (same value, kept as a
# separate constant there -- parser isn't an installed `medrag` dependent yet).
#
# This is a FALLBACK, not a per-role tuning value -- it does not override the
# role-specific values env vars already carry (the SQL/coder model vs. the five
# chatbot roles were deliberately aligned at 65536 via CHATBOT_LLM_NUM_CTX /
# RECONCILER_NUM_CTX / CONTINUATION_NUM_CTX / SLOT_SELECTOR_NUM_CTX /
# INTENT_NUM_CTX). Consolidating this constant does not touch that
# decision -- each role still reads its own `{prefix}_NUM_CTX` first and only
# reaches this default when that's empty.
DEFAULT_OLLAMA_NUM_CTX = 16384
