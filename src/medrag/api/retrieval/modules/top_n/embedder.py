"""OpenAI-compatible embedding client.

Mirrored from the vectorization component's OpenAI-compatible embedder -- code
is NOT imported across components (ARCHITECTURE.md #5: this repo never imports
a specific producer project). Same three deliberate choices, same reasons:

* A SINGLE OpenAI-compatible client: ollama/openai/openrouter/local all speak
  `POST {base}/embeddings`.
* `anthropic` is deliberately unsupported, rejected with a clear error.
* stdlib `urllib` -- no third-party HTTP dependency in this module.

Inference never runs on this machine (ARCHITECTURE.md #10) -- this client is
unverified against a real server; tests stub `_post_json`, real verification
happens on the GPU machine.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from medrag.core.llm.client import OpenAICompatEmbedder
from medrag.core.llm.errors import ProviderError
from medrag.core.llm.providers import COMPAT_DEFAULT_URL as _COMPAT_DEFAULT_URL
from medrag.core.llm.providers import COMPAT_PROVIDERS as _COMPAT_PROVIDERS


@runtime_checkable
class Embedder(Protocol):
    """The seam this module depends on -- unit tests inject a fake."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# OpenAICompatEmbedder lives in medrag.core.llm.client (a byte-identical twin of
# the vectorization component's embedder) -- re-imported above and re-exported
# via __all__ below so existing `from retrieval.modules.top_n.embedder import
# OpenAICompatEmbedder` call sites keep working.


def _common(env: Mapping[str, str], prefix: str) -> tuple[str, str, str, str | None]:
    provider = (env.get(f"{prefix}_PROVIDER") or "").strip().lower()
    if not provider:
        raise ProviderError(f"{prefix}_PROVIDER is not set (see .env.example)")
    if provider == "anthropic":
        raise ProviderError(
            f"{prefix}_PROVIDER=anthropic is not supported; use an OpenAI-compatible "
            f"provider (ollama/openai/openrouter/local)"
        )
    if provider not in _COMPAT_PROVIDERS:
        raise ProviderError(f"unknown {prefix}_PROVIDER: {provider!r}")
    model = (env.get(f"{prefix}_MODEL") or "").strip()
    if not model:
        raise ProviderError(f"{prefix}_MODEL is not set")
    base_url = (env.get(f"{prefix}_BASE_URL") or "").strip() or _COMPAT_DEFAULT_URL.get(
        provider, ""
    )
    if not base_url:
        raise ProviderError(f"{prefix}_BASE_URL is required (no default root for {provider})")
    return provider, model, base_url, (env.get(f"{prefix}_API_KEY") or None)


def embedder_from_env(
    env: Mapping[str, str] = os.environ, *, timeout: float = 120.0
) -> OpenAICompatEmbedder:
    """Build an embedder from `EMBEDDING_*` -- shared across top_n/raptor
    (`.env.example`, [inference])."""
    provider, model, base_url, api_key = _common(env, "EMBEDDING")
    return OpenAICompatEmbedder(
        base_url=base_url, model=model, api_key=api_key, provider=provider, timeout=timeout
    )


__all__ = ["Embedder", "OpenAICompatEmbedder", "ProviderError", "embedder_from_env"]
