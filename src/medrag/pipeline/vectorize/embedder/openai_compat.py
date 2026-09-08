"""OpenAI-uyumlu embedding istemcisi.

Desen `chunker/chunker/enrichment/raptor/openai_compat.py`den AYNALANIR (kod
import edilmez — kök AGENTS.md mimari kuralı: bileşenler birbirini import
etmez). Aynalanan üç nüans, aynı gerekçelerle:

* TEK OpenAI-uyumlu istemci: ollama/openai/openrouter/local hepsi
  `POST {base}/embeddings` konuşur.
* `anthropic` bilinçli olarak desteklenmez, anlaşılır hatayla reddedilir.
* stdlib `urllib` — üçüncü parti HTTP bağımlılığı yok.

Bu makinede embedding koşulmadığından (kök AGENTS.md kalıcı kural) bu modül
gerçek sunucuya karşı DOĞRULANMAMIŞTIR — testler `_post_json`ı stub'lar;
gerçek doğrulama GPU makinesinde yapılır (bkz. vectorize/DEVLOG.md).

`ProviderError` (D-25), `_post_json` (D-26, `core/llm/http.py::post_json`
adıyla) ve `_COMPAT_DEFAULT_URL`/`_COMPAT_PROVIDERS` (D-27,
`core/llm/providers.py`) `medrag.core.llm`'e taşındı — bu bir bileşen
importu DEĞİL (K-20: `core/` en az iki bileşen kullanıyorsa oraya girer,
"bileşenler birbirini import etmez" kuralı yalnız bileşenden bileşene
importu kapsıyor).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from medrag.core.llm.client import OpenAICompatEmbedder
from medrag.core.llm.errors import ProviderError
from medrag.core.llm.providers import COMPAT_DEFAULT_URL as _COMPAT_DEFAULT_URL
from medrag.core.llm.providers import COMPAT_PROVIDERS as _COMPAT_PROVIDERS

# D-31: OpenAICompatEmbedder moved to medrag.core.llm.client (byte-identical
# twin with retrieval/src/retrieval/modules/top_n/embedder.py) -- re-imported
# above so existing `from medrag.pipeline.vectorize.embedder.openai_compat import
# OpenAICompatEmbedder` call sites keep working.


def _ortak(env: Mapping[str, str], prefix: str) -> tuple[str, str, str, str | None]:
    provider = (env.get(f"{prefix}_PROVIDER") or "").strip().lower()
    if not provider:
        raise ProviderError(f"{prefix}_PROVIDER tanımsız (bkz. .env.example)")
    if provider == "anthropic":
        raise ProviderError(
            f"{prefix}_PROVIDER=anthropic desteklenmiyor; OpenAI-uyumlu bir "
            f"sağlayıcı kullanın (ollama/openai/openrouter/local)")
    if provider not in _COMPAT_PROVIDERS:
        raise ProviderError(f"bilinmeyen {prefix}_PROVIDER: {provider!r}")
    model = (env.get(f"{prefix}_MODEL") or "").strip()
    if not model:
        raise ProviderError(f"{prefix}_MODEL tanımsız")
    base_url = (env.get(f"{prefix}_BASE_URL") or "").strip() \
        or _COMPAT_DEFAULT_URL.get(provider, "")
    if not base_url:
        raise ProviderError(f"{prefix}_BASE_URL gerekli ({provider} için "
                            f"varsayılan kök yok)")
    return provider, model, base_url, (env.get(f"{prefix}_API_KEY") or None)


def embedder_from_env(env: Mapping[str, str] = os.environ,
                       *, timeout: float = 120.0) -> OpenAICompatEmbedder:
    """`EMBEDDING_*` değişkenlerinden embedder kur."""
    provider, model, base_url, api_key = _ortak(env, "EMBEDDING")
    return OpenAICompatEmbedder(base_url=base_url, model=model, api_key=api_key,
                                provider=provider, timeout=timeout)
