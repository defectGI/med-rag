"""Shared OpenAI-compatible embedding client.

The consolidation this lives in anticipated 7 `OpenAICompat<Role>` clones. A
fresh read found only 4 real classes, and just 2 of them byte-identical
(variable naming aside): `urun/api/retrieval/modules/top_n/embedder.py` and
`urun/pipeline/vectorize/embedder/openai_compat.py`, both named
`OpenAICompatEmbedder`, both a thin `POST {base}/embeddings` client with a
single `embed(texts)` method, no retry logic.

The other two are NOT clones of this pattern and are deliberately NOT merged
here:

* `urun/api/answering_model.py::OpenAICompatAnsweringModel` -- a chat
  client (10 init params, contract-retry loop, two internal chat paths:
  OpenAI-compatible and Ollama-native), not an embedder.
* `urun/pipeline/parser/llm/openai_compat.py::OpenAICompatClient` -- a
  chat+vision client with its own garbage-output retry loop and an active
  Ollama-restart recovery step (`ollama_recovery.restart_local_ollama`); also
  not an installed `medrag` dependent yet.

Only retrieval's `embedder_from_env`/`_common` and vectorize's
`embedder_from_env`/`_ortak` env-parsing helpers stay in their own modules --
those weren't part of the duplication this targets, just the class.
"""

from __future__ import annotations

from collections.abc import Sequence

from medrag.core.llm.errors import ProviderError
from medrag.core.llm.http import post_json as _post_json


class OpenAICompatEmbedder:
    """Client for any server speaking `POST {base}/embeddings`."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        provider: str = "openai",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider = provider
        self.name = f"{provider}:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = _post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": list(texts)},
            api_key=self.api_key,
            timeout=self.timeout,
        )
        try:
            data = response["data"]
            # Order is guaranteed by the API's own "index" field; trust it.
            vectors: list[list[float] | None] = [None] * len(texts)
            for record in data:
                vectors[record["index"]] = [float(x) for x in record["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"unexpected embeddings response shape: {str(response)[:500]}"
            ) from exc
        if any(v is None for v in vectors):
            raise ProviderError(
                f"embeddings response covers {sum(v is not None for v in vectors)}"
                f"/{len(texts)} inputs"
            )
        return vectors  # type: ignore[return-value]


__all__ = ["OpenAICompatEmbedder"]
