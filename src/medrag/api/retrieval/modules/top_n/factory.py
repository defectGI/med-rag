"""Production wiring: build a real embedder + Qdrant store + TopNRetriever
from env.

GPU/inference rule (ARCHITECTURE.md #10): the embedder points at a remote,
OpenAI-compatible endpoint (`EMBEDDING_*`, shared with `raptor`) -- configure
it here, never run inference on this machine. Qdrant *search* itself is not
inference, but the collection it searches is typically populated on the GPU
machine (by `vectorize` in this project's reference pipeline).

Same embedding model required: whatever model wrote the vectors in Qdrant
must be the one used to embed the query too (a mismatched model gives
meaningless similarity, and a mismatched vector size makes Qdrant itself error
out).
"""

from __future__ import annotations

import os

from medrag.api.retrieval.modules.top_n.embedder import embedder_from_env
from medrag.api.retrieval.modules.top_n.retriever import TopNRetriever
from medrag.api.retrieval.modules.top_n.store import QdrantVectorStore


def build_default_retriever(
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str | None = None,
    embedding_timeout: float = 120.0,
) -> TopNRetriever:
    """Wire a `TopNRetriever` from `QDRANT_*` (connection) + `EMBEDDING_*`
    (embedder, shared with `raptor`) env vars.

    `QDRANT_URL`/`QDRANT_API_KEY`/`QDRANT_COLLECTION` use the SAME prefix-less
    names the writer side (`vectorize`) uses for URL/API_KEY, so an operator
    does not have to remember a second scheme for "the same Qdrant". The older
    `TOP_N_QDRANT_*` names were renamed with no back-compat alias.

    ``qdrant_url``/``collection_name`` are the two values a consumer MUST
    provide (as a param or via env) -- there is no default: this module
    never builds the collection, so it cannot guess where one lives.
    """
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError(
            "qdrant_url is required (pass it, or set QDRANT_URL) -- the "
            "address of the Qdrant instance the vectors were written to."
        )
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION")
    if not collection_name:
        raise ValueError(
            "collection_name is required (pass it, or set QDRANT_COLLECTION)."
        )
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY") or None

    embedder = embedder_from_env(timeout=embedding_timeout)
    store = QdrantVectorStore.from_env(
        url=qdrant_url, collection_name=collection_name, api_key=qdrant_api_key
    )
    return TopNRetriever(embedder, store)


__all__ = ["build_default_retriever"]
