"""top_n -- vector (embedding) similarity retrieval over an existing Qdrant
collection.

This module never builds the collection it queries -- that is `vectorize`'s
job in this project's reference pipeline (or any producer writing the same
payload shape: `text`, `node_id`, `doc_id`, `heading_path`, `page_start`/
`page_end`, `keywords`, ...). This is a format contract (ARCHITECTURE.md #9),
not a code dependency -- `vectorize` is never imported.

Typical use::

    from medrag.api.retrieval.modules.top_n import build_default_retriever

    retriever = build_default_retriever()          # env-driven (on the inference machine)
    results = await retriever.retrieve("de1000 operating temperature", k=5)

In tests, construct ``TopNRetriever`` directly with a fake ``Embedder`` and a
fake ``VectorStore`` -- no network, no GPU here (ARCHITECTURE.md #10).
"""

from medrag.api.retrieval.modules.top_n.embedder import (
    Embedder,
    OpenAICompatEmbedder,
    ProviderError,
    embedder_from_env,
)
from medrag.api.retrieval.modules.top_n.factory import build_default_retriever
from medrag.api.retrieval.modules.top_n.retriever import TopNRetriever
from medrag.api.retrieval.modules.top_n.store import QdrantVectorStore, VectorStore

__all__ = [
    "Embedder",
    "OpenAICompatEmbedder",
    "ProviderError",
    "QdrantVectorStore",
    "TopNRetriever",
    "VectorStore",
    "build_default_retriever",
    "embedder_from_env",
]
