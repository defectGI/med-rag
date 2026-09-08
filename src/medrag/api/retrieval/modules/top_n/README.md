# top_n

Status: **usable** (v1: dense retrieval only, over Qdrant).

Vector similarity search over an existing Qdrant collection. Two seams:

- `Embedder` (query -> vector) -- shipped `OpenAICompatEmbedder` talks to any
  `POST {base}/embeddings` server (ollama/openai/openrouter/local); shared
  `EMBEDDING_*` env with `raptor`.
- `VectorStore` (vector -> scored hits) -- shipped `QdrantVectorStore` is
  read-only (search only). This module never builds/writes the collection;
  it queries one that already exists.

```python
from retrieval.modules.top_n import build_default_retriever

retriever = build_default_retriever()   # QDRANT_*/EMBEDDING_* env
results = await retriever.retrieve("de1000 operating temperature", k=5)
```

**The query MUST be embedded with the same model that wrote the vectors** --
a mismatched model produces meaningless similarity, and a mismatched vector
size makes Qdrant reject the search outright.

**Payload shape is a format contract, not a code dependency.** Any producer
writing Qdrant points with `text`/`node_id`/`doc_id`/`heading_path`/... in the
payload can be queried here (ARCHITECTURE.md #9) -- in this project's
reference pipeline that producer is `vectorize`, but it is never imported.

Not yet implemented: sparse/keyword-combined search, corpus filtering by
`doc_id`/`_profile.*` scope (Qdrant's own filter API would carry this).

Extra: `pip install -e ".[top_n]"`
