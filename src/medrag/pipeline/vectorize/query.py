"""Read-only search helper: embeds a question text and fetches the nearest N
chunks (full metadata + score) from Qdrant.

Usage: ``python -m vectorize.query "question text"`` [--k 5] [--json]

COMPLETELY SEPARATE from `cli.py`'s write side: the concrete, working example
of how of a consuming project connects to Qdrant. Configured from the same
`EMBEDDING_*`/`QDRANT_*` `.env` values -- the question is vectorized with the
SAME embedding model (a different model would mismatch on size/meaning; see
the `.env.example` note).

`search()` is for programmatic use -- a chatbot/retrieval layer can import this
module and call it directly; the CLI is not mandatory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping

from medrag.pipeline.vectorize.config import load_config
from medrag.pipeline.vectorize.embedder import ProviderError, embedder_from_env
from medrag.pipeline.vectorize.store import QdrantVectorStore

log = logging.getLogger("vectorize.query")


def search(query: str, *, k: int = 5, env: Mapping[str, str] = os.environ) -> list[dict]:
    """Configures from `.env`, embeds `query`, and returns the nearest `k`
    results (`score` + all payload fields: `text`, `doc_id`, `node_id`,
    `heading_path`, `page_start`/`page_end`, `keywords`, ...)."""
    cfg = load_config(env.get("VECTORIZE_CONFIG") or None)
    embedder = embedder_from_env(env, timeout=cfg.embedding.timeout_seconds)

    qdrant_url = (env.get("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise ProviderError("QDRANT_URL unset (see .env.example)")

    store = QdrantVectorStore.from_env(
        collection_name=cfg.qdrant.collection_name, distance=cfg.qdrant.distance,
        on_dim_mismatch=cfg.qdrant.on_dim_mismatch,
        upsert_batch_size=cfg.qdrant.upsert_batch_size,
        url=qdrant_url, api_key=env.get("QDRANT_API_KEY") or None)

    vector = embedder.embed([query])[0]
    return store.search(vector, limit=k)


def _render_human(results: list[dict]) -> None:
    if not results:
        print("(no results)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n#{i}  score={r.get('score', 0):.4f}  doc_id={r.get('doc_id')}"
              f"  node_id={r.get('node_id')}")
        if r.get("heading_path"):
            print(f"    heading_path: {' > '.join(r['heading_path'])}")
        if r.get("page_start") is not None:
            print(f"    page: {r['page_start']}-{r.get('page_end')}")
        if r.get("keywords"):
            print(f"    keywords: {', '.join(r['keywords'])}")
        if r.get("source_path"):
            print(f"    source: {r['source_path']}")
        print(f"    text:\n{r.get('text', '')}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vectorize.query")
    parser.add_argument("query", help="text to search for")
    parser.add_argument("--k", type=int, default=5,
                        help="number of results to return (default 5)")
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv()

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        results = search(args.query, k=args.k)
    except ProviderError as exc:
        log.error("%s", exc)
        return 2
    except Exception:
        log.exception("search failed")
        return 1

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _render_human(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())