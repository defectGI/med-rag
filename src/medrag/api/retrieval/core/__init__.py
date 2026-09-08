"""retrieval.core — shared interfaces and ingestion-contract schemas.

Every module depends only on this package (ARCHITECTURE.md #2). Kept small and
stable; no module-specific logic leaks in here.
"""

from medrag.api.retrieval.core.interfaces import (
    IntentLabel,
    IntentResult,
    RetrievalResult,
    Retriever,
    run_sync,
)
from medrag.api.retrieval.core.schemas import (
    ChunkFile,
    ChunkNode,
    DocumentFile,
    DocumentNode,
    ImageRef,
    Product,
    ProductFile,
    ProductNode,
    iter_chunk_files,
    load_chunk_file,
    load_chunks_dir,
    load_document_file,
    load_product_file,
)

# RUF022: sorted alphabetically; grouping comments dropped (each name's source
# module is already clear from the import blocks above).
__all__ = [
    "ChunkFile",
    "ChunkNode",
    "DocumentFile",
    "DocumentNode",
    "ImageRef",
    "IntentLabel",
    "IntentResult",
    "Product",
    "ProductFile",
    "ProductNode",
    "RetrievalResult",
    "Retriever",
    "iter_chunk_files",
    "load_chunk_file",
    "load_chunks_dir",
    "load_document_file",
    "load_product_file",
    "run_sync",
]
