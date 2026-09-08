"""Ingestion contract schemas.

These pydantic models define the *format contract* for the JSON produced by an
upstream ingestion/parsing project (the reference producer is `medrag`,
but this layer never imports it — see ARCHITECTURE.md #9). Any project that
emits data in these shapes can feed this retrieval layer.

Design stance (ARCHITECTURE.md #2, "core small and stable"): we model only the
fields this retrieval layer actually consumes and set ``extra="ignore"`` so that
producer-side additions or internal bookkeeping fields never break validation.
This keeps the contract resilient to upstream format drift.

Field origins (verified against real producer output):
- ``ChunkNode``    <- ``*.chunks.json`` -> ``nodes[]``
- ``ProductNode``  <- ``product_nodes.json`` -> ``product_nodes[]``
- ``DocumentNode`` <- ``document_nodes.json`` -> ``documents[]``
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    """Base for every ingestion-contract model.

    ``extra="ignore"`` is deliberate: the producer emits many internal fields
    (block ids, flex bookkeeping, provenance, parser metadata) that this layer
    does not consume. Ignoring them means an upstream schema addition is a
    no-op here instead of a validation failure.
    """

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# chunks.json  (one file per document)
# ---------------------------------------------------------------------------


class ImageRef(_Contract):
    """Reference to an image extracted alongside a chunk."""

    image_id: str


class ChunkNode(_Contract):
    """A single retrievable text chunk of a document.

    This is the primary unit indexed by text-based retrieval (top_n, raptor).
    ``tree_level`` / ``child_ids`` carry the hierarchy RAPTOR will later use;
    ``prev_node_id`` / ``next_node_id`` allow window expansion around a hit.
    """

    node_id: str
    doc_id: str
    text: str

    tree_level: int = 0
    keywords: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)

    # Provenance for citation / display.
    source_path: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    token_count: int | None = None
    images: list[ImageRef] = Field(default_factory=list)

    prev_node_id: str | None = None
    next_node_id: str | None = None


class ChunkFile(_Contract):
    """Top-level shape of a single ``*.chunks.json`` file."""

    schema_version: int
    doc_id: str
    nodes: list[ChunkNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# product_nodes.json  (structured product graph)
# ---------------------------------------------------------------------------


class ProductCategory(_Contract):
    reference_code: str | None = None


class Product(_Contract):
    """The leaf payload on a ``type == "product"`` node."""

    product_code: str
    acme_code: str | None = None
    display_name: str | None = None

    list_price: float | None = None
    price_on_request: bool = False
    valid_until: str | None = None

    price_break_10_99: float | None = None
    price_break_100_499: float | None = None
    price_break_500_999: float | None = None

    is_variant: bool = False
    variant_base: str | None = None


class ProductNode(_Contract):
    """A node in the product hierarchy: a family, subfamily, or a product.

    ``product`` is populated only when ``type == "product"`` (``is_leaf`` true).
    ``type`` is kept as a plain string (not an enum) for drift resilience.
    """

    node_id: str
    type: str  # "family" | "subfamily" | "product"
    source_ref: str | None = None
    parent_id: str | None = None
    depth: int | None = None
    is_leaf: bool = False

    family: str | None = None
    subfamily: str | None = None
    subfamily_2: str | None = None

    category: ProductCategory | None = None
    product: Product | None = None


class ProductFile(_Contract):
    """Top-level shape of ``product_nodes.json``."""

    generated_at: str | None = None
    summary: dict = Field(default_factory=dict)
    product_nodes: list[ProductNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# document_nodes.json  (document metadata + file locations)
# ---------------------------------------------------------------------------


class DocumentIdentity(_Contract):
    doc_id: str
    file_name: str | None = None
    extension: str | None = None


class DocumentLocation(_Contract):
    rel_path: str | None = None
    file_path: str | None = None


class DocumentScan(_Contract):
    content_hash: str | None = None
    size_bytes: int | None = None
    last_modified_time: str | None = None
    scan_status: str | None = None


class DocumentLinks(_Contract):
    """Links from a document to the product nodes it belongs to."""

    owner_ids: list[str] = Field(default_factory=list)
    link_count: int | None = None
    resolution_method: str | None = None


class DocumentNode(_Contract):
    """Metadata for a single source document (datasheet, brochure, image, ...).

    ``doc_type`` is a plain string (not an enum) so new producer doc types do
    not break validation. Known values observed: BROCHURE, DATASHEET,
    PRODUCT_IMAGE, CATALOGUE, CE_DECLARATION, TECHNICAL_DRAWING, STP,
    USER_MANUAL, QUICK_START_GUIDE.
    """

    identity: DocumentIdentity
    location: DocumentLocation = Field(default_factory=DocumentLocation)
    scan: DocumentScan = Field(default_factory=DocumentScan)
    doc_type: str | None = None
    is_active: bool = True
    links: DocumentLinks = Field(default_factory=DocumentLinks)


class DocumentFile(_Contract):
    """Top-level shape of ``document_nodes.json``."""

    generated_at: str | None = None
    summary: dict = Field(default_factory=dict)
    documents: list[DocumentNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_chunk_file(path: str | Path) -> ChunkFile:
    """Load and validate a single ``*.chunks.json`` file."""
    return ChunkFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_product_file(path: str | Path) -> ProductFile:
    """Load and validate a ``product_nodes.json`` file."""
    return ProductFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_document_file(path: str | Path) -> DocumentFile:
    """Load and validate a ``document_nodes.json`` file."""
    return DocumentFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def iter_chunk_files(directory: str | Path):
    """Yield ``ChunkFile`` for every ``*.chunks.json`` under ``directory``."""
    for p in sorted(Path(directory).glob("*.chunks.json")):
        yield load_chunk_file(p)


def load_chunks_dir(directory: str | Path) -> list[ChunkNode]:
    """Flatten every chunk node across all ``*.chunks.json`` in a directory."""
    nodes: list[ChunkNode] = []
    for cf in iter_chunk_files(directory):
        nodes.extend(cf.nodes)
    return nodes


__all__ = [
    "ChunkFile",
    "ChunkNode",
    "DocumentFile",
    "DocumentIdentity",
    "DocumentLinks",
    "DocumentLocation",
    "DocumentNode",
    "DocumentScan",
    "ImageRef",
    "Product",
    "ProductCategory",
    "ProductFile",
    "ProductNode",
    "iter_chunk_files",
    "load_chunk_file",
    "load_chunks_dir",
    "load_document_file",
    "load_product_file",
]
