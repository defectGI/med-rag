"""In-document cross-reference resolution.

The `core/engine.py` output produces `CrossRef.target_chunk_id=None`
candidates; this module binds them to their real targets. Resolution uses
ONLY `Heading.anchor_id` (the source's own anchor/bookmark id) — `CrossRef.link`
is already the source's raw `href`, and that's the only data that reliably
binds it to a heading.

`anchor_id` is currently absent from the first_parse IR (same category of
open item as `header_rows`). Until that field exists, this module does
NOT heuristically match by heading text (slugify, fuzzy, etc.) — the
wrong-binding risk was explicitly rejected. Each unresolved candidate
stays `target_chunk_id=None` and is logged; nothing is silently swallowed.

Multiple headings sharing the same `anchor_id` (parser bug/collision) are
treated as ambiguous: the anchor is dropped from the map and every link
to that anchor is logged as unresolvable (the "treat as unresolvable"
choice).
"""

from __future__ import annotations

import logging

from medrag.pipeline.chunker.core.chunk import ChunkSet
from medrag.pipeline.chunker.core.document import Document

logger = logging.getLogger(__name__)


def _anchor_to_heading_id(doc: Document) -> dict[str, str]:
    """anchor_id -> heading block id. Colliding anchor_ids are dropped."""
    eslesme: dict[str, str] = {}
    belirsiz: set[str] = set()
    for block in doc.blocks:
        if block.kind == "heading" and block.anchor_id:
            if block.anchor_id in eslesme:
                belirsiz.add(block.anchor_id)
            else:
                eslesme[block.anchor_id] = block.id
    for anchor_id in belirsiz:
        del eslesme[anchor_id]
    return eslesme


def _block_id_to_node_id(chunk_set: ChunkSet) -> dict[str, str]:
    """Source block id -> the leaf node id that first carries it."""
    eslesme: dict[str, str] = {}
    for node in chunk_set.leaves():
        for block_id in node.source_block_ids:
            eslesme.setdefault(block_id, node.node_id)
    return eslesme


def resolve_cross_refs(chunk_set: ChunkSet, doc: Document) -> ChunkSet:
    """Resolve `target_chunk_id=None` candidates via `Heading.anchor_id`.

    `chunk_set` must be the output of `chunk_document(doc, ...)` — otherwise
    block-id matches are meaningless. Already-resolved entries
    (`target_chunk_id` filled) are left alone — idempotent.
    """
    anchor_to_heading = _anchor_to_heading_id(doc)
    block_to_node = _block_id_to_node_id(chunk_set)

    yeni_dugumler = []
    for node in chunk_set.nodes:
        if not node.cross_refs:
            yeni_dugumler.append(node)
            continue

        yeni_refs = []
        degisti = False
        for ref in node.cross_refs:
            if ref.target_chunk_id is not None:
                yeni_refs.append(ref)
                continue
            heading_id = anchor_to_heading.get(ref.link[1:])
            hedef = block_to_node.get(heading_id) if heading_id else None
            if hedef is None:
                logger.warning(
                    "Unresolvable in-document link: %r (source block %s, node %s)",
                    ref.link, ref.source_block_id, node.node_id)
                yeni_refs.append(ref)
            else:
                yeni_refs.append(ref.model_copy(update={"target_chunk_id": hedef}))
                degisti = True

        yeni_dugumler.append(
            node.model_copy(update={"cross_refs": yeni_refs}) if degisti else node)

    return chunk_set.model_copy(update={"nodes": yeni_dugumler})