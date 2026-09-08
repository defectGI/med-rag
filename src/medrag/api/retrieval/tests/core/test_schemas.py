"""Schema contract tests.

Fixtures are hand-authored minimal shapes that mirror the producer format —
medrag output is never copied into this repo (ARCHITECTURE.md #5/#9).
Each fixture includes an unmodelled extra field to prove ``extra="ignore"``.
"""

from __future__ import annotations

import json

from medrag.api.retrieval.core import schemas


def test_chunk_file_parses_and_ignores_extras():
    raw = {
        "schema_version": 1,
        "doc_id": "DOC_X",
        "generator_debug": {"unused": True},  # extra at file level
        "nodes": [
            {
                "node_id": "DOC_X::c0",
                "doc_id": "DOC_X",
                "text": "hello world",
                "heading_path": ["Intro"],
                "page_start": 1,
                "page_end": 1,
                "images": [{"image_id": "img_abc"}],
                "next_node_id": "DOC_X::c1",
                # producer internals we don't model:
                "source_block_ids": ["b0", "b1"],
                "flex_applied": False,
                "provenance_summary": "verified",
            }
        ],
    }
    cf = schemas.ChunkFile.model_validate(raw)
    assert cf.doc_id == "DOC_X"
    assert len(cf.nodes) == 1
    node = cf.nodes[0]
    assert node.text == "hello world"
    assert node.heading_path == ["Intro"]
    assert node.images[0].image_id == "img_abc"
    assert node.next_node_id == "DOC_X::c1"
    # unmodelled fields dropped, not retained
    assert not hasattr(node, "source_block_ids")


def test_product_file_leaf_and_container_nodes():
    raw = {
        "generated_at": "2026-07-07T10:07:54+03:00",
        "summary": {"products": 1},
        "product_nodes": [
            {
                "node_id": "n_fam",
                "type": "family",
                "parent_id": None,
                "depth": 1,
                "is_leaf": False,
                "family": "PXI EXPRESS SYSTEMS",
            },
            {
                "node_id": "n_prod",
                "type": "product",
                "parent_id": "n_fam",
                "depth": 3,
                "is_leaf": True,
                "family": "PXI EXPRESS SYSTEMS",
                "product": {
                    "product_code": "PN1162",
                    "acme_code": "DC1052800001",
                    "display_name": "AMD Kria SoM PCIe Development Kit",
                    "list_price": 2150.0,
                    "price_on_request": False,
                },
            },
        ],
    }
    pf = schemas.ProductFile.model_validate(raw)
    assert len(pf.product_nodes) == 2
    fam, prod = pf.product_nodes
    assert fam.product is None
    assert prod.product is not None
    assert prod.product.product_code == "PN1162"
    assert prod.product.list_price == 2150.0


def test_document_file_nested_shape():
    raw = {
        "generated_at": "2026-07-06T09:16:18+03:00",
        "summary": {"scanned_files": 1},
        "documents": [
            {
                "identity": {
                    "doc_id": "a4b5dce3",
                    "file_name": "ACME_PN5029_Brochure.pdf",
                    "extension": ".pdf",
                },
                "location": {"rel_path": "AVIONICS/ACME_PN5029_Brochure.pdf"},
                "scan": {"content_hash": "sha256:abc", "scan_status": "NEW"},
                "doc_type": "BROCHURE",
                "is_active": True,
                "links": {"owner_ids": ["n_5d09"], "link_count": 1},
                "parse": {"status": "SUCCESS"},  # extra sub-object, ignored
            }
        ],
    }
    df = schemas.DocumentFile.model_validate(raw)
    doc = df.documents[0]
    assert doc.identity.doc_id == "a4b5dce3"
    assert doc.doc_type == "BROCHURE"
    assert doc.links.owner_ids == ["n_5d09"]


def test_loaders_roundtrip(tmp_path):
    p = tmp_path / "DOC_Y.chunks.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "doc_id": "DOC_Y",
                "nodes": [{"node_id": "DOC_Y::c0", "doc_id": "DOC_Y", "text": "t"}],
            }
        ),
        encoding="utf-8",
    )
    cf = schemas.load_chunk_file(p)
    assert cf.doc_id == "DOC_Y"

    nodes = schemas.load_chunks_dir(tmp_path)
    assert [n.node_id for n in nodes] == ["DOC_Y::c0"]
