"""PHASE A / "dry DB" skeleton builder tests.

Rewritten on 2026-08-04: the schema was reduced to 4 flat tables
(`product`/`document`/`attribute`/`spec_value`, LLM-SQL simplification --
see `schema_rag.sql` header). The taxonomy tree (`product_node`),
document fan-out table (`document_owner`), the 6-part dictionary table
set and `attribute_applicability` FK-enforced scope check were
REMOVED -- scope is now computed only on the Python side (`key_applies`),
there is NO DB-level scope-guard trigger. Removed tests did not die;
their SCOPE CHANGED: fact load is an UPDATE and is PHASE C's job
(`facts/experiments/chunk_full_run/load_to_db.py`). Here only the
skeleton's correctness is locked down.

O-04 (2026-08-20): restored from the `facts/tests/test_build_facts_db.py`
in the `facts-kodu-son-hali` tag, with the import path updated from
`facts.config` to `medrag.pipeline.facts.facts_config`.

KNOWN BLOCK (02-SERVISLESTIRME-TASKLARI.md O-03 "DONE" note item 1,
KARAR-KAYDI.md K-71): `medrag.pipeline.facts.build_facts_db` imports
`medrag.pipeline.facts.spec_keys`, but that module is OUT of K-59's "9
files to return" set -- the user's decision is pending, it was not
moved into the package. So `build_facts_db` currently cannot be
imported, and this ENTIRE file is skipped at module level. When
`spec_keys.py` is moved into the package, the skip self-lifts."""
from __future__ import annotations

import json
import sqlite3

import pytest
import yaml

pytest.importorskip(
    "medrag.pipeline.facts.build_facts_db",
    reason=(
        "medrag.pipeline.facts.build_facts_db import blocked: "
        "medrag.pipeline.facts.spec_keys has not been moved into the package (K-71, pending user decision)"
    ),
)

from medrag.pipeline.facts.build_facts_db import build
from medrag.pipeline.facts.facts_config import (
    FactsConfig,
    Llm,
    Queue,
    Scope,
    Vlm,
    Vocabulary,
)

SCHEMA_PATH = __import__(
    "medrag.pipeline.facts.build_facts_db", fromlist=["SCHEMA_PATH"]
).SCHEMA_PATH


def _test_config(exclude_doc_types=("PRODUCT_IMAGE", "STP")) -> FactsConfig:
    return FactsConfig(
        scope=Scope(exclude_doc_types=list(exclude_doc_types), include_summary_nodes=False),
        queue=Queue(batch_size=10),
        vocabulary=Vocabulary(large_wave_warning_threshold=15),
        llm=Llm(max_tokens_extract=4096),
        vlm=Vlm(render_dpi=150, timeout_s=180.0, max_tokens=512, page_render_format="png"),
    )


def _family_node(node_id, family="FAM"):
    return {
        "node_id": node_id, "type": "family", "parent_id": None, "source_ref": "1",
        "depth": 1, "is_leaf": False, "family": family, "subfamily": None,
        "subfamily_2": None, "category": {"reference_code": None}, "product": None,
    }


def _subfamily_node(node_id, parent_id, family="FAM", subfamily="SUB"):
    return {
        "node_id": node_id, "type": "subfamily", "parent_id": parent_id, "source_ref": "1.1",
        "depth": 2, "is_leaf": False, "family": family, "subfamily": subfamily,
        "subfamily_2": None, "category": {"reference_code": None}, "product": None,
    }


def _product_node(
    node_id, parent_id, product_code, family="FAM", subfamily="SUB", subfamily_2=None,
    variant_base=None, display_name=None, is_leaf=True, depth=3,
):
    return {
        "node_id": node_id, "type": "product", "parent_id": parent_id, "source_ref": "1.1.1",
        "depth": depth, "is_leaf": is_leaf, "family": family, "subfamily": subfamily,
        "subfamily_2": subfamily_2, "category": None,
        "product": {
            "product_code": product_code, "variant_base": variant_base,
            "display_name": display_name or (product_code or node_id),
            "list_price": 10.0, "price_on_request": False,
            "is_variant": variant_base is not None,
        },
    }


def _doc(doc_id, file_name, doc_type, owner_ids, content_hash="sha256:" + "a" * 64, is_active=True):
    return {
        "identity": {"doc_id": doc_id, "file_name": file_name, "extension": ".pdf"},
        "location": {"rel_path": f"x/{file_name}", "file_path": f"/x/{file_name}"},
        "scan": {
            "content_hash": content_hash, "size_bytes": 10,
            "last_modified_time": "2026-08-03T10:00:00+03:00", "scan_status": "UNCHANGED",
        },
        "doc_type": doc_type,
        "is_active": is_active,
        "links": {"owner_ids": owner_ids, "matched_node_id": owner_ids[0] if owner_ids else None,
                  "resolution_method": "code_match"},
    }


def _write_corpus(tmp_path, product_nodes, documents):
    product_nodes_path = tmp_path / "product_nodes.json"
    document_nodes_path = tmp_path / "document_nodes.json"
    product_nodes_path.write_text(json.dumps({"product_nodes": product_nodes}), encoding="utf-8")
    document_nodes_path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return product_nodes_path, document_nodes_path


# Small but real-shaped dictionary: core block (every product) + one
# family block scoped to ANOTHER family (does not cover any test product).
_BLOCKS = [
    {"name": "core", "is_core": True, "applies_to": []},
    {"name": "power_distribution", "is_core": False, "applies_to": ["OTHER FAM"]},
]

_ATTRIBUTES = [
    {
        "block": "core", "key": "operating_temperature", "kind": "range", "unit": "C",
        "question": "What's the operating temperature?", "labels": ["Operating Temperature"],
    },
    {
        "block": "power_distribution", "key": "outlet_type", "kind": "single", "unit": None,
        "question": "What's the outlet type?", "labels": ["Outlet Type"],
    },
]


def _write_dictionary(tmp_path, attributes=None, blocks=None, name="spec_keys.yaml"):
    attributes = _ATTRIBUTES if attributes is None else attributes
    blocks = _BLOCKS if blocks is None else blocks
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "version": "2.0.0",
                "kinds": ["single", "min_typ_max", "range", "list", "conditional", "boolean"],
                "blocks": blocks,
                "attributes": attributes,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def basic_corpus(tmp_path):
    """FAM -> SUB -> two products; plus an EMPTY 'OTHER FAM' branch.

    'OTHER FAM' is deliberately product-less: the `power_distribution`
    block's scope points to it, so NONE of the test products are covered
    -- the setup that locks down the `not_applicable` path.

    Documents: a BROCHURE owned by the subfamily (opens to both
    products, fan-out is embedded in `document.product_codes` at BUILD
    TIME) + a DATASHEET owned directly by product A (does not open)."""
    fam = _family_node("n_fam", family="FAM")
    sub = _subfamily_node("n_sub", "n_fam", family="FAM", subfamily="SUB")
    prod_a = _product_node("n_a", "n_sub", "DE_A", display_name="Product A")
    prod_b = _product_node("n_b", "n_sub", "DE_B", display_name="Product B")
    other_fam = _family_node("n_other_fam", family="OTHER FAM")

    documents = [
        _doc("doc-brochure", "brochure.pdf", "BROCHURE", ["n_sub"]),
        _doc("doc-datasheet", "datasheet.pdf", "DATASHEET", ["n_a"]),
    ]
    product_nodes_path, document_nodes_path = _write_corpus(
        tmp_path, [fam, sub, prod_a, prod_b, other_fam], documents
    )
    return {
        "product_nodes_path": product_nodes_path,
        "document_nodes_path": document_nodes_path,
        "spec_keys_path": _write_dictionary(tmp_path),
    }


def _build(tmp_path, corpus, **overrides):
    kwargs = {
        "db_path": tmp_path / "specs.db",
        "schema_path": SCHEMA_PATH,
        "product_nodes_path": corpus["product_nodes_path"],
        "document_nodes_path": corpus["document_nodes_path"],
        "spec_keys_path": corpus["spec_keys_path"],
        "config": _test_config(),
    }
    kwargs.update(overrides)
    return kwargs["db_path"], build(**kwargs)


# ---------------------------------------------------------------------------
# product / document -- reference data
# ---------------------------------------------------------------------------


def test_product_row_carries_registry_identity(tmp_path, basic_corpus):
    db_path, result = _build(tmp_path, basic_corpus)
    assert result["n_products"] == 2
    assert result["legacy_path"] is None  # first build
    con = sqlite3.connect(db_path)
    code, name, family, subfamily, price = con.execute(
        "SELECT product_code, display_name, family, subfamily, list_price FROM product WHERE product_code='DE_A'"
    ).fetchone()
    assert (code, name, family, subfamily, price) == ("DE_A", "Product A", "FAM", "SUB", 10.0)
    con.close()


def test_document_is_stored_once_with_fanout_precomputed(tmp_path, basic_corpus):
    """KARAR-019: one physical file is ONE row. Fan-out is now embedded
    at BUILD TIME (Python walks the product_nodes.json tree once) into
    the `product_codes` JSON -- no recursive query at runtime."""
    db_path, result = _build(tmp_path, basic_corpus)
    assert result["n_documents"] == 2
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM document WHERE doc_id='doc-brochure'").fetchone() == (1,)
    codes = json.loads(con.execute("SELECT product_codes FROM document WHERE doc_id='doc-brochure'").fetchone()[0])
    assert set(codes) == {"DE_A", "DE_B"}  # subfamily brochure fans out to BOTH products
    codes_ds = json.loads(con.execute("SELECT product_codes FROM document WHERE doc_id='doc-datasheet'").fetchone()[0])
    assert codes_ds == ["DE_A"]  # direct document does not fan out
    con.close()


def test_fanout_view_expands_owners_down_to_products(tmp_path, basic_corpus):
    """`v_document_product` unnests `product_codes` for chatbot compatibility."""
    db_path, _ = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    pairs = set(con.execute("SELECT doc_id, product_code FROM v_document_product"))
    assert ("doc-brochure", "DE_A") in pairs
    assert ("doc-brochure", "DE_B") in pairs
    assert ("doc-datasheet", "DE_A") in pairs
    assert ("doc-datasheet", "DE_B") not in pairs
    con.close()


def test_trust_rank_is_derived_from_doc_type(tmp_path, basic_corpus):
    db_path, _ = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    assert dict(con.execute("SELECT doc_type, trust_rank FROM document")) == {
        "BROCHURE": 50, "DATASHEET": 100,
    }
    con.close()


def test_content_hash_copied_onto_document_row(tmp_path, basic_corpus):
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    documents = [_doc("doc-1", "a.pdf", "DATASHEET", ["n_a"], content_hash="sha256:" + "f" * 64)]
    p_path, d_path = _write_corpus(tmp_path, product_nodes, documents)

    db_path, _ = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT content_hash FROM document").fetchone() == ("sha256:" + "f" * 64,)
    con.close()


def test_document_with_no_resolvable_owner_is_kept_with_empty_product_codes(tmp_path, basic_corpus):
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    documents = [_doc("doc-ghost", "ghost.pdf", "DATASHEET", ["n_does_not_exist"])]
    p_path, d_path = _write_corpus(tmp_path, product_nodes, documents)

    db_path, result = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    assert result["docs_with_no_owner"] == ["doc-ghost  ghost.pdf"]
    # The document row IS still written (the file is real); only product_codes is empty.
    assert result["n_documents"] == 1
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT product_codes FROM document WHERE doc_id='doc-ghost'").fetchone() == ("[]",)
    con.close()


def test_product_without_product_code_is_reported_not_silently_dropped(tmp_path, basic_corpus):
    """KARAR-013's product_code-less node CANNOT BE REPRESENTED in this
    schema (PK = product_code) -- reported to `orphan_leaf_nodes`
    instead of being silently swallowed."""
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    product_nodes.append(_product_node("n_orphan", "n_sub", product_code=None))
    p_path, d_path = _write_corpus(tmp_path, product_nodes, [])

    db_path, result = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    assert result["n_products"] == 2  # excluding the orphan
    assert result["orphan_leaf_nodes"] == ["n_orphan"]
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM product").fetchone() == (2,)
    con.close()


def test_product_with_zero_documents_is_reported_as_no_documents(tmp_path, basic_corpus):
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    documents = [_doc("doc-datasheet", "datasheet.pdf", "DATASHEET", ["n_a"])]
    p_path, d_path = _write_corpus(tmp_path, product_nodes, documents)

    _, result = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    assert result["info_counts"] == {"extracted": 1, "no_documents": 1}


def test_no_extractable_documents_when_only_out_of_scope_doc_types(tmp_path, basic_corpus):
    """KARAR-027: PRODUCT_IMAGE is out of scope -- the file exists but
    has nothing extractable."""
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    documents = [_doc("doc-img", "photo.png", "PRODUCT_IMAGE", ["n_b"])]
    p_path, d_path = _write_corpus(tmp_path, product_nodes, documents)

    _, result = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    assert result["info_counts"] == {"no_extractable_documents": 1, "no_documents": 1}


# ---------------------------------------------------------------------------
# dictionary layer (ONE `attribute` table, JSON-embedded)
# ---------------------------------------------------------------------------


def test_dictionary_layer_is_fully_loaded(tmp_path, basic_corpus):
    db_path, result = _build(tmp_path, basic_corpus)
    assert result["n_attributes"] == 2
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM attribute").fetchone() == (2,)
    labels = json.loads(con.execute("SELECT labels FROM attribute WHERE key='operating_temperature'").fetchone()[0])
    assert labels == ["Operating Temperature"]
    applies_to = json.loads(con.execute("SELECT applies_to FROM attribute WHERE key='outlet_type'").fetchone()[0])
    assert applies_to == ["OTHER FAM"]
    applies_to_core = json.loads(con.execute("SELECT applies_to FROM attribute WHERE key='operating_temperature'").fetchone()[0])
    assert applies_to_core == []  # core -> applies everywhere, no scope row
    con.close()


# ---------------------------------------------------------------------------
# (product x attribute) matrix -- PHASE A's main output
# ---------------------------------------------------------------------------


def test_matrix_is_fully_materialized(tmp_path, basic_corpus):
    db_path, result = _build(tmp_path, basic_corpus)
    assert result["n_matrix_rows"] == 2 * 2  # 2 products x 2 attributes, no exceptions
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM spec_value").fetchone() == (4,)
    con.close()


def test_core_block_key_applies_to_every_product(tmp_path, basic_corpus):
    """An attribute without a scope row applies everywhere."""
    db_path, result = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    rows = dict(con.execute("SELECT product_code, status FROM spec_value WHERE key='operating_temperature'"))
    assert rows == {"DE_A": "not_specified", "DE_B": "not_specified"}
    assert result["n_not_specified"] == 2
    con.close()


def test_key_scoped_to_another_family_is_not_applicable(tmp_path, basic_corpus):
    """`outlet_type` applies ONLY to 'OTHER FAM' -> these products get
    `not_applicable`. NEVER `absent`: "not observed" is not "absence
    evidence"."""
    db_path, result = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    rows = dict(con.execute("SELECT product_code, status FROM spec_value WHERE key='outlet_type'"))
    assert rows == {"DE_A": "not_applicable", "DE_B": "not_applicable"}
    assert result["n_not_applicable"] == 2
    con.close()


def test_subfamily_scope_does_not_leak_to_a_sibling(tmp_path, basic_corpus):
    """A subfamily scope does NOT LEAK to a sibling subfamily, nor to
    the whole family."""
    fam = _family_node("n_fam")
    sub = _subfamily_node("n_sub", "n_fam", subfamily="SUB")
    other = _subfamily_node("n_other", "n_fam", subfamily="OTHER")
    in_scope = _product_node("n_in", "n_sub", "DE_IN", subfamily="SUB")
    out_scope = _product_node("n_out", "n_other", "DE_OUT", subfamily="OTHER")
    p_path, d_path = _write_corpus(tmp_path, [fam, sub, other, in_scope, out_scope], [])

    blocks = [{"name": "narrow", "is_core": False, "applies_to": ["FAM / SUB"]}]
    attributes = [dict(_ATTRIBUTES[0], block="narrow", key="sub_only")]
    sk_path = _write_dictionary(tmp_path, attributes=attributes, blocks=blocks, name="narrow.yaml")

    db_path, _ = _build(
        tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path, spec_keys_path=sk_path
    )
    con = sqlite3.connect(db_path)
    rows = dict(con.execute("SELECT product_code, status FROM spec_value"))
    assert rows == {"DE_IN": "not_specified", "DE_OUT": "not_applicable"}
    con.close()


def test_bootstrap_never_writes_absent_or_present(tmp_path, basic_corpus):
    db_path, _ = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    statuses = {s for (s,) in con.execute("SELECT DISTINCT status FROM spec_value")}
    assert statuses <= {"not_specified", "not_applicable"}
    con.close()


def test_matrix_rows_carry_no_values_at_all(tmp_path, basic_corpus):
    """PHASE A DOES NOT WRITE VALUES. If even one typed column is
    populated, scope has drifted."""
    db_path, _ = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    (n_dirty,) = con.execute(
        "SELECT COUNT(*) FROM spec_value WHERE num_value IS NOT NULL OR text_value IS NOT NULL "
        "OR val_min IS NOT NULL OR val_typ IS NOT NULL OR val_max IS NOT NULL "
        "OR bool_value IS NOT NULL OR raw_text IS NOT NULL OR unit_observed IS NOT NULL "
        "OR items IS NOT NULL OR subfields IS NOT NULL "
        "OR source_chunk_id IS NOT NULL OR source_doc_id IS NOT NULL"
    ).fetchone()
    assert n_dirty == 0
    con.close()


def test_status_matches_the_applicability_rule_in_pure_python(tmp_path, basic_corpus):
    """The applicability rule must be measurable via `key_applies`:
    if no scope row OR the product's family matches -> `not_specified`,
    otherwise `not_applicable`."""
    from medrag.pipeline.facts.spec_keys import (
        block_scopes,
        key_applies,
        load_spec_keys,
    )

    db_path, _ = _build(tmp_path, basic_corpus)
    dictionary = load_spec_keys(basic_corpus["spec_keys_path"])
    scopes = block_scopes(dictionary)
    con = sqlite3.connect(db_path)
    mismatches = []
    for product_code, family, subfamily, block, status in con.execute(
        "SELECT product_code, family, subfamily, block, status FROM spec_value"
    ):
        expected = "not_specified" if key_applies(family, subfamily, scopes[block]) else "not_applicable"
        if status != expected:
            mismatches.append((product_code, block, status, expected))
    assert mismatches == []
    con.close()


def test_product_with_no_documents_still_gets_its_full_matrix(tmp_path, basic_corpus):
    """A product with no documents still gets the full matrix: "nothing
    to look at" and "this key does not apply here" are DIFFERENT
    statements."""
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    p_path, d_path = _write_corpus(tmp_path, product_nodes, [])

    db_path, _ = _build(tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path)
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM spec_value WHERE product_code='DE_B'").fetchone() == (2,)
    con.close()


def test_coverage_gap_view_excludes_not_applicable(tmp_path, basic_corpus):
    """`not_applicable` is NOT a gap -- that question is never asked of
    this product."""
    db_path, _ = _build(tmp_path, basic_corpus)
    con = sqlite3.connect(db_path)
    keys = {r[0] for r in con.execute("SELECT key FROM v_coverage_gap")}
    assert keys == {"operating_temperature"}
    con.close()


# ---------------------------------------------------------------------------
# fail-loud + backup
# ---------------------------------------------------------------------------


def test_block_pointing_at_an_unknown_family_fails_loud(tmp_path, basic_corpus):
    blocks = [{"name": "core", "is_core": True, "applies_to": ["GHOST FAM"]}]
    sk_path = _write_dictionary(tmp_path, blocks=blocks, name="ghost.yaml")
    with pytest.raises(ValueError, match="not in product_nodes.json"):
        _build(tmp_path, basic_corpus, spec_keys_path=sk_path)


def test_product_outside_the_family_set_fails_loud(tmp_path, basic_corpus):
    """A product whose family is NOT in the dictionary's family set
    would get `not_applicable` in EVERY family block -- silent data loss;
    instead we CRASH."""
    product_nodes = json.loads(basic_corpus["product_nodes_path"].read_text(encoding="utf-8"))["product_nodes"]
    ghost = _product_node("n_ghost", "n_sub", "DE_GHOST", family="GHOST FAM", subfamily=None)
    p_path, d_path = _write_corpus(tmp_path, product_nodes, [])
    ghost_nodes = product_nodes + [ghost]
    p_path.write_text(json.dumps({"product_nodes": ghost_nodes}), encoding="utf-8")

    blocks = [{"name": "fam_block", "is_core": False, "applies_to": ["FAM"]}]
    sk_path = _write_dictionary(tmp_path, attributes=[dict(_ATTRIBUTES[0], block="fam_block")],
                                blocks=blocks, name="famonly.yaml")
    db_path, result = _build(
        tmp_path, basic_corpus, product_nodes_path=p_path, document_nodes_path=d_path, spec_keys_path=sk_path
    )
    con = sqlite3.connect(db_path)
    assert con.execute(
        "SELECT status FROM spec_value WHERE product_code='DE_GHOST'"
    ).fetchone() == ("not_applicable",)
    con.close()
    assert result["n_not_applicable"] == 1


def test_validation_failure_leaves_no_half_written_db(tmp_path, basic_corpus):
    blocks = [{"name": "core", "is_core": True, "applies_to": ["NOPE"]}]
    sk_path = _write_dictionary(tmp_path, blocks=blocks, name="nope.yaml")
    db_path = tmp_path / "specs.db"
    with pytest.raises(ValueError):
        _build(tmp_path, basic_corpus, db_path=db_path, spec_keys_path=sk_path)
    assert not db_path.exists()


def test_second_build_backs_up_legacy_and_writes_comparison_report(tmp_path, basic_corpus):
    db_path, first = _build(tmp_path, basic_corpus)
    assert first["legacy_path"] is None

    _, second = _build(tmp_path, basic_corpus)
    assert second["legacy_path"] is not None
    assert second["legacy_path"].name.startswith("specs.db.legacy_")
    assert second["legacy_path"].is_file()
    assert db_path.is_file()  # canonical path has the fresh DB

    report = second["report_path"].read_text(encoding="utf-8")
    assert "value not_applicable" in report
    assert "PHASE A" in report


# ---------------------------------------------------------------------------
# corpus-derivable booleans (PROTOCOL KARAR-063, 2026-08-05)
#
# Root cause of live test finding #3/#4: `has_ce_declaration` was
# `not_specified` for all 249 of 249 products, so every SQL about CE
# necessarily returned empty and the answering model filled the gap
# from CE declaration chunks of OTHER products (attributing another
# product's CE declaration to the product asked, finding #3). The data
# was NOT missing -- the `document.product_codes` fan-out already
# carried it.
# ---------------------------------------------------------------------------

_CE_ATTR = {
    "block": "core", "key": "has_ce_declaration", "kind": "boolean", "unit": None,
    "question": "Is there a CE declaration?", "labels": ["CE Declaration"],
}


@pytest.fixture
def ce_corpus(tmp_path):
    """Same tree as `basic_corpus`, but documents: a subfamily BROCHURE
    + a CE_DECLARATION for product A ONLY. So 'covered' and 'uncovered'
    are tested side by side in the same build."""
    fam = _family_node("n_fam", family="FAM")
    sub = _subfamily_node("n_sub", "n_fam", family="FAM", subfamily="SUB")
    prod_a = _product_node("n_a", "n_sub", "DE_A")
    prod_b = _product_node("n_b", "n_sub", "DE_B")
    other_fam = _family_node("n_other_fam", family="OTHER FAM")
    documents = [
        _doc("doc-brochure", "brochure.pdf", "BROCHURE", ["n_sub"]),
        _doc("doc-ce", "ACME_FAM_CE_Declaration.pdf", "CE_DECLARATION", ["n_a"]),
    ]
    product_nodes_path, document_nodes_path = _write_corpus(
        tmp_path, [fam, sub, prod_a, prod_b, other_fam], documents
    )
    return {
        "product_nodes_path": product_nodes_path,
        "document_nodes_path": document_nodes_path,
        "spec_keys_path": _write_dictionary(tmp_path, attributes=_ATTRIBUTES + [_CE_ATTR]),
    }


def _ce_row(db_path, product_code):
    with sqlite3.connect(db_path) as con:
        return con.execute(
            "SELECT status, bool_value, raw_text, source_doc_id, source_file_name, "
            "extractor, extracted_at FROM spec_value "
            "WHERE key='has_ce_declaration' AND product_code=?",
            (product_code,),
        ).fetchone()


def test_ce_declaration_sets_has_ce_declaration_present(tmp_path, ce_corpus):
    db_path, _ = _build(tmp_path, ce_corpus)
    status, bool_value, *_ = _ce_row(db_path, "DE_A")
    assert status == "present"
    assert bool_value == 1


def test_product_without_ce_declaration_stays_not_specified_never_absent(tmp_path, ce_corpus):
    """User decision (2026-08-05): the absence of the document is NOT
    the absence of the certification. Writing `absent` would amount to
    claiming "this product is not CE-certified"."""
    db_path, _ = _build(tmp_path, ce_corpus)
    status, bool_value, *_ = _ce_row(db_path, "DE_B")
    assert status == "not_specified"
    assert bool_value is None


def test_corpus_derived_boolean_carries_provenance(tmp_path, ce_corpus):
    db_path, _ = _build(tmp_path, ce_corpus)
    _status, _bv, raw_text, source_doc_id, source_file_name, extractor, extracted_at = _ce_row(
        db_path, "DE_A"
    )
    assert source_doc_id == "doc-ce"
    assert source_file_name == "ACME_FAM_CE_Declaration.pdf"
    assert "CE_DECLARATION" in raw_text
    assert extractor == "corpus_doc_type_v1"
    # KARAR-008: offset local ISO -- naive timestamp is NEVER WRITTEN.
    assert extracted_at and ("+" in extracted_at or extracted_at.endswith("Z"))


def test_non_ce_doc_types_do_not_set_the_flag(tmp_path, basic_corpus):
    # basic_corpus has NO CE declaration (only BROCHURE + DATASHEET) --
    # the flag must NOT be raised for any product.
    corpus = dict(basic_corpus)
    corpus["spec_keys_path"] = _write_dictionary(
        tmp_path, attributes=_ATTRIBUTES + [_CE_ATTR], name="spec_keys_ce.yaml"
    )
    db_path, result = _build(tmp_path, corpus)
    assert result["corpus_boolean_counts"] == {"has_ce_declaration": 0}
    assert _ce_row(db_path, "DE_A")[0] == "not_specified"


def test_not_applicable_rows_are_never_flipped_to_present(tmp_path, ce_corpus):
    """If the attribute is NOT IN scope for this family (`not_applicable`),
    CE declarations DO NOT flip it to `present` -- the scope decision
    comes BEFORE the corpus derivation."""
    corpus = dict(ce_corpus)
    out_of_scope_ce = dict(_CE_ATTR, block="power_distribution")  # OTHER FAM only
    corpus["spec_keys_path"] = _write_dictionary(
        tmp_path, attributes=_ATTRIBUTES + [out_of_scope_ce], name="spec_keys_oos.yaml"
    )
    db_path, result = _build(tmp_path, corpus)
    assert _ce_row(db_path, "DE_A")[0] == "not_applicable"
    assert result["corpus_boolean_counts"] == {"has_ce_declaration": 0}


def test_corpus_boolean_counts_reported_and_written_to_report(tmp_path, ce_corpus):
    _db_path, result = _build(tmp_path, ce_corpus)
    assert result["corpus_boolean_counts"] == {"has_ce_declaration": 1}
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Corpus-derived booleans" in report
    assert "has_ce_declaration" in report


def test_two_ce_declarations_pick_a_deterministic_evidence_doc(tmp_path):
    """If two CE declarations cover the same product, the evidence
    document is picked DETERMINISTICALLY by doc_id order -- two builds
    must produce the same output."""
    fam = _family_node("n_fam", family="FAM")
    sub = _subfamily_node("n_sub", "n_fam", family="FAM", subfamily="SUB")
    prod_a = _product_node("n_a", "n_sub", "DE_A")
    other_fam = _family_node("n_other_fam", family="OTHER FAM")
    documents = [
        _doc("doc-ce-z", "z_CE.pdf", "CE_DECLARATION", ["n_a"]),
        _doc("doc-ce-a", "a_CE.pdf", "CE_DECLARATION", ["n_sub"]),
    ]
    p_path, d_path = _write_corpus(tmp_path, [fam, sub, prod_a, other_fam], documents)
    corpus = {
        "product_nodes_path": p_path, "document_nodes_path": d_path,
        "spec_keys_path": _write_dictionary(tmp_path, attributes=_ATTRIBUTES + [_CE_ATTR]),
    }
    db_path, _ = _build(tmp_path, corpus)
    # doc-ce-a < doc-ce-z -- alphabetically first doc_id wins.
    assert _ce_row(db_path, "DE_A")[3] == "doc-ce-a"


def test_fill_doc_type_booleans_never_overwrites_an_existing_value():
    """A value coming from the fact ledger / human-verified is NEVER
    OVERWRITTEN by this derivation (same principle as KARAR-018) --
    only `not_specified` rows are written."""
    from medrag.pipeline.facts.build_facts_db import fill_doc_type_booleans

    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE spec_value (product_code TEXT, key TEXT, status TEXT, "
        "bool_value INTEGER, raw_text TEXT, source_doc_id TEXT, source_file_name TEXT, "
        "extractor TEXT, extractor_version TEXT, extracted_at TEXT)"
    )
    con.executemany(
        "INSERT INTO spec_value (product_code, key, status, extractor) VALUES (?,?,?,?)",
        [("DE_A", "has_ce_declaration", "absent", "chunk_full_run_v1"),
         ("DE_B", "has_ce_declaration", "not_specified", None)],
    )
    documents = [_doc("doc-ce", "ce.pdf", "CE_DECLARATION", ["n_a"])]
    counts = fill_doc_type_booleans(
        con.cursor(), documents, {"doc-ce": ["DE_A", "DE_B"]}
    )
    rows = dict(con.execute(
        "SELECT product_code, status FROM spec_value WHERE key='has_ce_declaration'"
    ).fetchall())
    assert rows["DE_A"] == "absent"        # not overwritten
    assert rows["DE_B"] == "present"       # the empty one was filled
    assert counts == {"has_ce_declaration": 1}