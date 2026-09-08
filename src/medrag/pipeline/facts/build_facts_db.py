"""Builds the `db/specs.db` skeleton from the node corpus + spec dictionary.

**PHASE A / "dry DB" (PROTOCOL KARAR-038).** This script ONLY builds the
skeleton: fixed/pre-prepared reference data (product, document, attribute
dictionary) + the FULL (product x attribute) matrix. **NOT a single spec
VALUE is written** -- every `spec_value` row's value columns are born NULL.
Filling the values is PHASE C's job
(`facts/experiments/chunk_full_run/load_to_db.py`) and is out of scope for
this file.

**2026-08-04 -- LLM-SQL simplification (user decision, see schema_rag.sql
header comment).** The previous version (2026-08-03, KARAR-045..053)
materialized the taxonomy tree (`product_node`, 297 nodes, recursive) +
6-part attribute dictionary + 3-part fact layer (14 tables) into the DB
as-is -- text-to-SQL models (especially small/local ones) write multi-JOIN
chains unreliably, so it was reduced to 4 flat tables (`product`/
`document`/`attribute`/`spec_value`). The taxonomy tree and document
fan-out (KARAR-014) no longer LIVE in the DB -- this script itself walks
the tree ONCE (in Python) and writes the result
(`document.product_codes` JSON array) flattened; no recursive query runs
at runtime (LLM SQL).

PROTOCOL KARAR-016: `specs.db` is a DERIVED product. This script is its
SOLE writer -- a manual `INSERT` is a contract violation. On every run:

  1. The current specs.db is backed up as `specs.db.legacy_<ts>`
     (KARAR-008 folder-safe timestamp) -- read-only comparison reference,
     NEVER a fact source (KARAR-016).
  2. `schema_rag.sql` is built from scratch and populated from:
       - `product` <- chatbot-corpus/product_info/product_nodes.json
                       (only leaf products; family/subfamily/subfamily_2
                       are copied as PLAIN TEXT, the taxonomy tree is NOT
                       WRITTEN)
       - `document` <- chatbot-corpus/document_info/document_nodes.json;
                       `product_codes` (the ONE-SHOT Python solution to
                       fan-out) is embedded in every row
       - `attribute` <- spec_schema/spec_keys.yaml (facts.spec_keys), a
                       SINGLE table with JSON-embedded labels/conditions/
                       subfields/applies_to
       - `spec_value` <- CARTESIAN PRODUCT of (leaf product x attribute),
                       tagged with the structural rule below. Fact flow,
                       LLM, and human approval do NOT enter here.
  3. A comparison report is written
     (`legacy_comparison_<ts>.md`) -- informational only; KARAR-016
     forbids using it to back-fill facts.

**Matrix rule.** For each (leaf product, attribute) pair: if the
attribute's block covers the product's family/subfamily, the row is born
`not_specified`; otherwise `not_applicable`. The rule lives in a single
place, `facts.spec_keys.key_applies`.

Run: `uv run python -m medrag.pipeline.facts.build_facts_db`
Lives in `src/medrag/pipeline/facts/`. Reads chatbot-corpus/ (override via
CHATBOT_CORPUS_DIR); writes `src/medrag/pipeline/facts/db/specs.db`
(+ legacy backup + comparison report).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from medrag.core.paths import resolve_specs_db_path
from medrag.pipeline.facts.facts_config import FactsConfig, load_config
from medrag.pipeline.facts.spec_keys import (
    SPEC_KEYS_PATH,
    attribute_rows_flat,
    block_scopes,
    key_applies,
    load_spec_keys,
    ordered_attributes,
    validate_dictionary,
    validate_products,
)

_FACTS_ROOT = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/ (code)
_REPO_ROOT = Path(__file__).resolve().parents[4]  # medrag/ (facts/pipeline/urun/src/<repo>)
# DATA did not move with the code (O-03 scope: code only) -- specs.db /
# schema_rag.sql still live in `facts/db/` at the repo root (same principle
# as discover.py's `_FACTS_DATA_ROOT`).
_FACTS_DATA_ROOT = _REPO_ROOT / "facts"
_DB_DIR = _FACTS_DATA_ROOT / "db"
DB_PATH = resolve_specs_db_path()
SCHEMA_PATH = _DB_DIR / "schema_rag.sql"
CORPUS_ROOT = Path(os.environ.get("CHATBOT_CORPUS_DIR", _REPO_ROOT / "chatbot-corpus"))
PRODUCT_NODES_PATH = CORPUS_ROOT / "product_info" / "product_nodes.json"
DOCUMENT_NODES_PATH = CORPUS_ROOT / "document_info" / "document_nodes.json"

# KARAR-025: a product that has no source (extractable document). In PHASE
# A only used to produce `info_status` -- the matrix is still produced IN
# FULL for these products (a product having no document does NOT mean
# the key is NOT APPLICABLE to it; it just means "nothing to look at yet",
# which is the definition of `not_specified`).
NO_SOURCE_INFO_STATUSES = frozenset({"no_documents", "no_extractable_documents"})

_STATUSES = ("present", "not_specified", "absent", "not_applicable", "conflicting", "superseded")


def compute_info_status(doc_types: set[str], *, config: FactsConfig) -> str:
    """KARAR-025's `info_status` derivation, in a single place. `facts.absent_scan`
    also imports this (while that pass remains in service)."""
    if not doc_types:
        return "no_documents"
    if any(config.in_scope_doc_type(dt) for dt in doc_types):
        return "extracted"
    return "no_extractable_documents"


def _now_stamp() -> str:
    """KARAR-008 folder-safe display timestamp (never parsed back)."""
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def _now_iso() -> str:
    """KARAR-008 offset local ISO-8601 (same helper as extract/link/differ)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# CORPUS-DERIVABLE boolean attributes: the existence of a doc_type for a
# product ALONE proves them -- no LLM extraction is needed at all, so this
# is the job of the deterministic builder, not the fact ledger
# (KARAR-016).
#
# Why here: `has_ce_declaration` was `not_specified` for all 249 of 249
# products (root cause of 2026-08-05 live test finding #4) -- every SQL
# about CE necessarily returned empty, and the answering model filled
# the gap from CE declaration chunks of OTHER products (attributing
# another product's CE declaration to the product asked, finding #3).
# The data was NOT missing: `document.product_codes` fan-out already
# carried the 143 product codes covered by CE declaration documents;
# nobody was looking.
#
# The `doc_type -> key` mapping is a DERIVATION RULE, not tuning -- it
# therefore lives here, not in config (root CONFIG.md taxonomy). If a new
# corpus-derivable boolean appears, ONE LINE is added here; the flow is
# unchanged.
_DOC_TYPE_BOOLEANS: dict[str, str] = {"has_ce_declaration": "CE_DECLARATION"}

_CORPUS_EXTRACTOR = "corpus_doc_type_v1"
_CORPUS_EXTRACTOR_VERSION = "build_facts_db:1"


def fill_doc_type_booleans(
    cur, documents: list[dict], doc_product_codes: dict[str, list[str]]
) -> dict[str, int]:
    """Populates `_DOC_TYPE_BOOLEANS` from the `document.product_codes`
    fan-out.

    Writes ONLY to rows with `status='not_specified'`. Two things are
    deliberately PRESERVED: (a) `not_applicable` (the attribute does not
    apply to this family at all, so CE is not flipped to "present"),
    (b) any previously written value (`present`/`absent`/`conflicting`)
    -- a value coming from the fact ledger or human-verified is NEVER
    overwritten by this derivation (same principle as KARAR-018).

    An uncovered product stays `not_specified`; `absent` is NOT written
    (user decision, 2026-08-05): the absence of the document is not
    the absence of the certification. Writing `absent` would amount to
    claiming "this product is not CE-certified", which is the
    mirror-image of the hallucination just corrected.

    Returns: key -> count of rows actually updated."""
    now = _now_iso()
    counts: dict[str, int] = {}
    for key, doc_type in sorted(_DOC_TYPE_BOOLEANS.items()):
        # code -> first (deterministic: by doc_id order) evidence document.
        owners: dict[str, tuple[str, str]] = {}
        for doc in sorted(documents, key=lambda d: d["identity"]["doc_id"]):
            if doc["doc_type"] != doc_type:
                continue
            doc_id = doc["identity"]["doc_id"]
            file_name = doc["identity"]["file_name"]
            for code in doc_product_codes.get(doc_id, []):
                owners.setdefault(code, (doc_id, file_name))

        params = [
            (f"{doc_type}: {file_name}", doc_id, file_name,
             _CORPUS_EXTRACTOR, _CORPUS_EXTRACTOR_VERSION, now, key, code)
            for code, (doc_id, file_name) in sorted(owners.items())
        ]
        cur.executemany(
            "UPDATE spec_value SET status='present', bool_value=1, raw_text=?, "
            "source_doc_id=?, source_file_name=?, extractor=?, extractor_version=?, "
            "extracted_at=? WHERE key=? AND product_code=? AND status='not_specified'",
            params,
        )
        counts[key] = cur.execute(
            "SELECT COUNT(*) FROM spec_value WHERE key=? AND extractor=?",
            (key, _CORPUS_EXTRACTOR),
        ).fetchone()[0]
    return counts


# ---------------------------------------------------------------------------
# Node corpus
# ---------------------------------------------------------------------------


def load_product_nodes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["product_nodes"]


def load_documents(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documents"]


def corpus_scopes(product_nodes: list[dict]) -> tuple[set[str], set[str]]:
    """The actual family / `family / subfamily` sets against which the
    dictionary's scope names are validated -- same as its counterpart in
    `chatbot-corpus/spec_schema/scripts/build_spec_keys.py` (KARAR-046)."""
    families = {n["family"] for n in product_nodes}
    subfamilies = {f"{n['family']} / {n['subfamily']}" for n in product_nodes if n.get("subfamily")}
    return families, subfamilies


def resolve_model(product_node: dict) -> str | None:
    """Human-readable product id (`product_code`, or `variant_base` as
    fallback) -- the alias used by the ledger/fact layer
    (`reduce.py`/`absent_scan.py`/`catalog_verify.py`) for the
    `Fact.model` string field. `None` signals unresolvable identities,
    like the KARAR-013 product without a product_code -- callers skip
    and report these, they do not invent."""
    p = product_node["product"] or {}
    return p.get("product_code") or p.get("variant_base")


def compute_doc_product_codes(documents: list[dict], product_nodes: list[dict]) -> dict[str, list[str]]:
    """`doc_id -> sorted unique product_code list` -- the BUILD-TIME,
    ONE-SHOT resolution of KARAR-014 fan-out (2026-08-04 simplification).
    Previously this was the `document_owner` M:N table + the
    `v_document_product` recursive CTE; now it is computed once in
    Python here and embedded as a JSON column `document.product_codes`
    -- no tree traversal happens at runtime (LLM SQL)."""
    by_id = {n["node_id"]: n for n in product_nodes}
    children: dict[str, list[str]] = {}
    for n in product_nodes:
        parent_id = n.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(n["node_id"])

    memo: dict[str, frozenset[str]] = {}

    def leaf_codes_under(node_id: str) -> frozenset[str]:
        if node_id in memo:
            return memo[node_id]
        node = by_id.get(node_id)
        if node is None:
            memo[node_id] = frozenset()
            return memo[node_id]
        if node["type"] == "product":
            code = resolve_model(node)
            result = frozenset({code}) if code else frozenset()
        else:
            result = frozenset().union(*(leaf_codes_under(c) for c in children.get(node_id, [])))
        memo[node_id] = result
        return result

    out: dict[str, list[str]] = {}
    for doc in documents:
        doc_id = doc["identity"]["doc_id"]
        owner_ids = doc["links"]["owner_ids"]
        codes: set[str] = set()
        for owner_id in owner_ids:
            codes |= leaf_codes_under(owner_id)
        out[doc_id] = sorted(codes)
    return out


# ---------------------------------------------------------------------------
# Legacy backup + comparison
# ---------------------------------------------------------------------------


def backup_legacy(*, db_path: Path, stamp: str) -> Path | None:
    """KARAR-016: move (not copy) the current specs.db out of the way so a
    fresh one can be built at the canonical path; the moved file IS the
    'specs.db.legacy_<ts>' reference. Returns None if there was nothing to
    back up (first run)."""
    if not db_path.is_file():
        return None
    legacy_path = db_path.with_name(f"specs.db.legacy_{stamp}")
    shutil.move(str(db_path), str(legacy_path))
    return legacy_path


def _legacy_stats(legacy_path: Path) -> dict:
    con = sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n_products = cur.execute("SELECT COUNT(*) FROM product").fetchone()[0] if "product" in tables else None
        n_documents = cur.execute("SELECT COUNT(*) FROM document").fetchone()[0] if "document" in tables else None

        value_table = "spec_value" if "spec_value" in tables else (
            "attribute_value" if "attribute_value" in tables else None
        )
        if value_table:
            status_counts = dict(
                cur.execute(f"SELECT status, COUNT(*) FROM {value_table} GROUP BY status").fetchall()
            )
            key_table = "attribute" if "attribute" in tables else None
            if key_table == "attribute":
                n_keys = cur.execute("SELECT COUNT(*) FROM attribute").fetchone()[0]
            else:
                n_keys = None
        else:
            status_counts, n_keys = {}, None

        return {
            "n_products": n_products, "n_documents": n_documents, "n_keys": n_keys,
            "status_counts": status_counts, "value_table": value_table,
        }
    finally:
        con.close()


def write_comparison_report(*, legacy_path: Path | None, new_stats: dict, stamp: str, report_dir: Path) -> Path:
    report_path = report_dir / f"legacy_comparison_{stamp}.md"
    lines = [
        f"# specs.db rebuild comparison -- {stamp}",
        "",
        "KARAR-016: the legacy DB below is a READ-ONLY reference. No fact was",
        "copied from it into the new build. This build is PHASE A (the dry",
        "skeleton): it writes no EXTRACTED spec value, so a near-zero `present`",
        "count is the expected, correct outcome -- not a loss.",
        "",
        "EXCEPTION (PROTOCOL KARAR-063, 2026-08-05): corpus-derivable booleans",
        "(today only `has_ce_declaration`) ARE written here, because a doc_type's",
        "existence proves them on its own -- no LLM, no fact ledger. So a small",
        "non-zero `present` count is expected; see `corpus_boolean_counts` below.",
        "",
    ]
    if legacy_path is None:
        lines.append("No legacy specs.db found -- this is the first build.")
    else:
        legacy = _legacy_stats(legacy_path)
        lines += [
            f"Legacy DB: `{legacy_path.name}` (value table: `{legacy['value_table']}`)",
            "",
            "| metric | legacy | new |",
            "|---|---|---|",
            f"| products | {legacy['n_products']} | {new_stats['n_products']} |",
            f"| document rows | {legacy['n_documents']} | {new_stats['n_documents']} |",
            f"| dictionary keys | {legacy['n_keys']} | {new_stats['n_attributes']} |",
        ]
        lines += [
            f"| value {status} | {legacy['status_counts'].get(status, 0)} | "
            f"{new_stats['status_counts'].get(status, 0)} |"
            for status in _STATUSES
        ]
    # KARAR-063: booleans filled from the corpus -- the report MUST explain
    # why `present` is not zero.
    corpus_counts = new_stats.get("corpus_boolean_counts") or {}
    if corpus_counts:
        lines += ["", "## Corpus-derived booleans (KARAR-063)", ""]
        lines += [f"- `{key}`: {count} product(s) set to `present`"
                  for key, count in sorted(corpus_counts.items())]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Build core (parameterized for tests; `main()` calls this with real paths)
# ---------------------------------------------------------------------------


def build(
    *,
    db_path: Path = DB_PATH,
    schema_path: Path = SCHEMA_PATH,
    product_nodes_path: Path = PRODUCT_NODES_PATH,
    document_nodes_path: Path = DOCUMENT_NODES_PATH,
    spec_keys_path: Path = SPEC_KEYS_PATH,
    config: FactsConfig | None = None,
) -> dict:
    """Runs one full skeleton build; returns a summary dict (also used by tests)."""
    config = config or load_config()

    product_nodes = load_product_nodes(product_nodes_path)
    families, subfamilies = corpus_scopes(product_nodes)

    # Dictionary and taxonomy are validated FIRST: better to fail loudly
    # without writing a single row than to leave a half-written DB
    # (root CLAUDE.md fail-loud).
    dictionary = load_spec_keys(spec_keys_path)
    validate_dictionary(dictionary, families, subfamilies)
    validate_products(product_nodes, families)

    stamp = _now_stamp()
    legacy_path = backup_legacy(db_path=db_path, stamp=stamp)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()
    cur.executescript(schema_path.read_text(encoding="utf-8"))

    leaf_nodes = [n for n in product_nodes if n["type"] == "product"]

    # --- product rows (only leaf products AND only those WITH a product_code;
    # the single KARAR-013 node without a product_code CANNOT BE REPRESENTED
    # in this schema -- PK is now product_code itself, no node_id. Known,
    # accepted loss: 250 -> 249, NOT a fact loss since that node has no
    # facts to begin with.
    # ---------------------------------------------------------------------
    product_rows: list[tuple] = []
    code_by_node_id: dict[str, str] = {}
    orphan_leaf_nodes: list[str] = []
    for n in leaf_nodes:
        p = n.get("product")
        if p is None:
            raise ValueError(f"leaf product node {n['node_id']!r} has no 'product' payload")
        code = resolve_model(n)
        if code is None:
            orphan_leaf_nodes.append(n["node_id"])
            continue
        code_by_node_id[n["node_id"]] = code
        product_rows.append(
            (
                code, p.get("display_name"), n["family"], n.get("subfamily"), n.get("subfamily_2"),
                p.get("acme_code"), p.get("list_price"), 1 if p.get("price_on_request") else 0,
                p.get("valid_until"), p.get("price_break_10_99"), p.get("price_break_100_499"),
                p.get("price_break_500_999"), 1 if p.get("is_variant") else 0, p.get("variant_base"),
            )
        )
    cur.executemany(
        "INSERT INTO product (product_code, display_name, family, subfamily, subfamily_2, acme_code, "
        "list_price, price_on_request, valid_until, price_break_10_99, price_break_100_499, "
        "price_break_500_999, is_variant, variant_base) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        product_rows,
    )

    documents = load_documents(document_nodes_path)
    doc_product_codes = compute_doc_product_codes(documents, product_nodes)

    # --- document rows (KARAR-019: one physical file = one row) ---
    doc_rows: list[tuple] = []
    docs_with_no_owner: list[str] = []
    for doc in documents:
        doc_id = doc["identity"]["doc_id"]
        codes = doc_product_codes.get(doc_id, [])
        if not codes:
            docs_with_no_owner.append(f"{doc_id}  {doc['identity']['file_name']}")
        doc_rows.append(
            (
                doc_id, doc["identity"]["file_name"], doc["identity"]["extension"],
                doc["location"]["rel_path"], doc["location"]["file_path"], doc["scan"]["content_hash"],
                doc["scan"].get("size_bytes"), doc["scan"].get("last_modified_time"), doc["doc_type"],
                1 if doc.get("is_active", True) else 0, json.dumps(codes, ensure_ascii=False),
            )
        )
    cur.executemany(
        "INSERT INTO document (doc_id, file_name, extension, rel_path, file_path, content_hash, "
        "size_bytes, last_modified_time, doc_type, is_active, product_codes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        doc_rows,
    )

    # Each leaf product's seen doc_type set -- for KARAR-025's "extracted"
    # definition (now directly from `document.product_codes`, no extra
    # table/view needed).
    doc_types_by_code: dict[str, set[str]] = {code: set() for code in code_by_node_id.values()}
    for doc in documents:
        for code in doc_product_codes.get(doc["identity"]["doc_id"], []):
            if code in doc_types_by_code:
                doc_types_by_code[code].add(doc["doc_type"])

    # --- dictionary layer: ONE `attribute` table (2026-08-04) ---
    cur.executemany(
        "INSERT INTO attribute (block, key, kind, unit, enum_values, accepts_units, labels, conditions, "
        "subfields, applies_to, is_core, is_active) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        attribute_rows_flat(dictionary),
    )

    # --- spec_value: the FULL (leaf product x attribute) matrix, NO VALUES ---
    scopes = block_scopes(dictionary)
    attrs = ordered_attributes(dictionary)

    matrix_rows: list[tuple] = []
    n_not_specified = 0
    for n in leaf_nodes:
        code = code_by_node_id.get(n["node_id"])
        if code is None:
            continue
        family, subfamily = n["family"], n.get("subfamily")
        for a in attrs:
            applicable = key_applies(family, subfamily, scopes[a["block"]])
            n_not_specified += applicable
            matrix_rows.append((
                code, family, subfamily, a["block"], a["key"], a["kind"], a.get("unit"),
                "not_specified" if applicable else "not_applicable",
            ))

    cur.executemany(
        "INSERT INTO spec_value (product_code, family, subfamily, block, key, kind, unit, status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        matrix_rows,
    )

    # --- corpus-derivable booleans (PROTOCOL KARAR-063) ---
    # RIGHT AFTER the matrix is born VALUELESS: attributes that follow
    # deterministically from a doc_type's existence (today only
    # `has_ce_declaration`) are populated here -- they do NOT go to the
    # LLM and do NOT need the fact ledger. This closes the root cause of
    # live test findings #3/#4 (see `_DOC_TYPE_BOOLEANS` comment).
    corpus_boolean_counts = fill_doc_type_booleans(cur, documents, doc_product_codes)

    # --- info_status (KARAR-025, NOT a DB column -- only this build's summary) ---
    info_counts: dict[str, int] = {}
    for code in code_by_node_id.values():
        status = compute_info_status(doc_types_by_code.get(code, set()), config=config)
        info_counts[status] = info_counts.get(status, 0) + 1

    con.commit()

    new_stats = {
        "n_products": len(product_rows),
        "n_documents": len(doc_rows),
        "n_attributes": len(attrs),
        "status_counts": dict(cur.execute("SELECT status, COUNT(*) FROM spec_value GROUP BY status").fetchall()),
        # KARAR-063: booleans filled deterministically from the corpus --
        # never silently; they appear in the build summary.
        "corpus_boolean_counts": corpus_boolean_counts,
    }
    con.close()

    report_path = write_comparison_report(
        legacy_path=legacy_path, new_stats=new_stats, stamp=stamp, report_dir=db_path.parent
    )

    return {
        "stamp": stamp,
        "legacy_path": legacy_path,
        "report_path": report_path,
        "n_products": len(product_rows),
        "n_orphan_leaf_nodes": len(orphan_leaf_nodes),
        "orphan_leaf_nodes": orphan_leaf_nodes,
        "n_documents": len(doc_rows),
        "n_attributes": len(attrs),
        "n_matrix_rows": len(matrix_rows),
        "n_not_specified": n_not_specified,
        "n_not_applicable": len(matrix_rows) - n_not_specified,
        "info_counts": info_counts,
        "docs_with_no_owner": docs_with_no_owner,
        # KARAR-063 -- key -> count of rows set to `present` from the corpus.
        "corpus_boolean_counts": corpus_boolean_counts,
    }


def main() -> None:
    result = build()
    print(f"products added: {result['n_products']} (leaf nodes without product_code, skipped: "
          f"{result['n_orphan_leaf_nodes']})")
    for nid in result["orphan_leaf_nodes"]:
        print(f"  SKIPPED (no product_code): {nid}")
    print(f"document rows inserted: {result['n_documents']} "
          f"(with zero valid owners: {len(result['docs_with_no_owner'])})")
    for line in result["docs_with_no_owner"]:
        print(f"  SKIPPED (no owner): {line}")
    print(f"info_status: {result['info_counts']}")
    # KARAR-063 -- never silently: how many products have corpus-derived
    # `present` rows visible in the terminal on every run.
    print(f"corpus-derived booleans: {result['corpus_boolean_counts']}")
    print(f"dictionary: {result['n_attributes']} attributes")
    print(
        f"spec_value matrix: {result['n_matrix_rows']} rows "
        f"({result['n_not_specified']} not_specified, {result['n_not_applicable']} not_applicable, "
        "0 present -- PHASE A writes no values)"
    )
    if result["legacy_path"] is not None:
        print(f"legacy DB backed up to: {result['legacy_path']}")
    print(f"comparison report: {result['report_path']}")


if __name__ == "__main__":
    main()