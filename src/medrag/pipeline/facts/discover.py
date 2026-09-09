"""Builds the product -> (document, chunk) manifest for the WHOLE corpus --
the generalisation of atom_vs_chunk_bench's 4-product `build_dataset.py`.
This file does NOT write to facts.db (only opens `db/specs.db` read-only);
its sole job is to build the input to be fed to the LLM in `run_full.py`.

For product/document IDENTITY the sole source is specs.db itself:
`build_facts_db.py` (PHASE A) already populated `product`/`document` from
`product_nodes.json`/`document_nodes.json` and `v_document_product` view --
this script QUERIES that recursive fan-out instead of walking the tree a
SECOND time in Python (same pattern as build_facts_db.py's own
`doc_types_by_node`).

For chunk TEXT, two sources, in priority order:
  1. `all_chunks.json` (CHUNKER_OUTPUT_DIR/all_chunks.json, chunker's REAL
     output) -- NOT YET AVAILABLE on this machine (only chunker/storage/
     README.md exists, measurement of 2026-08-03). When present this is
     the SINGLE correct source: leaf chunks (tree_level==0),
     content_sha256/text/page_start/page_end/heading_path.
  2. Bridge (TEMPORARY): the partial atoms.ndjson (the `--atoms-bridge`,
     default `C:\\path\\to\\data\\atoms.ndjson`) + `facts/snapshots/*.txt`
     used by `facts/experiments/atom_vs_chunk_bench/build_dataset.py`.
     ONLY the doc_id -> chunk_sha256 map + facts/snapshots storage from
     that run are BORROWED (atom STRATEGY is not used -- user decision:
     never again); the YAN PRODUCT (doc_id -> chunk_sha256 map +
     facts/snapshots storage) of that run is the only thing borrowed.
     Until replaced by chunker's real output, this has LIMITED SCOPE --
     see README "Scope (critical)" section.

If a document cannot be found in ANY source (no text), it is NOT silently
skipped -- it is written to `ProductManifest.missing_docs`, and
`run_full.py` reports it.

Multi-owner document fan-out (2026-08-04, user decision -- REPLACES the
old `max_doc_owners=1` total-exclusion filter): for a document with
`document.product_codes` length >=2 (catalogue/brochure), the question
of WHICH chunk belongs to WHICH product(s) is now answered from
`results/chunk_ownership_all.json` produced by
`facts/experiments/catalog_chunk_ownership/build_all.py` (deterministic
for single-owner documents, per-chunk LLM labelling + confidence>=0.8
gate for multi-owner ones -- see that script's docstring). A chunk
that is not `accepted=true` for THIS product in that table does NOT
enter that product's manifest (silently no -- it is written to
`chunks_rejected_by_ownership`). When the table is absent (build_all.py
never ran), multi-owner documents fall back to the OLD behaviour:
EXCLUDED ENTIRELY (`docs_excluded_fanout`), because an uninformed
fan-out is worse than silent misattribution.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from medrag.core.paths import resolve_specs_db_path

HERE = Path(__file__).resolve().parent
FACTS_ROOT = HERE  # src/medrag/pipeline/facts/ (code)
REPO_ROOT = HERE.parents[3]  # medrag/ (facts/pipeline/medrag/src/<repo>)
# DATA (db/specs.db, snapshots/) was NOT moved with the code (O-03 scope
# was code only); the actual working data still lives in `facts/` at the
# repo root (facts/db/specs.db, facts/snapshots/*.txt). To avoid
# confusing `FACTS_ROOT` (code) with this, a different name.
_FACTS_DATA_ROOT = REPO_ROOT / "facts"
DB_PATH = resolve_specs_db_path()
SNAPSHOTS_DIR = _FACTS_DATA_ROOT / "snapshots"
CHUNKER_OUTPUT_ROOT = Path(os.environ.get("CHUNKER_OUTPUT_DIR", REPO_ROOT / "chunker" / "storage"))
ALL_CHUNKS_PATH = CHUNKER_OUTPUT_ROOT / "all_chunks.json"
DEFAULT_ATOMS_BRIDGE = Path(r"C:\path\to\data\atoms.ndjson")

from medrag.core.paths import resolve_chunk_ownership_dir
from medrag.pipeline.facts.facts_config import load_config


def _chunk_ownership_path() -> Path:
    """Call-time invocation of the SAME `resolve_chunk_ownership_dir()`
    resolver as `catalog_chunk_ownership/build_all.py::RESULTS_DIR`
    (the writer) -- NOT a frozen-at-import path, so it is reread on
    every `main()` call (required for tests' `monkeypatch.setenv(
    "CHUNK_OWNERSHIP_DIR", ...)`). The legacy name
    (`DEFAULT_CHUNK_OWNERSHIP_PATH`) is preserved below as a
    backward-compatibility module-level value; tests should use this
    function or `load_chunk_ownership_index(path=None)`, not the
    import-time value."""
    return resolve_chunk_ownership_dir() / "chunk_ownership_all.json"


DEFAULT_CHUNK_OWNERSHIP_PATH = _chunk_ownership_path()

# facts/config/default.toml [scope].exclude_doc_types uses the SAME values
# (KARAR-027).


@dataclass
class ChunkRow:
    chunk_sha256: str
    text: str | None
    doc_ids: list[str]
    page_start: int | None
    page_end: int | None
    heading_path: list[str] = field(default_factory=list)
    source: str = "unknown"  # 'all_chunks' | 'atoms_bridge'
    anchors_applied: list[str] = field(default_factory=list)
    # KARAR-064: the heading_anchor(s) by which this chunk's text was SLICED.
    # Empty = no slicing, the full chunk is sent (today's/default behaviour).


@dataclass
class ProductManifest:
    product_code: str
    family: str | None
    subfamily: str | None
    display_name: str | None
    doc_ids: list[str]                 # in-scope, is_active document ids (fan-out included)
    docs_missing_text: list[str]       # doc_ids: present in v_document_product but no chunk
                                        # text could be found in any source
    chunks: list[ChunkRow]             # unique chunks (deduped by chunk_sha256)
    docs_excluded_fanout: list[str] = field(default_factory=list)
    # Multi-owner documents for which the chunk-ownership table could
    # NOT BE FOUND -- table-bearing multi-owner documents are NOT here,
    # their rejected chunks go to `chunks_rejected_by_ownership`
    # (granular).
    chunks_rejected_by_ownership: list[dict] = field(default_factory=list)
    # Chunks of a multi-owner document that the ownership table marks as
    # NOT accepted=true for THIS product (low confidence or belonging to
    # another product) -- {"doc_id", "chunk_sha256", "reason"}. No
    # silent drops.
    anchors_not_found: list[dict] = field(default_factory=list)
    # KARAR-064: ownership table gave a `heading_anchor` that could NOT
    # BE FOUND in the chunk text -> no slicing, full chunk sent. No
    # silent skipping.


def _connect_ro(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(
            f"{db_path} does not exist -- first build the skeleton via "
            "`cd facts && python -m facts.build_facts_db` (PHASE A)."
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _load_all_chunks_leaf_index(path: Path = ALL_CHUNKS_PATH) -> dict[str, list[dict]]:
    """`{doc_id: [leaf chunk dict, ...]}` -- only tree_level==0 (applies
    the same `include_summary_nodes=false` default from KARAR-027;
    summary nodes CANNOT serve as evidence). Empty dict if the file is
    missing (caller falls back to the bridge)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents", {})
    out: dict[str, list[dict]] = {}
    for doc_id, chunk_set in documents.items():
        nodes = chunk_set.get("nodes", [])
        leaves = [n for n in nodes if n.get("tree_level", 0) == 0]
        if leaves:
            out[doc_id] = leaves
    return out


def _load_atoms_bridge_index(path: Path) -> dict[str, dict[str, dict]]:
    """`{doc_id: {chunk_sha256: {page_start, page_end}}}` -- built from
    the doc_id/chunk_sha256/page_* fields of atoms.ndjson (the
    partial pass-1 run on this machine); the atom's OWN `raw` text is
    NOT USED (that text is atomised, not chunk text) -- the real
    chunk text is read from facts/snapshots/<hex>.txt."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            doc_id, sha = a["doc_id"], a["chunk_sha256"]
            slot = out.setdefault(doc_id, {})
            row = slot.setdefault(sha, {"page_start": set(), "page_end": set()})
            if a.get("page_start") is not None:
                row["page_start"].add(a["page_start"])
            if a.get("page_end") is not None:
                row["page_end"].add(a["page_end"])
    return out


def _read_snapshot(chunk_sha256: str, *, root: Path = SNAPSHOTS_DIR) -> str | None:
    hexpart = chunk_sha256.split(":", 1)[-1]
    path = root / f"{hexpart}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def owner_count_by_doc(con: sqlite3.Connection) -> dict[str, int]:
    """`{doc_id: how many products it fans out to}` -- derived from
    `document.product_codes` (the JSON array computed at build time,
    KARAR-055). Whether a multi-owner document (>=2) goes through the
    chunk-ownership table or (when absent) is EXCLUDED ENTIRELY is
    decided by this count."""
    out: dict[str, int] = {}
    for doc_id, product_codes in con.execute("SELECT doc_id, product_codes FROM document"):
        try:
            out[doc_id] = len(json.loads(product_codes or "[]"))
        except (TypeError, json.JSONDecodeError):
            out[doc_id] = 0
    return out


_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def slice_by_anchor(text: str, anchor: str) -> str | None:
    """Slices the slice from the anchor heading line to the NEXT heading
    (exclusive).

    PROTOCOL KARAR-064 clause 1/4: the ownership table marks WHICH section
    of a chunk belongs to this product; the fact-producing model sees
    ONLY that section, making it STRUCTURALLY impossible to misattribute
    another product's numbers to it.

    A section includes its OWN sub-sections: the slice stops at a
    heading of the SAME or HIGHER (fewer `#`) level. This distinction is
    critical -- in the PN1253 catalogue the family-wide section
    `# 3. ENVIRONMENTAL SPECIFICATIONS` carries the temperature values
    under `## 3.1`/`## 3.2` sub-headings; stopping at any heading would
    leave the span empty, recreating the SAME loss that was just fixed.

    If the anchor is not found in the text, returns `None` -- the caller
    interprets it as "NO slicing performed, the full chunk will be sent"
    and WRITES it to the report (no silent skipping). `chunk_sha256`
    is UNCHANGED either way: the evidence envelope stays the full-chunk
    snapshot (KARAR-064, KARAR-015/018 preserved)."""
    start = text.find(anchor)
    if start < 0:
        return None
    level = len(anchor) - len(anchor.lstrip("#"))
    rest = text[start + len(anchor):]
    for match in _HEADING_RE.finditer(rest):
        head = match.group(0).strip()
        if len(head) - len(head.lstrip("#")) <= level:
            return (anchor + rest[: match.start()]).strip()
    return (anchor + rest).strip()


def load_chunk_anchor_index(
    path: Path | None = None,
) -> dict[tuple[str, str, str], list[str]]:
    """`{(doc_id, chunk_id, product_code): [heading_anchor, ...]}` --
    ONLY rows with `accepted=true` AND a non-empty `heading_anchor`.

    A product may have multiple sections in the same chunk (its own
    table + a family-wide environmental section), so the value is a
    LIST. Accepted rows without an anchor do NOT enter here -- they
    mean "whole chunk" (KARAR-064 clause 2).

    N-22: `path=None` (default) -- the path is resolved at call time via
    `_chunk_ownership_path()` (not a frozen-at-import default), so
    `CHUNK_OWNERSHIP_DIR` env var is effective in tests via
    `monkeypatch.setenv`."""
    path = path if path is not None else _chunk_ownership_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], list[str]] = {}
    for row in data.get("rows", []):
        anchor = row.get("heading_anchor")
        if not row.get("accepted") or not row.get("product_code") or not anchor:
            continue
        key = (row["doc_id"], row["chunk_id"], row["product_code"])
        if anchor not in out.setdefault(key, []):
            out[key].append(anchor)
    return out


def load_chunk_ownership_index(
    path: Path | None = None,
) -> dict[tuple[str, str], set[str]]:
    """`{(doc_id, chunk_id): {accepted product codes}}` -- ONLY
    `accepted=true` rows (confidence>=threshold, already applied in
    `build_all.py`). Returns empty dict if the file is missing -- caller
    reads this as "no table, exclude multi-owner documents entirely"
    (safe fallback to the old `max_doc_owners=1` behaviour).

    N-22: `path=None` (default) -- same call-time-resolve rationale as
    `load_chunk_anchor_index`."""
    path = path if path is not None else _chunk_ownership_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], set[str]] = {}
    for row in data.get("rows", []):
        if not row.get("accepted") or not row.get("product_code"):
            continue
        key = (row["doc_id"], row["chunk_id"])
        out.setdefault(key, set()).add(row["product_code"])
    return out


def build_product_manifest(
    con: sqlite3.Connection,
    product_code: str,
    *,
    all_chunks_index: dict[str, list[dict]],
    atoms_bridge_index: dict[str, dict[str, dict]],
    exclude_doc_types: frozenset[str],
    owner_counts: dict[str, int] | None = None,
    chunk_ownership_index: dict[tuple[str, str], set[str]] | None = None,
    chunk_anchor_index: dict[tuple[str, str, str], list[str]] | None = None,
) -> ProductManifest:
    cur = con.cursor()
    row = cur.execute(
        "SELECT product_code, family, subfamily, display_name FROM product "
        "WHERE UPPER(product_code) = UPPER(?)",
        (product_code,),
    ).fetchone()
    if row is None:
        raise ValueError(f"product_code={product_code!r} is not in specs.db (product table)")
    product_code, family, subfamily, display_name = row

    doc_rows = cur.execute(
        "SELECT DISTINCT doc_id, doc_type FROM v_document_product "
        "WHERE product_code = ? AND is_active = 1",
        (product_code,),
    ).fetchall()
    in_scope_docs = [(did, dt) for did, dt in doc_rows if dt not in exclude_doc_types]

    # Multi-owner document fan-out (2026-08-04, KARAR revision -- REPLACES
    # the old `max_doc_owners=1` total-exclusion with CHUNK-BASED
    # narrowing): if a document belongs to >=2 products, WHICH chunk of
    # it belongs to THIS product is queried from `chunk_ownership_index`.
    # If the index is MISSING (build_all.py never ran), safe fallback:
    # exclude entirely (old behaviour).
    counts = owner_counts if owner_counts is not None else owner_count_by_doc(con)
    ownership_index = chunk_ownership_index or {}
    docs_excluded_fanout: list[str] = []
    chunks_rejected_by_ownership: list[dict] = []
    anchors_not_found: list[dict] = []
    if not ownership_index:
        kept = []
        for did, dt in in_scope_docs:
            if counts.get(did, 1) >= 2:
                docs_excluded_fanout.append(did)
            else:
                kept.append((did, dt))
        in_scope_docs = kept

    chunks_by_sha: dict[str, ChunkRow] = {}
    docs_missing_text: list[str] = []

    for doc_id, _doc_type in in_scope_docs:
        is_multi_owner = counts.get(doc_id, 1) >= 2 and bool(ownership_index)
        leaves = all_chunks_index.get(doc_id)
        if leaves:
            for n in leaves:
                sha = n.get("content_sha256") or n["node_id"]
                if is_multi_owner:
                    accepted_codes = ownership_index.get((doc_id, n["node_id"]))
                    if not accepted_codes or product_code not in accepted_codes:
                        chunks_rejected_by_ownership.append({
                            "doc_id": doc_id, "chunk_sha256": sha,
                            "reason": "not accepted for this product_code (chunk_ownership_index)",
                        })
                        continue
                r = chunks_by_sha.get(sha)
                if r is None:
                    chunks_by_sha[sha] = ChunkRow(
                        chunk_sha256=sha, text=n.get("text"), doc_ids=[doc_id],
                        page_start=n.get("page_start"), page_end=n.get("page_end"),
                        heading_path=n.get("heading_path") or [], source="all_chunks",
                    )
                elif doc_id not in r.doc_ids:
                    r.doc_ids.append(doc_id)
            continue

        bridge = atoms_bridge_index.get(doc_id)
        if not bridge:
            docs_missing_text.append(doc_id)
            continue
        for sha, pages in bridge.items():
            if is_multi_owner:
                accepted_codes = ownership_index.get((doc_id, sha))
                if not accepted_codes or product_code not in accepted_codes:
                    chunks_rejected_by_ownership.append({
                        "doc_id": doc_id, "chunk_sha256": sha,
                        "reason": "not accepted for this product_code (chunk_ownership_index)",
                    })
                    continue
            r = chunks_by_sha.get(sha)
            if r is not None:
                if doc_id not in r.doc_ids:
                    r.doc_ids.append(doc_id)
                continue
            text = _read_snapshot(sha)
            ps = sorted(pages["page_start"]) if pages["page_start"] else []
            pe = sorted(pages["page_end"]) if pages["page_end"] else []
            chunks_by_sha[sha] = ChunkRow(
                chunk_sha256=sha, text=text, doc_ids=[doc_id],
                page_start=ps[0] if ps else None, page_end=pe[-1] if pe else None,
                source="atoms_bridge",
            )

    # KARAR-064: if the ownership table gives only a SPECIFIC section
    # of a chunk to this product, only that section(s) goes to the LLM.
    # `chunk_sha256` does NOT CHANGE -- evidence envelope stays the full
    # chunk snapshot.
    anchors = chunk_anchor_index or {}
    for key, row in chunks_by_sha.items():
        if row.text is None:
            continue
        wanted: list[str] = []
        for doc_id in row.doc_ids:
            wanted += anchors.get((doc_id, key, product_code), [])
        if not wanted:
            continue
        pieces, missing = [], []
        for anchor in wanted:
            piece = slice_by_anchor(row.text, anchor)
            (pieces if piece else missing).append(piece or anchor)
        if missing:
            anchors_not_found.append({
                "chunk_sha256": key, "product_code": product_code, "anchors": missing,
            })
            continue
        row.text = "\n\n".join(pieces)
        row.anchors_applied = list(wanted)

    chunks = sorted(chunks_by_sha.values(), key=lambda c: c.chunk_sha256)
    missing_snapshot = [c.chunk_sha256 for c in chunks if c.text is None]
    if missing_snapshot:
        chunks = [c for c in chunks if c.text is not None]

    return ProductManifest(
        product_code=product_code, family=family, subfamily=subfamily,
        display_name=display_name, doc_ids=[d for d, _ in in_scope_docs],
        docs_missing_text=sorted(set(docs_missing_text) | set()), chunks=chunks,
        docs_excluded_fanout=sorted(set(docs_excluded_fanout)),
        chunks_rejected_by_ownership=chunks_rejected_by_ownership,
        anchors_not_found=anchors_not_found,
    )


def all_product_codes(con: sqlite3.Connection) -> list[str]:
    cur = con.cursor()
    return [r[0] for r in cur.execute(
        "SELECT product_code FROM product WHERE product_code IS NOT NULL ORDER BY product_code"
    ).fetchall()]


def products_for_doc_id(con: sqlite3.Connection, doc_id: str) -> list[str]:
    """O-12: when a document changes, which products need to be RE-PROCESSED
    -- via `v_document_product` (same view `build_product_manifest` uses as
    its source, NO SECOND fan-out computation is opened). `is_active=1`
    rows (inactive product-document links) are EXCLUDED -- they already do
    not enter `build_product_manifest` either."""
    cur = con.cursor()
    return [r[0] for r in cur.execute(
        "SELECT DISTINCT product_code FROM v_document_product "
        "WHERE doc_id = ? AND is_active = 1 ORDER BY product_code",
        (doc_id,),
    ).fetchall()]


def load_indexes(atoms_bridge_path: Path = DEFAULT_ATOMS_BRIDGE):
    """`(all_chunks_index, atoms_bridge_index, exclude_doc_types,
    chunk_ownership_index, chunk_anchor_index)` -- the last two replaced
    `max_doc_owners` on 2026-08-04 (chunk-based narrowing, instead of
    `docs_excluded_fanout`'s old full-exclusion)."""
    config = load_config()
    exclude = frozenset(config.scope.exclude_doc_types)
    all_chunks_index = _load_all_chunks_leaf_index()
    atoms_bridge_index = _load_atoms_bridge_index(atoms_bridge_path)
    chunk_ownership_index = load_chunk_ownership_index()
    chunk_anchor_index = load_chunk_anchor_index()
    return all_chunks_index, atoms_bridge_index, exclude, chunk_ownership_index, chunk_anchor_index


if __name__ == "__main__":
    con = _connect_ro()
    all_chunks_index, atoms_bridge_index, exclude, chunk_ownership_index, chunk_anchor_index = load_indexes()
    owner_counts = owner_count_by_doc(con)
    n_multi_owner = sum(1 for v in owner_counts.values() if v >= 2)
    _ownership_path = _chunk_ownership_path()
    print(f"all_chunks.json found: {ALL_CHUNKS_PATH.is_file()} ({ALL_CHUNKS_PATH})")
    print(f"atoms bridge found: {DEFAULT_ATOMS_BRIDGE.is_file()} ({DEFAULT_ATOMS_BRIDGE})")
    print(f"chunk_ownership_index found: {_ownership_path.is_file()} "
          f"({_ownership_path}), {len(chunk_ownership_index)} (doc,chunk) accepted")
    print(f"multi-owner (>=2) documents: {n_multi_owner}/{len(owner_counts)} "
          f"({'chunk-based narrowing ACTIVE' if chunk_ownership_index else 'table MISSING -- excluded entirely'})")
    codes = all_product_codes(con)
    n_full = n_partial = n_none = 0
    for code in codes:
        m = build_product_manifest(
            con, code, all_chunks_index=all_chunks_index,
            atoms_bridge_index=atoms_bridge_index, exclude_doc_types=exclude,
            owner_counts=owner_counts, chunk_ownership_index=chunk_ownership_index,
            chunk_anchor_index=chunk_anchor_index,
        )
        if not m.doc_ids:
            continue
        if not m.docs_missing_text and m.chunks:
            n_full += 1
        elif m.chunks:
            n_partial += 1
        else:
            n_none += 1
    print(f"total products (with at least one extractable document): {n_full + n_partial + n_none}")
    print(f"  full coverage (chunk text for all documents): {n_full}")
    print(f"  partial coverage (some documents missing): {n_partial}")
    print(f"  no chunk text at all: {n_none}")