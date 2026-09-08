"""
The single writer for the lineage panel. READS the existing artifacts
(document_nodes.json, chunker/storage/all_chunks.json, facts/db/specs.db, root
storage/) and WRITES to none of them -- a read-only observation layer. Each run
APPENDS new rows to `lineage.db` (never overwrites), so the run-history
accumulates.

Run: `python ingest.py` (from this folder, or from the repo root
`python src/medrag/api/panel/ingest.py`). The DB lives outside src/, in the
`lineage_panel_data/` folder at the repo root.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from medrag.core.db.sqlite import BUSY_TIMEOUT_MS

# src/medrag/api/panel/ingest.py -> parents[4] is the repo root.
ROOT = Path(__file__).resolve().parents[4]
# lineage.db lives outside src/ -- in the `lineage_panel_data/` folder at the repo root.
LINEAGE_DB = ROOT / "lineage_panel_data" / "lineage.db"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

DOCUMENT_NODES_PATH = ROOT / "chatbot-corpus" / "document_info" / "document_nodes.json"
ALL_CHUNKS_PATH = ROOT / "chunker" / "storage" / "all_chunks.json"
SPECS_DB_PATH = ROOT / "facts" / "db" / "specs.db"
QDRANT_STORAGE_DIR = ROOT / "storage" / "collections"


def now_iso() -> str:
    # Offset-aware local ISO, second precision.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_document_nodes():
    with open(DOCUMENT_NODES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


def load_chunk_sets():
    if not ALL_CHUNKS_PATH.exists():
        return {}
    with open(ALL_CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("documents", {})


# Known-weak signal: the root storage/ directory it was originally derived from
# no longer exists, so this only reflects whatever mtime remains on disk.
def vectorize_last_touched():
    if not QDRANT_STORAGE_DIR.exists():
        return None
    latest = None
    for p in QDRANT_STORAGE_DIR.rglob("*"):
        if p.is_file():
            mtime = p.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=UTC).astimezone().isoformat(timespec="seconds")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))


def ingest_parse_and_chunk_stages(conn, docs, chunk_sets, ingested_at):
    rows_runs = []
    rows_edges = []

    for doc in docs:
        doc_id = doc["identity"]["doc_id"]
        content_hash = doc.get("scan", {}).get("content_hash")
        parse = doc.get("parse", {})
        chunk = doc.get("chunk", {})

        rows_runs.append((
            "parse", doc_id, parse.get("status"), parse.get("parsed_from_hash"),
            parse.get("last_parsed"), 0, None,
            f"parser_version={parse.get('parser_version')}", ingested_at,
        ))

        if chunk:
            chunked_from_hash = chunk.get("chunked_from_hash")
            has_run = bool(chunked_from_hash)
            is_stale = int(has_run and bool(content_hash) and chunked_from_hash != content_hash)
            stale_reason = (
                "chunked_from_hash, dokumanin guncel content_hash'iyle uyusmuyor "
                "(dokuman degisti ama yeniden chunklanmadi)"
                if is_stale else None
            )
            rows_runs.append((
                "chunk", doc_id, chunk.get("status"), chunked_from_hash,
                chunk.get("chunked_at"), is_stale, stale_reason,
                f"chunker_version={chunk.get('chunker_version')}", ingested_at,
            ))
            if has_run:
                rows_edges.append(("document", doc_id, "chunk_set", doc_id, ingested_at))

    return rows_runs, rows_edges


def ingest_facts_stage(conn, docs, ingested_at):
    """specs.db has no per-doc lineage record (the designed `extraction_run` table
    was removed) -- we derive it retroactively from spec_value.source_doc_id and
    extracted_at."""
    rows_runs = []
    rows_edges = []
    rows_conflicts = []

    chunk_by_doc = {}
    for doc in docs:
        doc_id = doc["identity"]["doc_id"]
        chunk_by_doc[doc_id] = doc.get("chunk", {}).get("chunked_at")

    if not SPECS_DB_PATH.exists():
        return rows_runs, rows_edges, rows_conflicts

    specs = sqlite3.connect(f"file:{SPECS_DB_PATH}?mode=ro", uri=True)
    # The WAL exemption on specs.db does not apply here -- a read-only open does
    # not touch journal_mode, only busy_timeout is set.
    specs.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    specs.row_factory = sqlite3.Row

    per_doc_latest = {}
    for row in specs.execute(
        "SELECT source_doc_id, MAX(extracted_at) AS latest, COUNT(*) AS n "
        "FROM spec_value WHERE source_doc_id IS NOT NULL GROUP BY source_doc_id"
    ):
        per_doc_latest[row["source_doc_id"]] = (row["latest"], row["n"])

    for doc_id, (latest, n) in per_doc_latest.items():
        chunked_at = chunk_by_doc.get(doc_id)
        is_stale = 0
        stale_reason = None
        latest_dt = _parse_dt(latest)
        chunked_dt = _parse_dt(chunked_at)
        if latest_dt and chunked_dt and latest_dt < chunked_dt:
            is_stale = 1
            stale_reason = "specs.db'deki en son extraction, dokumanin en son chunk'lanma zamanindan ONCE"
        rows_runs.append((
            "facts", doc_id, "PRESENT" if n else None, None,
            latest, is_stale, stale_reason, f"spec_value_rows={n}", ingested_at,
        ))
        rows_edges.append(("chunk_set", doc_id, "spec_value_group", doc_id, ingested_at))

    for row in specs.execute(
        "SELECT value_id, product_code, block, key, condition, raw_text, extracted_at, source_doc_id "
        "FROM spec_value WHERE status='conflicting'"
    ):
        rows_conflicts.append((
            "spec_value", str(row["value_id"]), row["product_code"], row["block"], row["key"],
            f"condition={row['condition']!r} raw_text={row['raw_text']!r} source_doc_id={row['source_doc_id']!r}",
            row["extracted_at"] or ingested_at, ingested_at,
        ))
        rows_edges.append(("spec_value_group", row["source_doc_id"], "conflict", str(row["value_id"]), ingested_at))

    for row in specs.execute("SELECT product_code, family, subfamily FROM product"):
        rows_edges.append(("product", row["product_code"], "family", row["family"] or "(bilinmiyor)", ingested_at))

    for row in specs.execute("SELECT doc_id, product_codes FROM document"):
        try:
            codes = json.loads(row["product_codes"] or "[]")
        except json.JSONDecodeError:
            codes = []
        for code in codes:
            rows_edges.append(("chunk_set", row["doc_id"], "product", code, ingested_at))

    specs.close()
    return rows_runs, rows_edges, rows_conflicts


def text_sha256(text: str) -> str:
    """Every hash is stored as `sha256:<hex>`."""
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_chunk_index(chunk_sets, ingested_at):
    """Converts `all_chunks.json` into chunk_index rows.

    The sole reason this index exists: to resolve, in a SINGLE query, which FILE
    an evidence chunk from a conversation log (`turns.jsonl::chunks[].id`) came
    from. `source_path` is stored here and VISIBLE in the panel -- it stays
    hidden on the end-user surface.
    """
    rows = []
    for doc_id, chunk_set in chunk_sets.items():
        scope = chunk_set.get("scope")
        for node in chunk_set.get("nodes", []):
            node_id = node.get("node_id")
            if not node_id:
                continue
            text = node.get("text") or ""
            rows.append((
                node_id,
                node.get("doc_id") or doc_id,
                scope,
                node.get("source_path"),
                node.get("fmt"),
                node.get("tree_level"),
                json.dumps(node.get("heading_path") or [], ensure_ascii=False),
                node.get("page_start"),
                node.get("page_end"),
                node.get("token_count"),
                text_sha256(text),
                text,
                json.dumps(node.get("images") or [], ensure_ascii=False),
                ingested_at,
            ))
    return rows


def ingest_vectorize_stage(ingested_at):
    last_touched = vectorize_last_touched()
    return [(
        "vectorize", "__all__", "ON_DISK" if last_touched else "MISSING",
        None, last_touched, 0, None,
        "kok storage/ klasorunun en son degisen dosyasi (koda dokunulmuyor, sadece mtime)",
        ingested_at,
    )]


def main():
    conn = sqlite3.connect(LINEAGE_DB)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    ensure_schema(conn)

    ingested_at = now_iso()
    docs = load_document_nodes()
    chunk_sets = load_chunk_sets()

    parse_chunk_runs, parse_chunk_edges = ingest_parse_and_chunk_stages(conn, docs, chunk_sets, ingested_at)
    facts_runs, facts_edges, conflicts = ingest_facts_stage(conn, docs, ingested_at)
    vectorize_runs = ingest_vectorize_stage(ingested_at)

    conn.executemany(
        "INSERT INTO pipeline_runs (stage, node_id, status, input_hash, ran_at, is_stale, stale_reason, "
        "output_summary, ingested_at) VALUES (?,?,?,?,?,?,?,?,?)",
        parse_chunk_runs + facts_runs + vectorize_runs,
    )
    conn.executemany(
        "INSERT INTO lineage_edges (src_type, src_id, dst_type, dst_id, ingested_at) VALUES (?,?,?,?,?)",
        parse_chunk_edges + facts_edges,
    )
    conn.executemany(
        "INSERT INTO conflicts (node_type, node_id, product_code, block, key, detail, flagged_at, ingested_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        conflicts,
    )

    # chunk_index is NOT append-only (see the rationale in schema.sql): it is a
    # snapshot index of the current chunk set, refreshed on every run.
    chunk_rows = build_chunk_index(chunk_sets, ingested_at)
    conn.execute("DELETE FROM chunk_index")
    conn.executemany(
        "INSERT INTO chunk_index (node_id, doc_id, scope, source_path, fmt, tree_level, heading_path, "
        "page_start, page_end, token_count, text_sha256, text, images, ingested_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        chunk_rows,
    )
    conn.commit()

    n_runs = len(parse_chunk_runs) + len(facts_runs) + len(vectorize_runs)
    n_stale = sum(1 for r in parse_chunk_runs + facts_runs if r[5])
    print(f"ingest tamam: {n_runs} run kaydı, {len(parse_chunk_edges) + len(facts_edges)} kenar, "
          f"{len(conflicts)} conflict, {n_stale} stale-işaretli aşama, "
          f"{len(chunk_rows)} chunk indekslendi. -> {LINEAGE_DB}")
    conn.close()


if __name__ == "__main__":
    main()
