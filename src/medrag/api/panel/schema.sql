-- The panel's OWN observation database -- it does not touch specs.db and does
-- not break the specs.db schema contract. Append-only: pipeline_runs gets new
-- rows on every ingest.py run and is never overwritten -- run-history
-- accumulates this way.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    stage         TEXT NOT NULL,       -- 'parse' | 'chunk' | 'facts' | 'vectorize'
    node_id       TEXT NOT NULL,       -- doc_id (facts are also summarized per doc_id)
    status        TEXT,                -- the source's own status field (SUCCESS/SKIPPED/...)
    input_hash    TEXT,                -- sha256:<hex>
    ran_at        TEXT,                -- timestamp in the source (offset ISO)
    is_stale      INTEGER NOT NULL DEFAULT 0,
    stale_reason  TEXT,
    output_summary TEXT,
    ingested_at   TEXT NOT NULL        -- time of this ingest.py run (offset ISO)
);

CREATE INDEX IF NOT EXISTS idx_runs_stage_node ON pipeline_runs(stage, node_id, ingested_at);

CREATE TABLE IF NOT EXISTS lineage_edges (
    edge_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type    TEXT NOT NULL,   -- 'document' | 'chunk_set' | 'product' | 'family'
    src_id      TEXT NOT NULL,
    dst_type    TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON lineage_edges(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON lineage_edges(dst_type, dst_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type     TEXT NOT NULL,   -- 'spec_value'
    node_id       TEXT NOT NULL,   -- specs.db value_id (text, for comparison)
    product_code  TEXT,
    block         TEXT,
    key           TEXT,
    detail        TEXT,
    flagged_at    TEXT NOT NULL,
    ingested_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conflicts_product ON conflicts(product_code);

-- chunk_index -- the SEARCHABLE index over `all_chunks.json`. An evidence chunk
-- from a conversation log (`turns.jsonl::chunks[].id`, i.e. `node_id`) resolves
-- to its source FILE here with a SINGLE query; otherwise an 1891-node JSON would
-- be scanned on every click.
--
-- NOTE -- this table is NOT append-only (unlike pipeline_runs): it is not a
-- HISTORY but a SNAPSHOT index of the current chunk set; every ingest.py run
-- refreshes its content (DELETE + INSERT). Rationale: `node_id` is a natural
-- primary key and "what does this chunk look like now" must have a single
-- answer. A chunk's change over TIME is already tracked via `text_sha256` +
-- pipeline_runs.
CREATE TABLE IF NOT EXISTS chunk_index (
    node_id      TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    scope        TEXT,
    source_path  TEXT,            -- source FILE path (visible in the panel)
    fmt          TEXT,
    tree_level   INTEGER,
    heading_path TEXT,            -- JSON array
    page_start   INTEGER,
    page_end     INTEGER,
    token_count  INTEGER,
    text_sha256  TEXT,            -- sha256:<hex> -- basis for staleness detection
    text         TEXT,            -- UNTRUNCATED body (the evidence panel shows truncation)
    images       TEXT,            -- JSON array (image_id/visual_type/ocr_text)
    ingested_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_index_doc ON chunk_index(doc_id);
