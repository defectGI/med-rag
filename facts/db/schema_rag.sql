-- =====================================================================
-- ACME Chatbot RAG -- specs.db (SQLite)
--
-- 2026-08-04 -- LLM-SQL SIMPLIFICATION (user decision). The previous
-- version (2026-08-03, KARAR-045..053) of chatbot-corpus/spec_schema/
-- schema.sql carried the taxonomy tree (product_node -> document_owner
-- recursive fan-out) + 6-part attribute dictionary (attribute_block/
-- attribute/attribute_condition/attribute_label/attribute_subfield/
-- attribute_applicability) + 3-part fact layer (attribute_value/
-- attribute_value_item/attribute_value_subfield + separate
-- fact_evidence/extraction_run) -- 14 tables + 6 views. Observation
-- (user, 2026-08-04): text-to-SQL models (especially small/local ones,
-- e.g. gemma4:26b) write multi-JOIN chains UNRELIABLY.
-- Legacy specs.db (facts/db/specs.db.legacy_2026-07-28_08-05-41) was 4
-- flat tables (product/spec_key/spec_value/document, `condition` plain
-- TEXT, everything readable with a single join) -- THIS VERSION
-- DELIBERATELY RETURNS TO THAT DESIGN, while preserving 2026-08-03's
-- gains (kind-payload guards, present/absent/not_specified/
-- not_applicable/conflicting separation, fully materialized matrix,
-- provenance).
--
-- THIS FILE IS NO LONGER the LITERAL SQLite TRANSLATION of
-- chatbot-corpus/spec_schema/schema.sql (the previous version's
-- docstring said so) -- DELIBERATE DIVERGENCE: the taxonomy tree
-- (product_node, 6-level recursive tree) and M:N document_owner are
-- NOT here, intentionally flattened for the chatbot's LLM-SQL surface.
-- Canonical taxonomy identity still lives in PG schema.sql;
-- `build_facts_db.py` (PHASE A) READS it (via product_nodes.json /
-- document_nodes.json) but WRITES THE RESULT (family/subfamily/
-- subfamily_2 plain text, doc->product_codes JSON list) to this
-- simplified schema -- tree traversal happens at BUILD TIME
-- (Python, once), not at QUERY TIME (LLM SQL).
--
-- 4 TABLES: product, document, attribute, spec_value.
-- + 2 VIEWS: v_product_spec (convenience, almost identical to spec_value),
--   v_document_product (BACKWARD COMPATIBILITY for
--   chatbot/chatbot/flows/doc_download.py's SqliteDocumentLookup --
--   same columns as the old document_owner+recursive-CTE view
--   (doc_id, product_code, doc_type, file_name, is_active), built via
--   json_each from `document.product_codes`; the chatbot code runs
--   WITHOUT MODIFICATION).
--
-- THINGS LOST / WEAKENED (asked of the user and APPROVED, 2026-08-04):
--   1. Taxonomy tree left the DB -- only flat family/subfamily/
--      subfamily_2 text remained. Category-level queries are no longer
--      solvable in the DB (the LLM did not need it; the chatbot's own
--      recursive code -- v_document_product view -- carries it).
--   2. `condition` is NOT a closed vocabulary -- now plain TEXT
--      (attribute.conditions JSON is only REFERENCE/display; not
--      validated at INSERT time; validated by the loader/code).
--   3. Multi-evidence (the old fact_evidence child table) is EMBEDDED in
--      `spec_value.evidence` JSON array -- data was NOT lost; it
--      cannot be queried row-by-row via SQL (and the LLM was not
--      expected to write it anyway).
--   4. `extraction_run` table deleted (audit-only) -- the per-run
--      doc_ids/chunk_shas list was lost; `spec_value.extractor*`/
--      `extracted_at` per-row trace remains.
--   5. `attribute.question`/`notes` are NOT COLUMNS ANY MORE -- this
--      is not "table-worthy" information (user); they are embedded in
--      the EXPLANATION text of schema.yaml produced by
--      `build_schema.py` (the LLM SEES them when reading the schema,
--      but cannot SELECT them via SQL).
--
-- Translation glossary (UNCHANGED from the previous version):
--   CREATE TYPE ... ENUM   -> CHECK (x IN (...))
--   plpgsql trigger        -> SQLite trigger + RAISE(ABORT, ...)
--   NUM_NONNULLS(a,b)      -> ((a IS NOT NULL) + (b IS NOT NULL))
-- =====================================================================

PRAGMA foreign_keys = ON;


-- =====================================================================
-- 1. PRODUCT -- identity + commercial data. NO taxonomy tree,
--    family/subfamily/subfamily_2 are plain text (copied from
--    product_nodes.json at build time, KARAR-025 -- code-populated).
-- =====================================================================

CREATE TABLE product (
    product_code      TEXT PRIMARY KEY,                -- 'PN1309'. The single
                                                        -- product from KARAR-013
                                                        -- that has a node_id
                                                        -- but no product_code
                                                        -- CANNOT BE REPRESENTED
                                                        -- in this schema -- the
                                                        -- build script SKIPS it
                                                        -- and reports it
                                                        -- (known, accepted
                                                        -- loss: 250 -> 249).
    display_name      TEXT NOT NULL,
    family             TEXT NOT NULL,
    subfamily          TEXT,
    subfamily_2        TEXT,
    acme_code         TEXT,
    list_price         NUMERIC,
    price_on_request   INTEGER NOT NULL DEFAULT 0 CHECK (price_on_request IN (0,1)),
    valid_until        TEXT,
    price_break_10_99    NUMERIC,
    price_break_100_499  NUMERIC,
    price_break_500_999  NUMERIC,
    is_variant         INTEGER NOT NULL DEFAULT 0 CHECK (is_variant IN (0,1)),
    variant_base       TEXT,
    CONSTRAINT product_variant_ck CHECK (is_variant = 0 OR variant_base IS NOT NULL)
);
CREATE INDEX product_family_ix ON product (family, subfamily);


-- =====================================================================
-- 2. DOCUMENT -- one physical file = one row (KARAR-019, unchanged).
--    Fan-out is NO LONGER a separate M:N table -- `product_codes` is
--    a JSON array computed at build time (the product_nodes.json tree
--    traversed ONCE in Python). No recursive query at runtime.
-- =====================================================================

CREATE TABLE document (
    doc_id         TEXT PRIMARY KEY,
    file_name      TEXT NOT NULL,
    extension      TEXT NOT NULL,
    rel_path       TEXT NOT NULL,
    file_path      TEXT,                             -- NEVER shown to end users / models
    content_hash   TEXT NOT NULL,                    -- 'sha256:...' (KARAR-002)
    size_bytes     INTEGER,
    last_modified_time TEXT,
    doc_type       TEXT NOT NULL CHECK (doc_type IN (
                       'BROCHURE','DATASHEET','PRODUCT_IMAGE','CATALOGUE','CE_DECLARATION',
                       'TECHNICAL_DRAWING','STP','USER_MANUAL','QUICK_START_GUIDE')),
    is_active      INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
    trust_rank     INTEGER GENERATED ALWAYS AS (
        CASE doc_type
            WHEN 'DATASHEET'         THEN 100
            WHEN 'USER_MANUAL'       THEN 90
            WHEN 'CE_DECLARATION'    THEN 85
            WHEN 'TECHNICAL_DRAWING' THEN 80
            WHEN 'CATALOGUE'         THEN 60
            WHEN 'BROCHURE'          THEN 50
            WHEN 'QUICK_START_GUIDE' THEN 40
            ELSE 10
        END) STORED,
    -- JSON array: every leaf product_code this file covers
    -- (directly + via inheritance/fan-out) -- KARAR-014 fan-out's
    -- PRE-COMPUTED build-time form. `v_document_product` view unnests it.
    product_codes  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX document_type_ix ON document (doc_type) WHERE is_active = 1;
CREATE INDEX document_hash_ix ON document (content_hash);


-- =====================================================================
-- 3. ATTRIBUTE -- attribute dictionary, collapsed into ONE table
--    (union of the previous 6 tables -- block/attribute/condition/
--    label/subfield/applicability). Not closed (only REFERENCE/display):
--    labels/conditions/subfields/applies_to are embedded as JSON arrays.
--
--    `question`/`notes` are DELIBERATELY MISSING -- this is not
--    "table-worthy" information (user decision, 2026-08-04): embedded in
    -- the EXPLANATION text of schema.yaml produced by
    -- `build_schema.py`, not as DB COLUMNS.
-- =====================================================================

CREATE TABLE attribute (
    block         TEXT NOT NULL,                     -- 'core','analog_io',...
    key           TEXT NOT NULL,                      -- 'operating_temperature'
    kind          TEXT NOT NULL
                  CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    unit          TEXT,                               -- canonical unit
    enum_values   TEXT,                                -- JSON array or NULL
    accepts_units TEXT,                                -- JSON array or NULL (R3.5)
    labels        TEXT NOT NULL DEFAULT '[]',          -- JSON array: labels observed in the corpus
    conditions    TEXT NOT NULL DEFAULT '[]',          -- JSON array: NON-CLOSED reference list
    subfields     TEXT NOT NULL DEFAULT '[]',          -- JSON array: [{"name":...,"unit":...}]
    applies_to    TEXT NOT NULL DEFAULT '[]',          -- JSON array: family/subfamily scope names
    is_core       INTEGER NOT NULL DEFAULT 0 CHECK (is_core IN (0,1)),
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
    PRIMARY KEY (block, key)
);
CREATE INDEX attribute_key_ix ON attribute (key);


-- =====================================================================
-- 4. SPEC_VALUE -- facts, collapsed into ONE table (union of the
--    previous attribute_value + attribute_value_item +
--    attribute_value_subfield + fact_evidence). family/subfamily/kind/
--    unit DENORMALIZED (so they can be read WITHOUT a product/attribute
--    join -- populated by code, not by the LLM).
--
--    PHASE A ("dry DB"): fills this table with the FULL (product x
--    attribute) matrix, value columns NULL. PHASE C (load_to_db.py)
--    UPDATEs it (present/absent), never INSERTs a new bootstrap row
--    (except extra condition/conflicting).
-- =====================================================================

CREATE TABLE spec_value (
    value_id      INTEGER PRIMARY KEY,
    product_code  TEXT    NOT NULL REFERENCES product(product_code) ON DELETE CASCADE,
    family        TEXT,                                -- denormalized (product.family)
    subfamily     TEXT,                                -- denormalized (product.subfamily)
    block         TEXT NOT NULL,
    key           TEXT NOT NULL,
    kind          TEXT NOT NULL
                  CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    unit          TEXT,                                -- denormalized (attribute.unit, canonical)
    condition     TEXT,                                -- plain TEXT (NOT a closed vocabulary)
    status        TEXT NOT NULL CHECK (status IN (
                      'present','not_specified','absent','not_applicable',
                      'conflicting','superseded')),

    num_value     NUMERIC,
    text_value    TEXT,
    val_min       NUMERIC,
    val_typ       NUMERIC,
    val_max       NUMERIC,
    bool_value    INTEGER CHECK (bool_value IS NULL OR bool_value IN (0,1)),
    items         TEXT,                                -- JSON array (kind=list): [{"ordinal":1,"item_code":...,"item_text":...}]
    subfields     TEXT,                                -- JSON array: [{"name":...,"num_value":...,"text_value":...,"unit_observed":...,"raw_text":...}]

    unit_observed TEXT,
    raw_text      TEXT,

    -- provenance -- PRIMARY source (highest trust_rank) directly in
    -- columns (file_name DENORMALIZED for join-free reads); extra/
    -- corroborating sources in the `evidence` JSON array.
    source_doc_id     TEXT REFERENCES document(doc_id) ON DELETE SET NULL,
    source_file_name  TEXT,                             -- denormalized (document.file_name)
    source_chunk_id   TEXT,
    evidence          TEXT NOT NULL DEFAULT '[]',        -- JSON array: [{"doc_id":...,"file_name":...,"chunk_sha256":...}]

    extractor         TEXT,
    extractor_version TEXT,
    confidence        REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    extracted_at      TEXT,

    review_status      TEXT NOT NULL DEFAULT 'auto'
                        CHECK (review_status IN ('auto','human_verified','rejected','needs_review','conflict')),
    conflict            INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT sv_absent_needs_source_ck
        CHECK (status <> 'absent' OR source_chunk_id IS NOT NULL),
    CONSTRAINT sv_empty_status_ck CHECK (
        status IN ('present','conflicting','superseded')
        OR (num_value IS NULL AND text_value IS NULL AND val_min IS NULL
            AND val_typ IS NULL AND val_max IS NULL AND bool_value IS NULL)),
    CONSTRAINT sv_range_ck CHECK (val_min IS NULL OR val_max IS NULL OR val_min <= val_max),
    CONSTRAINT sv_typ_ck   CHECK (val_typ IS NULL
        OR ((val_min IS NULL OR val_typ >= val_min) AND (val_max IS NULL OR val_typ <= val_max)))
);
CREATE INDEX sv_product_ix ON spec_value (product_code);
CREATE INDEX sv_key_ix     ON spec_value (key, status);
CREATE INDEX sv_num_ix     ON spec_value (key, num_value) WHERE num_value IS NOT NULL;
CREATE INDEX sv_range_ix   ON spec_value (key, val_min, val_max);
-- ONE live value per product+key+condition (same as the old
-- av_unique_live_uix).
CREATE UNIQUE INDEX sv_unique_live_uix
    ON spec_value (product_code, block, key, COALESCE(condition, ''))
    WHERE status IN ('present','not_specified','absent','not_applicable');

-- ---------------------------------------------------------------------
-- Guard: the payload must match the `kind` declared by the dictionary
-- (same as the old attribute_value_kind_guard -- no `attribute` join
-- is NEEDED now, kind is already spec_value's own column).
-- ---------------------------------------------------------------------
CREATE TRIGGER spec_value_kind_guard_ins
BEFORE INSERT ON spec_value
FOR EACH ROW WHEN NEW.status IN ('present','conflicting','superseded')
BEGIN
    SELECT CASE
        WHEN NEW.kind = 'single' AND ((NEW.num_value IS NOT NULL) + (NEW.text_value IS NOT NULL)) <> 1
            THEN RAISE(ABORT, 'kind=single needs exactly one of num_value/text_value')
        WHEN NEW.kind = 'boolean' AND NEW.bool_value IS NULL
            THEN RAISE(ABORT, 'kind=boolean needs bool_value')
        WHEN NEW.kind = 'range' AND NEW.val_min IS NULL AND NEW.val_max IS NULL
            THEN RAISE(ABORT, 'kind=range needs val_min and/or val_max')
        WHEN NEW.kind = 'min_typ_max'
             AND ((NEW.val_min IS NOT NULL) + (NEW.val_typ IS NOT NULL) + (NEW.val_max IS NOT NULL)) = 0
            THEN RAISE(ABORT, 'kind=min_typ_max needs at least one of min/typ/max')
        WHEN NEW.kind = 'list' AND (NEW.items IS NULL OR NEW.items = '[]')
            THEN RAISE(ABORT, 'kind=list needs items[]')
        WHEN NEW.kind = 'conditional' AND NEW.condition IS NULL
            THEN RAISE(ABORT, 'kind=conditional requires a condition')
    END;
END;

CREATE TRIGGER spec_value_kind_guard_upd
BEFORE UPDATE ON spec_value
FOR EACH ROW WHEN NEW.status IN ('present','conflicting','superseded')
BEGIN
    SELECT CASE
        WHEN NEW.kind = 'single' AND ((NEW.num_value IS NOT NULL) + (NEW.text_value IS NOT NULL)) <> 1
            THEN RAISE(ABORT, 'kind=single needs exactly one of num_value/text_value')
        WHEN NEW.kind = 'boolean' AND NEW.bool_value IS NULL
            THEN RAISE(ABORT, 'kind=boolean needs bool_value')
        WHEN NEW.kind = 'range' AND NEW.val_min IS NULL AND NEW.val_max IS NULL
            THEN RAISE(ABORT, 'kind=range needs val_min and/or val_max')
        WHEN NEW.kind = 'min_typ_max'
             AND ((NEW.val_min IS NOT NULL) + (NEW.val_typ IS NOT NULL) + (NEW.val_max IS NOT NULL)) = 0
            THEN RAISE(ABORT, 'kind=min_typ_max needs at least one of min/typ/max')
        WHEN NEW.kind = 'list' AND (NEW.items IS NULL OR NEW.items = '[]')
            THEN RAISE(ABORT, 'kind=list needs items[]')
        WHEN NEW.kind = 'conditional' AND NEW.condition IS NULL
            THEN RAISE(ABORT, 'kind=conditional requires a condition')
    END;
END;


-- =====================================================================
-- 5. RETRIEVAL HELPERS (view -- NOT a physical table, not counted)
-- =====================================================================

-- Main answer surface -- now ALMOST IDENTICAL to spec_value ITSELF
-- (family/subfamily/unit/source_file_name already denormalized); it
-- only exists to filter present/absent and add product.display_name.
CREATE VIEW v_product_spec AS
SELECT p.product_code,
       p.display_name,
       sv.family, sv.subfamily,
       sv.block, sv.key, sv.condition, sv.kind, sv.unit,
       sv.status,
       sv.num_value, sv.text_value, sv.val_min, sv.val_typ, sv.val_max, sv.bool_value,
       sv.unit_observed, sv.raw_text,
       sv.source_file_name AS source_document,
       sv.source_chunk_id
  FROM spec_value sv
  JOIN product p ON p.product_code = sv.product_code
 WHERE sv.status IN ('present','absent');

-- Coverage gap -- unchanged (no attribute join needed now; key/block are
-- already in spec_value; question lives in schema.yaml).
CREATE VIEW v_coverage_gap AS
SELECT product_code, family, block, key
  FROM spec_value
 WHERE status = 'not_specified';

-- BACKWARD COMPATIBILITY with chatbot/chatbot/flows/doc_download.py::
-- SqliteDocumentLookup -- SAME columns (doc_id, product_code, doc_type,
-- file_name, is_active), REPLACING the old recursive-CTE + document_owner
-- by unnesting `document.product_codes` JSON array via json_each. The
-- chatbot code runs WITHOUT MODIFICATION.
CREATE VIEW v_document_product AS
SELECT d.doc_id,
       je.value AS product_code,
       d.doc_type,
       d.file_name,
       d.is_active,
       d.trust_rank
  FROM document d, json_each(d.product_codes) je;
