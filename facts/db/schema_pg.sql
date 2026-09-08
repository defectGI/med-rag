-- ACME datasheet spec store — PostgreSQL schema (RAG-ready)
-- Generated from specs.db. Unified EAV: family & category are filterable metadata.
--
-- 2026-07-28 (PROTOCOL KARAR-016/019/025, ROADMAP 13b): mirrors schema_rag.sql --
-- see that file's header comment for the rationale of every column/table added
-- in this wave (product.info_status, document.content_hash, spec_value.review_status/
-- conflict/extraction_run_id, fact_evidence, extraction_run).
--
-- 2026-07-31 (ASAMA A "kuru DB" / iskelet, PROTOCOL KARAR-038): still a mirror --
-- spec_key's dictionary source moved to chatbot-corpus/spec_schema/spec_keys.yaml
-- (`scope` -> `applies_to`, `expected_kind` -> `kind`, new `labels`), `product`
-- gained node_id/product_code/subfamily_2/taxonomy_ids, and `spec_value.status`
-- gained `not_applicable`. Read schema_rag.sql's header for WHY `not_applicable`
-- is not `absent`; the reasoning is not repeated here so the two cannot drift.

DROP VIEW  IF EXISTS spec_chunk;
DROP TABLE IF EXISTS fact_evidence;
DROP TABLE IF EXISTS spec_value;
DROP TABLE IF EXISTS extraction_run;
DROP TABLE IF EXISTS spec_key;
DROP TABLE IF EXISTS document;
DROP TABLE IF EXISTS product;

CREATE TABLE product (
    model         TEXT PRIMARY KEY,
    node_id       TEXT NOT NULL UNIQUE,
    product_code  TEXT,
    family        TEXT NOT NULL,
    subfamily     TEXT,
    subfamily_2   TEXT,
    title         TEXT,
    taxonomy_ids  JSONB NOT NULL DEFAULT '[]',
    info_status   TEXT NOT NULL CHECK (info_status IN ('no_documents','no_extractable_documents','extracted'))
);

-- Multi-document catalog for a product (brochure/datasheet/CE decl./drawing/
-- STP/image/manual/quick-start). One doc_id can legitimately repeat across
-- several models (a subfamily-level file fanned out to its member products),
-- so the PK is the pair, not doc_id alone. NEVER expose source_path to end
-- users - it is resolved by a separate download-endpoint via doc_id.
CREATE TABLE document (
    doc_id        TEXT NOT NULL,
    model         TEXT NOT NULL REFERENCES product(model),
    doc_type      TEXT NOT NULL CHECK (doc_type IN ('BROCHURE','DATASHEET','PRODUCT_IMAGE','CATALOGUE','CE_DECLARATION','TECHNICAL_DRAWING','STP','USER_MANUAL','QUICK_START_GUIDE')),
    file_name     TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    content_hash  TEXT NOT NULL,
    PRIMARY KEY (doc_id, model)
);

CREATE INDEX idx_document_model    ON document(model);
CREATE INDEX idx_document_doc_type ON document(doc_type);

CREATE TABLE extraction_run (
    run_id             TEXT PRIMARY KEY,
    model              TEXT NOT NULL REFERENCES product(model),
    doc_ids            JSONB NOT NULL,
    chunk_shas         JSONB NOT NULL,
    extractor_version  TEXT NOT NULL,
    prompt_version     TEXT NOT NULL,
    generated_at       TEXT NOT NULL
);

CREATE TABLE spec_key (
    key            TEXT PRIMARY KEY,
    category       TEXT NOT NULL CHECK (category IN ('environmental','mechanical','power_input','power_distribution','switching','analog_io','digital_io','connectivity','video','operational','bus_coupler')),
    kind           TEXT NOT NULL CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    unit           TEXT,
    labels         JSONB NOT NULL DEFAULT '[]',
    applies_to     JSONB NOT NULL DEFAULT '[]',
    description    TEXT NOT NULL
);

CREATE TABLE spec_value (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model              TEXT NOT NULL REFERENCES product(model),
    key                TEXT NOT NULL REFERENCES spec_key(key),
    status             TEXT NOT NULL CHECK (status IN ('present','not_specified','absent','not_applicable')),
    kind               TEXT CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    value_num          DOUBLE PRECISION,
    value_min          DOUBLE PRECISION,
    value_typ          DOUBLE PRECISION,
    value_max          DOUBLE PRECISION,
    value_from         DOUBLE PRECISION,
    value_to           DOUBLE PRECISION,
    value_bool         BOOLEAN,
    value_text         TEXT,
    value_list         JSONB,
    unit               TEXT,
    condition          TEXT,
    raw                TEXT NOT NULL DEFAULT '',
    -- full-text search over the verbatim cell (RAG keyword retrieval)
    raw_tsv            tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(raw,''))) STORED,
    -- embedding column for pgvector (uncomment after: CREATE EXTENSION vector;)
    -- embedding vector(1536),
    review_status      TEXT NOT NULL DEFAULT 'auto'
                         CHECK (review_status IN ('auto','human_verified','rejected','needs_review','conflict')),
    conflict           BOOLEAN NOT NULL DEFAULT FALSE,
    extraction_run_id  TEXT REFERENCES extraction_run(run_id),
    UNIQUE (model, key)
);

CREATE TABLE fact_evidence (
    evidence_id     TEXT PRIMARY KEY,
    fact_id         BIGINT NOT NULL REFERENCES spec_value(id),
    doc_id          TEXT NOT NULL,
    chunk_sha256    TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    heading_path    JSONB NOT NULL DEFAULT '[]',
    chunk_node_id   TEXT,
    specificity     TEXT NOT NULL CHECK (specificity IN ('family','subfamily','subfamily_2','product')),
    derivation      TEXT NOT NULL CHECK (derivation IN ('direct','inherited')),
    overridden_by   TEXT REFERENCES fact_evidence(evidence_id),
    stale           BOOLEAN NOT NULL DEFAULT FALSE,
    added_at        TEXT NOT NULL
);

CREATE INDEX idx_fact_evidence_fact_id ON fact_evidence(fact_id);
CREATE INDEX idx_fact_evidence_doc_id  ON fact_evidence(doc_id);

CREATE INDEX idx_spec_value_key      ON spec_value(key);
CREATE INDEX idx_spec_value_model    ON spec_value(model);
CREATE INDEX idx_spec_value_status   ON spec_value(status);
CREATE INDEX idx_spec_value_num      ON spec_value(value_num);
CREATE INDEX idx_spec_value_rawtsv   ON spec_value USING GIN (raw_tsv);
CREATE INDEX idx_spec_value_list     ON spec_value USING GIN (value_list);
CREATE INDEX idx_spec_key_category   ON spec_key(category);
CREATE INDEX idx_spec_key_applies_to ON spec_key USING GIN (applies_to);
CREATE INDEX idx_product_family      ON product(family);
CREATE INDEX idx_product_subfamily   ON product(subfamily);
CREATE INDEX idx_product_taxonomy    ON product USING GIN (taxonomy_ids);

CREATE VIEW spec_chunk AS
SELECT v.id, v.model, p.family, k.category, v.key, v.status, v.unit, v.condition, v.raw,
       p.family || ' / ' || v.model || ' — ' || v.key || ': ' || v.raw AS chunk_text
FROM spec_value v
JOIN product  p ON p.model = v.model
JOIN spec_key k ON k.key   = v.key
WHERE v.status = 'present';
