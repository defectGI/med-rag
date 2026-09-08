-- =====================================================================
-- ACME Chatbot RAG — database schema (DDL only, no data)
-- PostgreSQL 15+ / pgvector
--
-- Sources this schema must hold:
--   product_nodes.json    297 nodes (12 family / 30 subfamily / 5 subfamily_2 / 250 product)
--   document_nodes.json   772 files, 9 doc types, M:N link to product nodes
--   all_chunks.json       198 chunked docs, 1891 chunks
--   attribute_schema_v3.json  spec-key dictionary governed by attribute_rules.txt
--
-- Layering:
--   1. taxonomy      product_node, product
--   2. corpus        document, document_owner, chunk (+ satellites), embedding
--   3. dictionary    attribute + condition/label/subfield/applicability  <- R0-R3 live here
--   4. facts         attribute_value  <- the RAG-facing extraction target
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- =====================================================================
-- 1. TAXONOMY
-- =====================================================================

CREATE TYPE product_node_type AS ENUM ('family', 'subfamily', 'subfamily_2', 'product');

CREATE TABLE product_node (
    node_id        TEXT PRIMARY KEY,                 -- 'n_6175f1289a78'
    node_type      product_node_type NOT NULL,
    parent_id      TEXT REFERENCES product_node(node_id) ON DELETE RESTRICT,
    source_ref     TEXT,                             -- '1.1.1' catalogue numbering
    depth          SMALLINT NOT NULL CHECK (depth BETWEEN 1 AND 4),
    is_leaf        BOOLEAN NOT NULL,
    family         TEXT NOT NULL,
    subfamily      TEXT,
    subfamily_2    TEXT,
    reference_code TEXT,                             -- category.reference_code
    CONSTRAINT product_node_root_ck
        CHECK ((parent_id IS NULL) = (node_type = 'family')),
    CONSTRAINT product_node_leaf_ck
        CHECK (is_leaf = (node_type = 'product'))
);
CREATE INDEX product_node_parent_ix ON product_node (parent_id);
CREATE INDEX product_node_family_ix ON product_node (family, subfamily);

-- Commercial / identity attributes. Deliberately NOT spec keys: R3.1 excludes
-- identity+order codes and R3.7 excludes taxonomy restatements, so price and
-- product_code live here and are answered from this table, never from attribute_value.
CREATE TABLE product (
    node_id           TEXT PRIMARY KEY REFERENCES product_node(node_id) ON DELETE CASCADE,
    product_code      TEXT NOT NULL,                 -- 'PN1162'
    acme_code        TEXT,                          -- 'DC1052800001'
    display_name      TEXT NOT NULL,
    list_price        NUMERIC(12,2),
    price_on_request  BOOLEAN NOT NULL DEFAULT FALSE,
    valid_until       DATE,
    price_break_10_99    NUMERIC(12,2),
    price_break_100_499  NUMERIC(12,2),
    price_break_500_999  NUMERIC(12,2),
    is_variant        BOOLEAN NOT NULL DEFAULT FALSE,
    variant_base      TEXT,                          -- product_code of the base variant
    CONSTRAINT product_price_ck
        CHECK (price_on_request OR list_price IS NOT NULL),
    CONSTRAINT product_variant_ck
        CHECK (NOT is_variant OR variant_base IS NOT NULL)
);
CREATE UNIQUE INDEX product_code_uix ON product (product_code);
CREATE INDEX product_name_trgm_ix ON product USING gin (display_name gin_trgm_ops);

CREATE TYPE product_relation_type AS ENUM
    ('compatible_with', 'accessory_of', 'variant_of', 'requires');

CREATE TABLE product_relation (
    from_node_id  TEXT NOT NULL REFERENCES product_node(node_id) ON DELETE CASCADE,
    to_node_id    TEXT NOT NULL REFERENCES product_node(node_id) ON DELETE CASCADE,
    relation      product_relation_type NOT NULL,
    evidence_chunk_id TEXT,                          -- FK added after chunk exists
    PRIMARY KEY (from_node_id, to_node_id, relation),
    CONSTRAINT product_relation_noself_ck CHECK (from_node_id <> to_node_id)
);


-- =====================================================================
-- 2. CORPUS
-- =====================================================================

CREATE TYPE doc_type AS ENUM (
    'BROCHURE', 'DATASHEET', 'PRODUCT_IMAGE', 'CATALOGUE', 'CE_DECLARATION',
    'TECHNICAL_DRAWING', 'STP', 'USER_MANUAL', 'QUICK_START_GUIDE');

CREATE TYPE scan_status  AS ENUM ('NEW','UNCHANGED','MODIFIED','MOVED','DELETED');
CREATE TYPE stage_status AS ENUM ('SUCCESS','FAILED','SKIPPED','PENDING');

CREATE TABLE document (
    doc_id         UUID PRIMARY KEY,
    file_name      TEXT NOT NULL,
    extension      TEXT NOT NULL,
    rel_path       TEXT NOT NULL,
    file_path      TEXT,
    content_hash   TEXT NOT NULL,                    -- 'sha256:...'
    size_bytes     BIGINT,
    last_modified_time TIMESTAMPTZ,
    scan_status    scan_status NOT NULL,
    doc_type       doc_type NOT NULL,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE,
    -- doc_type drives extraction trust: a DATASHEET spec table outranks a
    -- BROCHURE bullet for the same key on the same product.
    trust_rank     SMALLINT GENERATED ALWAYS AS (
        CASE doc_type
            WHEN 'DATASHEET'         THEN 100
            WHEN 'USER_MANUAL'       THEN 90
            WHEN 'CE_DECLARATION'    THEN 85
            WHEN 'TECHNICAL_DRAWING' THEN 80
            WHEN 'CATALOGUE'         THEN 60
            WHEN 'BROCHURE'          THEN 50
            WHEN 'QUICK_START_GUIDE' THEN 40
            ELSE 10
        END) STORED
);
CREATE INDEX document_type_ix ON document (doc_type) WHERE is_active;
CREATE INDEX document_hash_ix ON document (content_hash);

-- M:N. Observed: one brochure owned by 13 product nodes (link_count 13).
CREATE TABLE document_owner (
    doc_id             UUID NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    node_id            TEXT NOT NULL REFERENCES product_node(node_id) ON DELETE CASCADE,
    is_primary         BOOLEAN NOT NULL DEFAULT FALSE,   -- links.matched_node_id
    resolution_method  TEXT,                             -- 'code_match' | 'code_match_expanded' | 'folder_name_match_expanded'
    PRIMARY KEY (doc_id, node_id)
);
CREATE INDEX document_owner_node_ix ON document_owner (node_id);
CREATE UNIQUE INDEX document_owner_primary_uix
    ON document_owner (doc_id) WHERE is_primary;

-- Parse/OCR/chunk provenance, kept separate so re-parsing does not touch document.
CREATE TABLE document_pipeline (
    doc_id             UUID PRIMARY KEY REFERENCES document(doc_id) ON DELETE CASCADE,
    parser             TEXT,
    parser_version     TEXT,
    parse_status       stage_status,
    parsed_from_hash   TEXT,
    parsed_json_path   TEXT,
    last_parsed        TIMESTAMPTZ,
    parse_error        TEXT,
    ocr_model          TEXT,   ocr_status        stage_status, ocr_at        TIMESTAMPTZ,
    ocr_check_model    TEXT,   ocr_check_status  stage_status, ocr_check_at  TIMESTAMPTZ,
    table_desc_model   TEXT,   table_desc_status stage_status, table_desc_at TIMESTAMPTZ,
    chunk_status       stage_status,
    chunked_at         TIMESTAMPTZ,
    chunked_from_hash  TEXT,
    chunker_version    TEXT,
    ir_version         SMALLINT,
    -- TRUE when the chunked hash no longer matches the file on disk -> stale RAG index.
    is_stale           BOOLEAN GENERATED ALWAYS AS
                       (chunked_from_hash IS DISTINCT FROM parsed_from_hash) STORED
);

CREATE TYPE chunk_split_kind AS ENUM ('none','token_overflow','heading','table','list');
CREATE TYPE provenance_level AS ENUM ('verified','partial','unverified');

CREATE TABLE chunk (
    chunk_id        TEXT PRIMARY KEY,                 -- '<doc_id>::c0'
    doc_id          UUID NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    tree_level      SMALLINT NOT NULL DEFAULT 0,
    ordinal         INTEGER NOT NULL,                 -- parsed from ::cN, gives stable order
    body            TEXT NOT NULL,
    heading_path    TEXT[] NOT NULL DEFAULT '{}',     -- ['2. Hardware Overview','2.2. Hardware Specifications']
    source_path     TEXT,
    fmt             TEXT,                             -- 'pdf'
    source_block_ids TEXT[] NOT NULL DEFAULT '{}',    -- ['b0','b1',...] -> back to parser IR
    page_start      INTEGER,
    page_end        INTEGER,
    token_count     INTEGER,
    token_limit     INTEGER,
    flex_applied    BOOLEAN NOT NULL DEFAULT FALSE,
    flex_amount     INTEGER  NOT NULL DEFAULT 0,
    flex_reason     TEXT,
    split_kind      chunk_split_kind,
    split_index     SMALLINT,
    split_total     SMALLINT,
    overlap_units   INTEGER,
    prev_chunk_id   TEXT REFERENCES chunk(chunk_id) ON DELETE SET NULL,
    next_chunk_id   TEXT REFERENCES chunk(chunk_id) ON DELETE SET NULL,
    parent_chunk_id TEXT REFERENCES chunk(chunk_id) ON DELETE SET NULL,
    provenance      provenance_level,
    CONSTRAINT chunk_pages_ck  CHECK (page_end IS NULL OR page_end >= page_start),
    CONSTRAINT chunk_split_ck  CHECK (split_index IS NULL OR split_index <= split_total)
);
CREATE UNIQUE INDEX chunk_doc_ordinal_uix ON chunk (doc_id, ordinal);
CREATE INDEX chunk_heading_ix ON chunk USING gin (heading_path);
CREATE INDEX chunk_body_fts_ix ON chunk USING gin (to_tsvector('simple', body));

CREATE TABLE chunk_keyword (
    chunk_id TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    keyword  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, keyword)
);
CREATE INDEX chunk_keyword_kw_ix ON chunk_keyword (keyword);

CREATE TYPE visual_type AS ENUM
    ('product_photo','technical_drawing','schematic','table','chart','logo','other');

CREATE TABLE chunk_image (
    chunk_id    TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    image_id    TEXT NOT NULL,                        -- 'sha256:...' content-addressed, dedupes across docs
    visual_type visual_type,
    ocr_text    TEXT,
    PRIMARY KEY (chunk_id, image_id)
);
CREATE INDEX chunk_image_id_ix ON chunk_image (image_id);

CREATE TABLE chunk_cross_ref (
    chunk_id  TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    ref_label TEXT NOT NULL,                          -- 'Table 5', 'Figure 3'
    target_chunk_id TEXT REFERENCES chunk(chunk_id) ON DELETE SET NULL,
    PRIMARY KEY (chunk_id, ref_label)
);

ALTER TABLE product_relation
    ADD CONSTRAINT product_relation_evidence_fk
    FOREIGN KEY (evidence_chunk_id) REFERENCES chunk(chunk_id) ON DELETE SET NULL;

-- Embeddings kept out of chunk so the model can be swapped without rewriting text.
CREATE TABLE embedding_model (
    model_id   SMALLSERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    is_active  BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (name)
);

CREATE TABLE chunk_embedding (
    chunk_id  TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    model_id  SMALLINT NOT NULL REFERENCES embedding_model(model_id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,                  -- set to the active model's dimensions
    PRIMARY KEY (chunk_id, model_id)
);
CREATE INDEX chunk_embedding_hnsw_ix ON chunk_embedding
    USING hnsw (embedding vector_cosine_ops);


-- =====================================================================
-- 3. ATTRIBUTE DICTIONARY  (the rules materialised)
-- =====================================================================

CREATE TYPE attribute_kind AS ENUM
    ('single','min_typ_max','range','list','conditional','boolean');

-- Blocks from attribute_schema_v3.json: 'core' + one row per family block.
CREATE TABLE attribute_block (
    block_id   SMALLSERIAL PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,                  -- 'core','analog_io','switching_relay',...
    is_core    BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT
);

CREATE TABLE attribute (
    attribute_id  SERIAL PRIMARY KEY,
    block_id      SMALLINT NOT NULL REFERENCES attribute_block(block_id) ON DELETE RESTRICT,
    key           TEXT NOT NULL,                      -- 'operating_temperature'
    kind          attribute_kind NOT NULL,
    unit          TEXT,                               -- canonical unit; NULL for enum/boolean/list
    -- R0.1: askability is a NOT NULL column, not a convention. A key with no
    -- customer question cannot be inserted.
    question      TEXT NOT NULL,
    notes         TEXT,
    -- R0.3 / R0.4 audit trail, carried from the mining pass.
    evidence_ndocs           INTEGER,
    evidence_distinct_values INTEGER,
    is_family_defining       BOOLEAN NOT NULL DEFAULT FALSE,  -- R0.3 second clause
    is_active                BOOLEAN NOT NULL DEFAULT TRUE,
    retired_by_rule          TEXT,                            -- 'R0.4','R3.2',...
    retired_reason           TEXT,

    -- R1.1 no extremum prefixes
    CONSTRAINT attr_r1_1_ck CHECK (key !~ '^(max|min|typ|peak|nominal|rated|avg)_'),
    -- R1.1 no extremum/range suffixes either (v2 had _max, _range)
    CONSTRAINT attr_r1_1b_ck CHECK (key !~ '_(max|min|typ|range)$'),
    -- R1.2 conditions do not go in the name
    CONSTRAINT attr_r1_2_ck CHECK (key !~ '_(at_[0-9]|no_load|per_channel|front_panel|rear_panel)'),
    -- R1.4 counters follow exactly one pattern
    CONSTRAINT attr_r1_4_ck CHECK (key !~ '^(number_of_|total_number_of_|num_)'),
    -- R1.6 units do not go in the name
    CONSTRAINT attr_r1_6_ck CHECK (key !~ '_(v|mv|a|ma|w|hz|khz|mhz|mm|ohm|bit|bits|ms|us|db)$'),
    -- R0.1 the question must be a real question, not a placeholder
    CONSTRAINT attr_r0_1_ck CHECK (length(btrim(question)) >= 10),
    -- R0.2 a numeric kind needs a unit; enum/list/boolean must not carry one
    CONSTRAINT attr_r0_2_ck CHECK (
        (kind IN ('min_typ_max','range') AND unit IS NOT NULL)
        OR kind IN ('single','list','conditional','boolean')),
    -- R0.4 a key surviving in the corpus must show >= 2 distinct values,
    -- unless it has not been mined yet (NULL).
    CONSTRAINT attr_r0_4_ck CHECK (
        NOT is_active OR evidence_distinct_values IS NULL OR evidence_distinct_values >= 2),
    -- R0.3 spread: >=2 documents, or explicitly family-defining
    CONSTRAINT attr_r0_3_ck CHECK (
        NOT is_active OR evidence_ndocs IS NULL OR evidence_ndocs >= 2 OR is_family_defining),
    -- retirement bookkeeping
    CONSTRAINT attr_retire_ck CHECK (is_active OR retired_by_rule IS NOT NULL)
);
-- R2.5 first clause: one concept never takes two names *within a block*.
CREATE UNIQUE INDEX attribute_block_key_uix ON attribute (block_id, key);

-- R1.2 / R1.3 — the closed qualifier vocabulary. A value may only carry a
-- condition declared here, which is what stops 'ethernet_data_rate' coming back.
CREATE TABLE attribute_condition (
    condition_id  SERIAL PRIMARY KEY,
    attribute_id  INTEGER NOT NULL REFERENCES attribute(attribute_id) ON DELETE CASCADE,
    code          TEXT NOT NULL,                      -- 'ethernet','per_channel','channel_to_ground'
    label         TEXT,
    UNIQUE (attribute_id, code)
);

-- R1.5 — datasheet labels live here, never in the key name. Also the lookup
-- table the extractor uses to map a mined label to a key.
CREATE TABLE attribute_label (
    attribute_id  INTEGER NOT NULL REFERENCES attribute(attribute_id) ON DELETE CASCADE,
    label         TEXT NOT NULL,                      -- 'Max. Total Output Current'
    label_norm    TEXT GENERATED ALWAYS AS (lower(btrim(label))) STORED,
    PRIMARY KEY (attribute_id, label)
);
CREATE INDEX attribute_label_norm_ix ON attribute_label (label_norm);

-- R2.3 — accuracy/resolution/slew_rate attach to their parent quantity
-- instead of exploding into their own keys.
CREATE TABLE attribute_subfield (
    subfield_id   SERIAL PRIMARY KEY,
    attribute_id  INTEGER NOT NULL REFERENCES attribute(attribute_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,                      -- 'accuracy','resolution','slew_rate','memory_type'
    unit          TEXT,
    UNIQUE (attribute_id, name)
);

-- Family scoping: "generic core + family blocks". A core attribute with no row
-- here applies everywhere; a family attribute is only ever ASKED under its scope.
--
-- KARAR-051 (2026-08-03): this table is GUIDANCE, not a constraint. It decides
-- which questions get asked (and which matrix rows are born 'not_applicable'),
-- but it does NOT reject a value that arrives outside it -- see
-- `attribute_value_scope_guard`. Out-of-scope arrivals surface in
-- `v_out_of_scope_value` and are read as a signal to widen the scope.
CREATE TABLE attribute_applicability (
    attribute_id  INTEGER NOT NULL REFERENCES attribute(attribute_id) ON DELETE CASCADE,
    family        TEXT NOT NULL,
    subfamily     TEXT NOT NULL DEFAULT '',           -- '' = whole family (NOT NULL so it can key)
    PRIMARY KEY (attribute_id, family, subfamily)
);

-- Facets are retrieval labels, not spec keys — closed vocabulary per R0.2.
CREATE TABLE facet_type (
    facet_type_id SMALLSERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE                -- 'function_tags','application_tags','form_factor_tags'
);
CREATE TABLE facet_value (
    facet_value_id SERIAL PRIMARY KEY,
    facet_type_id  SMALLINT NOT NULL REFERENCES facet_type(facet_type_id) ON DELETE CASCADE,
    code           TEXT NOT NULL,
    label          TEXT,
    UNIQUE (facet_type_id, code)
);
CREATE TABLE product_facet (
    node_id        TEXT NOT NULL REFERENCES product_node(node_id) ON DELETE CASCADE,
    facet_value_id INTEGER NOT NULL REFERENCES facet_value(facet_value_id) ON DELETE CASCADE,
    PRIMARY KEY (node_id, facet_value_id)
);


-- =====================================================================
-- 4. FACTS  — extracted values (to be filled later; out of scope here)
-- =====================================================================

CREATE TYPE attribute_status AS ENUM (
    'present',        -- value explicitly stated in the document
    'not_specified',  -- key applies to this product, document gives no value
    'absent',         -- document explicitly declares the feature ABSENT; never inferred
    'conflicting',    -- PROPOSED: two documents disagree for the same product+key+condition
    'superseded'      -- PROPOSED: a newer document overrode this value
);

CREATE TABLE attribute_value (
    value_id      BIGSERIAL PRIMARY KEY,
    node_id       TEXT    NOT NULL REFERENCES product_node(node_id) ON DELETE CASCADE,
    attribute_id  INTEGER NOT NULL REFERENCES attribute(attribute_id) ON DELETE RESTRICT,
    condition_id  INTEGER REFERENCES attribute_condition(condition_id) ON DELETE RESTRICT,
    status        attribute_status NOT NULL,

    -- kind-typed payload; exactly one shape is populated, enforced below
    num_value     NUMERIC,          -- kind = single (numeric)
    text_value    TEXT,             -- kind = single (enum/string)
    val_min       NUMERIC,          -- kind = range | min_typ_max
    val_typ       NUMERIC,          -- kind = min_typ_max
    val_max       NUMERIC,          -- kind = range | min_typ_max
    bool_value    BOOLEAN,          -- kind = boolean
    -- list values live in attribute_value_item

    unit_observed TEXT,             -- unit as written ('kg','VA','GT/s') before normalisation to attribute.unit
    raw_text      TEXT,             -- the verbatim cell/sentence — required for auditing R3.1 stripping

    -- provenance: every fact points at the chunk it came from
    source_chunk_id TEXT REFERENCES chunk(chunk_id) ON DELETE SET NULL,
    source_doc_id   UUID REFERENCES document(doc_id) ON DELETE SET NULL,
    extractor       TEXT,
    extractor_version TEXT,
    confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- (leaf-only enforcement lives in attribute_value_scope_guard; a CHECK
    --  cannot reach product_node.node_type across the FK)
    -- 'absent' is a declaration, never an inference: it must cite a chunk.
    CONSTRAINT av_absent_needs_source_ck
        CHECK (status <> 'absent' OR source_chunk_id IS NOT NULL),
    -- 'not_specified' and 'absent' carry no payload.
    CONSTRAINT av_empty_status_ck CHECK (
        status IN ('present','conflicting','superseded')
        OR (num_value IS NULL AND text_value IS NULL AND val_min IS NULL
            AND val_typ IS NULL AND val_max IS NULL AND bool_value IS NULL)),
    -- a range must not be inverted
    CONSTRAINT av_range_ck CHECK (val_min IS NULL OR val_max IS NULL OR val_min <= val_max),
    CONSTRAINT av_typ_ck   CHECK (val_typ IS NULL
        OR ((val_min IS NULL OR val_typ >= val_min) AND (val_max IS NULL OR val_typ <= val_max)))
);
CREATE INDEX av_node_ix      ON attribute_value (node_id);
CREATE INDEX av_attr_ix      ON attribute_value (attribute_id, status);
CREATE INDEX av_num_ix       ON attribute_value (attribute_id, num_value) WHERE num_value IS NOT NULL;
CREATE INDEX av_range_ix     ON attribute_value (attribute_id, val_min, val_max);
CREATE INDEX av_chunk_ix     ON attribute_value (source_chunk_id);
-- One live value per product+key+condition; conflicts are modelled explicitly
-- via status='conflicting' rather than by allowing silent duplicates.
CREATE UNIQUE INDEX av_unique_live_uix
    ON attribute_value (node_id, attribute_id, COALESCE(condition_id, -1))
    WHERE status IN ('present','not_specified','absent');

-- kind = 'list' payload (signal_type, interfaces, compliance_standards, ...)
CREATE TABLE attribute_value_item (
    value_id   BIGINT NOT NULL REFERENCES attribute_value(value_id) ON DELETE CASCADE,
    ordinal    SMALLINT NOT NULL,
    item_code  TEXT NOT NULL,        -- normalised vocabulary term ('rs485','dsub_78')
    item_text  TEXT,                 -- as written ('EN 61326-1', 'RoHS 3')
    PRIMARY KEY (value_id, ordinal)
);
CREATE INDEX avi_code_ix ON attribute_value_item (item_code);

-- R2.3 payload: accuracy/resolution hang off the parent value.
CREATE TABLE attribute_value_subfield (
    value_id      BIGINT  NOT NULL REFERENCES attribute_value(value_id) ON DELETE CASCADE,
    subfield_id   INTEGER NOT NULL REFERENCES attribute_subfield(subfield_id) ON DELETE RESTRICT,
    num_value     NUMERIC,
    text_value    TEXT,
    unit_observed TEXT,
    raw_text      TEXT,
    PRIMARY KEY (value_id, subfield_id)
);

-- Enforces payload shape against the dictionary's declared kind. Done as a
-- trigger because a CHECK cannot reach attribute.kind across the FK.
CREATE OR REPLACE FUNCTION attribute_value_kind_guard() RETURNS trigger AS $$
DECLARE k attribute_kind; has_cond BOOLEAN;
BEGIN
    SELECT kind INTO k FROM attribute WHERE attribute_id = NEW.attribute_id;
    IF NEW.status NOT IN ('present','conflicting','superseded') THEN
        RETURN NEW;                             -- no payload to validate
    END IF;
    IF k = 'single' AND NUM_NONNULLS(NEW.num_value, NEW.text_value) <> 1 THEN
        RAISE EXCEPTION 'kind=single needs exactly one of num_value/text_value';
    ELSIF k = 'boolean' AND NEW.bool_value IS NULL THEN
        RAISE EXCEPTION 'kind=boolean needs bool_value';
    ELSIF k = 'range' AND NEW.val_min IS NULL AND NEW.val_max IS NULL THEN
        RAISE EXCEPTION 'kind=range needs val_min and/or val_max';
    ELSIF k = 'min_typ_max'
          AND NUM_NONNULLS(NEW.val_min, NEW.val_typ, NEW.val_max) = 0 THEN
        RAISE EXCEPTION 'kind=min_typ_max needs at least one of min/typ/max';
    END IF;
    -- kind='conditional' must name its qualifier (R1.2/R1.3 teeth at insert time)
    IF k = 'conditional' AND NEW.condition_id IS NULL THEN
        RAISE EXCEPTION 'kind=conditional requires a condition_id';
    END IF;
    -- a condition must belong to this attribute
    IF NEW.condition_id IS NOT NULL THEN
        SELECT TRUE INTO has_cond FROM attribute_condition
         WHERE condition_id = NEW.condition_id AND attribute_id = NEW.attribute_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'condition_id % does not belong to attribute %',
                NEW.condition_id, NEW.attribute_id;
        END IF;
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE TRIGGER attribute_value_kind_guard_trg
    BEFORE INSERT OR UPDATE ON attribute_value
    FOR EACH ROW EXECUTE FUNCTION attribute_value_kind_guard();

-- Structural placement guard: a value hangs off a LEAF PRODUCT. It does NOT
-- police applicability.
--
-- KARAR-051 (2026-08-03): `applies_to` is NOT a hard limit. It is guidance
-- handed to the extracting model in the prompt; if the model writes a value
-- outside a block's scope, that value is ACCEPTED, not rejected.
--
-- Why the earlier hard guard was wrong: applicability is observation-derived
-- and deliberately narrow, but the corpus contradicts that narrowness -- a
-- USB converter does have logic levels, a PXIe module does have a slot width.
-- Measured 2026-08-03: the out-of-scope values the model wrote had CORRECT
-- raw text (PN1204's datasheet really does say 'Output Voltage High 2.4V').
-- Rejecting them would lose real, document-backed facts -- either by breaking
-- the pipeline or by dropping them silently.
--
-- The boundary stays VISIBLE rather than enforcing: out-of-scope values are
-- reported by `v_out_of_scope_value` and read as evidence that the dictionary
-- needs widening. Only the bootstrap's own labelling stays locked: an
-- in-scope pair may not be born 'not_applicable' (that would be a bootstrap
-- bug, not data).
CREATE OR REPLACE FUNCTION attribute_value_scope_guard() RETURNS trigger AS $$
DECLARE fam TEXT; sub TEXT; ntype product_node_type; scoped BOOLEAN; applies BOOLEAN;
BEGIN
    SELECT family, subfamily, node_type INTO fam, sub, ntype
      FROM product_node WHERE node_id = NEW.node_id;
    IF ntype <> 'product' THEN
        RAISE EXCEPTION 'attribute_value only attaches to leaf products, got % node %',
            ntype, NEW.node_id;
    END IF;
    IF NEW.status = 'not_applicable' THEN
        SELECT EXISTS (SELECT 1 FROM attribute_applicability
                        WHERE attribute_id = NEW.attribute_id) INTO scoped;
        SELECT EXISTS (
            SELECT 1 FROM attribute_applicability a
             WHERE a.attribute_id = NEW.attribute_id
               AND a.family = fam
               AND (a.subfamily = '' OR a.subfamily = sub)) INTO applies;
        IF NOT scoped OR applies THEN
            RAISE EXCEPTION
                'status=not_applicable but attribute % DOES apply to family %',
                NEW.attribute_id, fam;
        END IF;
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE TRIGGER attribute_value_scope_guard_trg
    BEFORE INSERT OR UPDATE ON attribute_value
    FOR EACH ROW EXECUTE FUNCTION attribute_value_scope_guard();


-- =====================================================================
-- 5. RETRIEVAL HELPERS
-- =====================================================================

-- Flattened answer surface: one row per stated fact with its citation.
CREATE VIEW v_product_spec AS
SELECT p.product_code,
       p.display_name,
       pn.family,
       pn.subfamily,
       ab.name  AS block,
       a.key,
       ac.code  AS condition,
       a.kind,
       a.unit,
       av.status,
       av.num_value, av.text_value, av.val_min, av.val_typ, av.val_max, av.bool_value,
       av.unit_observed,
       av.raw_text,
       d.file_name AS source_document,
       d.doc_type,
       av.source_chunk_id
  FROM attribute_value av
  JOIN product_node pn ON pn.node_id = av.node_id
  JOIN product      p  ON p.node_id  = pn.node_id
  JOIN attribute    a  ON a.attribute_id = av.attribute_id
  JOIN attribute_block ab ON ab.block_id = a.block_id
  LEFT JOIN attribute_condition ac ON ac.condition_id = av.condition_id
  LEFT JOIN document d ON d.doc_id = av.source_doc_id
 WHERE av.status IN ('present','absent');

-- Which questions a product still cannot answer — drives extraction backlog
-- and lets the chatbot say "not stated in the datasheet" instead of hallucinating.
CREATE VIEW v_coverage_gap AS
SELECT pn.node_id, p.product_code, pn.family, a.key, a.question
  FROM product_node pn
  JOIN product p ON p.node_id = pn.node_id
  JOIN attribute a ON a.is_active
  JOIN attribute_block ab ON ab.block_id = a.block_id
 WHERE pn.node_type = 'product'
   AND (NOT EXISTS (SELECT 1 FROM attribute_applicability aa WHERE aa.attribute_id = a.attribute_id)
        OR EXISTS (SELECT 1 FROM attribute_applicability aa
                    WHERE aa.attribute_id = a.attribute_id
                      AND aa.family = pn.family
                      AND (aa.subfamily = '' OR aa.subfamily = pn.subfamily)))
   AND NOT EXISTS (SELECT 1 FROM attribute_value av
                    WHERE av.node_id = pn.node_id AND av.attribute_id = a.attribute_id);

-- Documents whose chunks no longer match the file on disk.
CREATE VIEW v_stale_document AS
SELECT d.doc_id, d.file_name, d.doc_type, dp.chunked_at, dp.last_parsed
  FROM document d JOIN document_pipeline dp ON dp.doc_id = d.doc_id
 WHERE dp.is_stale OR dp.chunk_status <> 'SUCCESS';
