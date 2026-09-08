CREATE VIEW v_coverage_gap AS
SELECT product_code, family, block, key
  FROM spec_value
 WHERE status = 'not_specified';

CREATE VIEW v_document_product AS
SELECT d.doc_id,
       je.value AS product_code,
       d.doc_type,
       d.file_name,
       d.is_active,
       d.trust_rank
  FROM document d, json_each(d.product_codes) je;

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

CREATE TABLE attribute (
    block         TEXT NOT NULL,                     -- 'core','analog_io',...
    key           TEXT NOT NULL,                      -- 'operating_temperature'
    kind          TEXT NOT NULL
                  CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    unit          TEXT,                               -- kanonik birim
    enum_values   TEXT,                                -- JSON dizi ya da NULL
    accepts_units TEXT,                                -- JSON dizi ya da NULL (R3.5)
    labels        TEXT NOT NULL DEFAULT '[]',          -- JSON dizi: korpusta gozlenen etiketler
    conditions    TEXT NOT NULL DEFAULT '[]',          -- JSON dizi: KAPALI OLMAYAN referans liste
    subfields     TEXT NOT NULL DEFAULT '[]',          -- JSON dizi: [{"name":...,"unit":...}]
    applies_to    TEXT NOT NULL DEFAULT '[]',          -- JSON dizi: aile/alt-aile kapsam adlari
    is_core       INTEGER NOT NULL DEFAULT 0 CHECK (is_core IN (0,1)),
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
    PRIMARY KEY (block, key)
);

CREATE TABLE document (
    doc_id         TEXT PRIMARY KEY,
    file_name      TEXT NOT NULL,
    extension      TEXT NOT NULL,
    rel_path       TEXT NOT NULL,
    file_path      TEXT,                             -- ASLA son kullaniciya/modele gosterilmez
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
    -- JSON dizi: bu dosyanin (dogrudan + miras/fan-out ile) kapsadigi HER
    -- yaprak urunun product_code'u -- KARAR-014 fan-out'unun build-zamanli
    -- ONCEDEN-HESAPLANMIS hali. `v_document_product` view'i bunu unnest eder.
    product_codes  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE product (
    product_code      TEXT PRIMARY KEY,                -- 'PN1309'. KARAR-013'un
                                                        -- node_id'si olan ama product_code'u
                                                        -- OLMAYAN tek urunu bu semada
                                                        -- TEMSIL EDILEMEZ -- build script
                                                        -- bunu ATLAR ve raporlar (bilinen,
                                                        -- kabul edilen kayip: 250 -> 249).
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

CREATE TABLE spec_value (
    value_id      INTEGER PRIMARY KEY,
    product_code  TEXT    NOT NULL REFERENCES product(product_code) ON DELETE CASCADE,
    family        TEXT,                                -- denormalize (product.family)
    subfamily     TEXT,                                -- denormalize (product.subfamily)
    block         TEXT NOT NULL,
    key           TEXT NOT NULL,
    kind          TEXT NOT NULL
                  CHECK (kind IN ('single','min_typ_max','range','list','conditional','boolean')),
    unit          TEXT,                                -- denormalize (attribute.unit, kanonik)
    condition     TEXT,                                -- duz TEXT (kapali sozluk DEGIL)
    status        TEXT NOT NULL CHECK (status IN (
                      'present','not_specified','absent','not_applicable',
                      'conflicting','superseded')),

    num_value     NUMERIC,
    text_value    TEXT,
    val_min       NUMERIC,
    val_typ       NUMERIC,
    val_max       NUMERIC,
    bool_value    INTEGER CHECK (bool_value IS NULL OR bool_value IN (0,1)),
    items         TEXT,                                -- JSON dizi (kind=list): [{"ordinal":1,"item_code":...,"item_text":...}]
    subfields     TEXT,                                -- JSON dizi: [{"name":...,"num_value":...,"text_value":...,"unit_observed":...,"raw_text":...}]

    unit_observed TEXT,
    raw_text      TEXT,

    -- provenance -- BIRINCIL kaynak (en yuksek trust_rank) dogrudan
    -- kolonlarda (join'siz okuma icin file_name DENORMALIZE); ek/destekleyici
    -- kaynaklar `evidence` JSON dizisinde.
    source_doc_id     TEXT REFERENCES document(doc_id) ON DELETE SET NULL,
    source_file_name  TEXT,                             -- denormalize (document.file_name)
    source_chunk_id   TEXT,
    evidence          TEXT NOT NULL DEFAULT '[]',        -- JSON dizi: [{"doc_id":...,"file_name":...,"chunk_sha256":...}]

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

CREATE INDEX attribute_key_ix ON attribute (key);

CREATE INDEX document_hash_ix ON document (content_hash);

CREATE INDEX document_type_ix ON document (doc_type) WHERE is_active = 1;

CREATE INDEX product_family_ix ON product (family, subfamily);

CREATE INDEX sv_key_ix     ON spec_value (key, status);

CREATE INDEX sv_num_ix     ON spec_value (key, num_value) WHERE num_value IS NOT NULL;

CREATE INDEX sv_product_ix ON spec_value (product_code);

CREATE INDEX sv_range_ix   ON spec_value (key, val_min, val_max);

CREATE UNIQUE INDEX sv_unique_live_uix
    ON spec_value (product_code, block, key, COALESCE(condition, ''))
    WHERE status IN ('present','not_specified','absent','not_applicable');

