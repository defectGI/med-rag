"""O-08 (I-18): `db_integrity.verify_db_integrity` kapisini kilitler.

`db_integrity.py` bilinclt olarak `load_to_db.py`den AYRI bir modul (bkz.
`db_integrity.py` docstring'i) -- `load_to_db.py` su an import-time'da
`FileNotFoundError` veriyor (K-71, prompts/*.md pakete tasinmadi), bu yuzden
bu testler O-08'in mantigini o blokeden BAGIMSIZ, tam calisir sekilde
dogrular (skip YOK).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from medrag.pipeline.facts import db_integrity

# Gercek `facts/db/schema.yaml`in kucultulmus bir aynasi -- testin ihtiyaci
# kadar tablo/sutun. `document`de KASITLI OLARAK `trust_rank` VAR (gercek
# schema.yaml'daki gibi) ki O-08'in bilinen istisnasi (document.trust_rank
# canli DB'de fiziksel kolon DEGIL) burada da sinansin.
_FAKE_SCHEMA_YAML = """
name: acme_spec_store
tables:
- name: spec_value
  columns:
  - name: value_id
  - name: product_code
  - name: block
  - name: key
  - name: status
  - name: evidence
- name: attribute
  columns:
  - name: block
  - name: key
  - name: kind
- name: product
  columns:
  - name: product_code
  - name: family
- name: document
  columns:
  - name: doc_id
  - name: file_name
  - name: trust_rank
- name: v_document_product
  columns:
  - name: doc_id
  - name: trust_rank
"""


def _write_schema_yaml(tmp_path) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(_FAKE_SCHEMA_YAML, encoding="utf-8")
    return path


def _make_valid_db(path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, PRIMARY KEY (block, key));
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY,
            product_code TEXT, block TEXT, key TEXT, condition TEXT,
            status TEXT, evidence TEXT
        );
        CREATE UNIQUE INDEX sv_unique_live_uix
            ON spec_value (product_code, block, key, COALESCE(condition, ''))
            WHERE status IN ('present','not_specified','absent','not_applicable');
        """
    )
    con.commit()
    return con


def test_valid_schema_and_no_trust_rank_column_passes(tmp_path):
    """Canli specs.db'nin gercek durumu: `document`de `trust_rank` FIZIKSEL
    kolon YOK ama schema.yaml onu listeliyor -- bu, O-08'in bilinen tek
    istisnasi (bkz. db_integrity.py docstring), false-positive fail ETMEMELI."""
    db_path = tmp_path / "specs.db"
    con = _make_valid_db(db_path)
    schema_path = _write_schema_yaml(tmp_path)

    db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)  # raise etmemeli


def test_missing_column_rejects_and_names_it(tmp_path):
    """Sema kasten bozulur: `spec_value.evidence` sutunu DUSURULUR."""
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, PRIMARY KEY (block, key));
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY,
            product_code TEXT, block TEXT, key TEXT, condition TEXT, status TEXT
        );
        CREATE UNIQUE INDEX sv_unique_live_uix
            ON spec_value (product_code, block, key, COALESCE(condition, ''))
            WHERE status IN ('present','not_specified','absent','not_applicable');
        """
    )
    con.commit()
    schema_path = _write_schema_yaml(tmp_path)

    with pytest.raises(db_integrity.SchemaIntegrityError) as exc_info:
        db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)
    assert "spec_value" in str(exc_info.value)
    assert "evidence" in str(exc_info.value)


def test_missing_table_rejects_and_names_it(tmp_path):
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, PRIMARY KEY (block, key));
        """
    )
    con.commit()
    schema_path = _write_schema_yaml(tmp_path)

    with pytest.raises(db_integrity.SchemaIntegrityError) as exc_info:
        db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)
    assert "spec_value" in str(exc_info.value)


def test_missing_unique_index_rejects(tmp_path):
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, PRIMARY KEY (block, key));
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY,
            product_code TEXT, block TEXT, key TEXT, condition TEXT,
            status TEXT, evidence TEXT
        );
        """
    )
    con.commit()
    schema_path = _write_schema_yaml(tmp_path)

    with pytest.raises(db_integrity.SchemaIntegrityError) as exc_info:
        db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)
    assert "sv_unique_live_uix" in str(exc_info.value)


def test_non_unique_index_with_the_right_name_rejects(tmp_path):
    """Isim dogru ama kisit UNIQUE DEGIL -- sessizce kabul EDILMEMELI."""
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, PRIMARY KEY (block, key));
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY,
            product_code TEXT, block TEXT, key TEXT, condition TEXT,
            status TEXT, evidence TEXT
        );
        CREATE INDEX sv_unique_live_uix ON spec_value (product_code, block, key);
        """
    )
    con.commit()
    schema_path = _write_schema_yaml(tmp_path)

    with pytest.raises(db_integrity.SchemaIntegrityError) as exc_info:
        db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)
    assert "UNIQUE" in str(exc_info.value)


def test_corrupt_database_file_rejects(tmp_path):
    """Gercek dosya-seviyesi bozulma: bir sqlite dosyasi degil, cop bayt."""
    db_path = tmp_path / "specs.db"
    db_path.write_bytes(b"bu bir sqlite dosyasi degil, kasten bozuk")
    con = sqlite3.connect(db_path)
    schema_path = _write_schema_yaml(tmp_path)

    with pytest.raises(db_integrity.SchemaIntegrityError):
        db_integrity.verify_db_integrity(con, schema_yaml_path=schema_path)


def test_real_schema_yaml_and_real_specs_db_shape_pass():
    """Uretimdeki `facts/db/schema.yaml`in GERCEK icerigiyle, gercek
    specs.db'nin sema SEKLINI (fixture olarak, canli veriye DOKUNMADAN)
    tasiyan bir DB olustur ve gecsin -- yalniz mock semayla degil, gercek
    dosyayla da dogrulanmis olsun."""
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE product (
            product_code TEXT PRIMARY KEY, display_name TEXT, family TEXT,
            subfamily TEXT, subfamily_2 TEXT, acme_code TEXT, list_price NUMERIC,
            price_on_request INTEGER DEFAULT 0, valid_until TEXT,
            price_break_10_99 NUMERIC, price_break_100_499 NUMERIC,
            price_break_500_999 NUMERIC, is_variant INTEGER DEFAULT 0, variant_base TEXT
        );
        CREATE TABLE document (
            doc_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, extension TEXT NOT NULL,
            rel_path TEXT NOT NULL, file_path TEXT, content_hash TEXT NOT NULL,
            size_bytes INTEGER, last_modified_time TEXT, doc_type TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1, product_codes TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE attribute (
            block TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL, unit TEXT,
            enum_values TEXT, accepts_units TEXT, labels TEXT NOT NULL DEFAULT '[]',
            conditions TEXT NOT NULL DEFAULT '[]', subfields TEXT NOT NULL DEFAULT '[]',
            applies_to TEXT NOT NULL DEFAULT '[]', is_core INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (block, key)
        );
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY, product_code TEXT NOT NULL, family TEXT,
            subfamily TEXT, block TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
            unit TEXT, condition TEXT, status TEXT NOT NULL, num_value NUMERIC,
            text_value TEXT, val_min NUMERIC, val_typ NUMERIC, val_max NUMERIC,
            bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT,
            raw_text TEXT, source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT,
            evidence TEXT NOT NULL DEFAULT '[]', extractor TEXT, extractor_version TEXT,
            confidence REAL, extracted_at TEXT
        );
        CREATE UNIQUE INDEX sv_unique_live_uix
            ON spec_value (product_code, block, key, COALESCE(condition, ''))
            WHERE status IN ('present','not_specified','absent','not_applicable');
        """
    )
    con.commit()

    db_integrity.verify_db_integrity(con)  # varsayilan schema_yaml_path = gercek dosya
