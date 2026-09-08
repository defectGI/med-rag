"""O-11 (I-17, K-70/K-78): "kaynagi unut" ortak silme fonksiyonunun kabul
testleri. Gercek Ollama/LLM cagrisi YOK -- yalniz gercek bir sqlite semasina
(`:memory:`, spec_value'nin gercek sutunlariyla) karsi calisir."""
from __future__ import annotations

import json
import sqlite3

import pytest

from medrag.pipeline.facts.forget_source import forget_source

_SHA_SURVIVOR = "sha256:" + "a" * 64
_SHA_DOC1_ONLY = "sha256:" + "b" * 64
_SHA_DOC2_CORROB = "sha256:" + "c" * 64
_SHA_GHOST = "sha256:" + "d" * 64
_LEGACY_REF = "11111111-1111-1111-1111-111111111111::c0"


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE spec_value (
        value_id INTEGER PRIMARY KEY, product_code TEXT, family TEXT, subfamily TEXT,
        block TEXT, key TEXT, kind TEXT, unit TEXT, condition TEXT, status TEXT,
        num_value REAL, text_value TEXT, val_min REAL, val_typ REAL, val_max REAL,
        bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
        source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT, evidence TEXT NOT NULL DEFAULT '[]',
        extractor TEXT, extractor_version TEXT, extracted_at TEXT)""")
    con.commit()
    return con


def _insert(con, value_id, key, status, source_doc_id, source_chunk_id, evidence, condition=None):
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, condition, status, "
        "source_doc_id, source_chunk_id, evidence) VALUES (?, 'DE1', 'core', ?, ?, ?, ?, ?, ?)",
        (value_id, key, condition, status, source_doc_id, source_chunk_id, json.dumps(evidence, ensure_ascii=False)),
    )
    con.commit()


def test_removing_only_source_drops_the_row_and_evidence_count_shrinks():
    """I-17 kabul: bir kaynak silinir -- ondan gelen kanit gitmis, tek
    kanidi bu kaynak olan satir tamamen DUSMUS."""
    con = _db()
    _insert(
        con, 1, "lonely_attr", "present", "doc1", _SHA_DOC1_ONLY,
        [{"doc_id": "doc1", "file_name": "f1.pdf", "chunk_sha256": _SHA_DOC1_ONLY}],
    )
    before_count = con.execute("SELECT COUNT(*) FROM spec_value").fetchone()[0]

    stats = forget_source(con, doc_ids=["doc1"])

    after_count = con.execute("SELECT COUNT(*) FROM spec_value").fetchone()[0]
    assert after_count == before_count - 1
    assert stats["rows_deleted"] == 1
    assert stats["rows_updated"] == 0
    assert stats["evidence_entries_removed"] == 1
    assert con.execute("SELECT COUNT(*) FROM spec_value WHERE value_id=1").fetchone()[0] == 0


def test_row_with_other_evidence_survives_and_evidence_count_drops_by_one():
    """I-17 kabul: baska kaynagi olan satir DURUYOR, kanit sayisi BIR azalmis,
    birincil kaynak alanlari kalan kanita hizalanmis."""
    con = _db()
    _insert(
        con, 2, "corroborated_attr", "present", "doc1", _SHA_SURVIVOR,
        [
            {"doc_id": "doc1", "file_name": "f1.pdf", "chunk_sha256": _SHA_SURVIVOR},
            {"doc_id": "doc2", "file_name": "f2.pdf", "chunk_sha256": _SHA_DOC2_CORROB},
        ],
    )

    stats = forget_source(con, doc_ids=["doc1"])

    row = con.execute(
        "SELECT status, source_doc_id, source_chunk_id, evidence FROM spec_value WHERE value_id=2"
    ).fetchone()
    assert row is not None, "baska kaniti olan satir dusmemeli"
    status, source_doc_id, source_chunk_id, evidence_json = row
    assert status == "present"
    evidence = json.loads(evidence_json)
    assert len(evidence) == 1
    assert evidence[0]["doc_id"] == "doc2"
    assert source_doc_id == "doc2"
    assert source_chunk_id == _SHA_DOC2_CORROB
    assert stats["rows_updated"] == 1
    assert stats["rows_deleted"] == 0
    assert stats["evidence_entries_removed"] == 1


def test_conflicting_row_left_without_evidence_is_also_dropped():
    """Dikkat notu: kaynagi gidince cakisma da biter -- `status='conflicting'`
    satir `sv_unique_live_uix`in disinda sessizce BIRIKMEMELI."""
    con = _db()
    _insert(
        con, 3, "conflicting_attr", "conflicting", "doc2", _SHA_DOC2_CORROB,
        [{"doc_id": "doc2", "file_name": "f2.pdf", "chunk_sha256": _SHA_DOC2_CORROB}],
    )

    stats = forget_source(con, doc_ids=["doc2"])

    assert con.execute("SELECT COUNT(*) FROM spec_value WHERE value_id=3").fetchone()[0] == 0
    assert stats["rows_deleted"] == 1


def test_untouched_rows_are_left_completely_alone():
    con = _db()
    _insert(
        con, 4, "unrelated_attr", "present", "doc3", "sha256:" + "e" * 64,
        [{"doc_id": "doc3", "file_name": "f3.pdf", "chunk_sha256": "sha256:" + "e" * 64}],
    )

    stats = forget_source(con, doc_ids=["doc1"])

    assert stats["rows_touched"] == 0
    row = con.execute("SELECT evidence FROM spec_value WHERE value_id=4").fetchone()
    assert json.loads(row[0]) == [{"doc_id": "doc3", "file_name": "f3.pdf", "chunk_sha256": "sha256:" + "e" * 64}]


def test_matches_by_content_sha256_not_just_doc_id():
    """O-11a'nin urettigi hayalet girdileri temizlemek icin de kullanilir --
    ayri kod yolu YOK, ayni fonksiyon content_sha256 listesiyle cagrilir."""
    con = _db()
    _insert(
        con, 5, "ghost_attr", "present", "doc1", _SHA_GHOST,
        [
            {"doc_id": "doc1", "file_name": "f1.pdf", "chunk_sha256": _SHA_GHOST},
            {"doc_id": "doc2", "file_name": "f2.pdf", "chunk_sha256": _SHA_DOC2_CORROB},
        ],
    )

    stats = forget_source(con, content_sha256s=[_SHA_GHOST])

    evidence = json.loads(con.execute("SELECT evidence FROM spec_value WHERE value_id=5").fetchone()[0])
    assert len(evidence) == 1
    assert evidence[0]["chunk_sha256"] == _SHA_DOC2_CORROB
    assert stats["evidence_entries_removed"] == 1


def test_bare_hex_content_sha256_is_normalized_to_sha256_prefixed_form():
    con = _db()
    bare_hex = "d" * 64
    _insert(con, 6, "bare_hex_attr", "present", "doc1", _SHA_GHOST, [
        {"doc_id": "doc1", "file_name": "f1.pdf", "chunk_sha256": _SHA_GHOST},
    ])

    stats = forget_source(con, content_sha256s=[bare_hex])

    assert stats["rows_deleted"] == 1


def test_legacy_uuid_colon_colon_c_reference_matches_literally():
    """K-70 gecis donemi: eski `<uuid>::cN` bicimli kanitlar da (henuz migre
    edilmemis) literal esitlikle silinebilmeli."""
    con = _db()
    _insert(con, 7, "legacy_attr", "present", "docL", _LEGACY_REF, [
        {"doc_id": "docL", "file_name": "legacy.pdf", "chunk_sha256": _LEGACY_REF},
    ])

    stats = forget_source(con, content_sha256s=[_LEGACY_REF])

    assert stats["rows_deleted"] == 1


def test_requires_at_least_one_selector():
    con = _db()
    with pytest.raises(ValueError):
        forget_source(con)


def test_commit_false_lets_caller_control_transaction():
    con = _db()
    _insert(con, 8, "attr", "present", "doc1", _SHA_DOC1_ONLY, [
        {"doc_id": "doc1", "file_name": "f1.pdf", "chunk_sha256": _SHA_DOC1_ONLY},
    ])

    forget_source(con, doc_ids=["doc1"], commit=False)
    # Ayni baglanti uzerinden hala gorunur (commit edilmemis olsa da ayni
    # connection kendi degisikligini gorur) -- asil kontrol: caller kendi
    # commit'ini cagirabiliyor, hata firlamiyor.
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM spec_value WHERE value_id=8").fetchone()[0] == 0
