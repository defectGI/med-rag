"""`chunk_store.py` + `sql_evidence.py` -- SQL row -> evidence chunk text.

Everything runs on `tmp_path` with its own mini `specs.db` + its own mini
chunk corpus -- the real `facts/db/specs.db` and `chunker/storage/` are NEVER
touched.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from medrag.api.chunk_store import ChunkText, clear_cache, load_chunk_index
from medrag.api.sql_evidence import attach_sql_evidence, resolve_sql_evidence

_DOC = "d1"
_CHUNK_A = f"{_DOC}::c1"
_CHUNK_B = f"{_DOC}::c9"


@pytest.fixture(autouse=True)
def _temiz_onbellek():
    """The corpus cache must not leak between tests."""
    clear_cache()
    yield
    clear_cache()


def _db(tmp_path, *, evidence_json: str | None = None):
    """A tiny specs.db containing `spec_value` + `document`."""
    path = tmp_path / "specs.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY, product_code TEXT, key TEXT,
            raw_text TEXT, source_chunk_id TEXT, source_doc_id TEXT, evidence TEXT
        );
        CREATE TABLE document (
            doc_id TEXT PRIMARY KEY, file_name TEXT, file_path TEXT
        );
        """
    )
    if evidence_json is None:
        evidence_json = json.dumps(
            [
                {"doc_id": _DOC, "file_name": "DS.pdf", "chunk_sha256": _CHUNK_A},
                {"doc_id": _DOC, "file_name": "DS.pdf", "chunk_sha256": _CHUNK_B},
            ]
        )
    conn.execute(
        "INSERT INTO spec_value VALUES (?,?,?,?,?,?,?)",
        (7, "PN1099", "weight", "2.4 kg", _CHUNK_A, _DOC, evidence_json),
    )
    conn.execute(
        "INSERT INTO document VALUES (?,?,?)", (_DOC, "DS.pdf", r"C:\korpus\DS.pdf")
    )
    conn.commit()
    conn.close()
    return str(path)


def _korpus(tmp_path) -> dict[str, ChunkText]:
    path = tmp_path / "all_chunks.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-07T00:00:00+03:00",
                "documents": {
                    _DOC: {
                        "doc_id": _DOC,
                        "nodes": [
                            {
                                "node_id": _CHUNK_A,
                                "doc_id": _DOC,
                                "text": "Ağırlık 2.4 kg olarak ölçülmüştür.",
                                "page_start": 4,
                                "page_end": 4,
                                "heading_path": ["1. Genel", "1.2. Mekanik"],
                            },
                            {
                                "node_id": _CHUNK_B,
                                "doc_id": _DOC,
                                "text": "Tabloda net ağırlık 2.4 kg.",
                                "page_start": 11,
                                "page_end": 12,
                                "heading_path": [],
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return load_chunk_index(path)


# --- chunk_store ------------------------------------------------------------


def test_korpus_indeksi_node_id_ile_anahtarlanir(tmp_path):
    index = _korpus(tmp_path)
    assert set(index) == {_CHUNK_A, _CHUNK_B}
    assert index[_CHUNK_A].page == "4"
    assert index[_CHUNK_B].page == "11-12"  # different start/end -> a range
    assert index[_CHUNK_A].section == "1. Genel > 1.2. Mekanik"
    assert index[_CHUNK_B].section is None  # empty heading_path -> no badge


def test_korpus_yoksa_bos_indeks_patlamaz(tmp_path):
    """Error policy: chunk text is an ADD-ON; if the corpus is missing the
    panel shows less, but the turn does NOT fail."""
    assert load_chunk_index(tmp_path / "yok.json") == {}


def test_bozuk_korpus_bos_indeks_doner(tmp_path):
    path = tmp_path / "bozuk.json"
    path.write_text("{ bu json değil", encoding="utf-8")
    assert load_chunk_index(path) == {}


# --- sql_evidence -----------------------------------------------------------


def test_value_id_ile_kanit_chunklari_cozulur(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index=index)

    assert [c["chunk_id"] for c in chunks] == [_CHUNK_A, _CHUNK_B]
    ilk = chunks[0]
    assert ilk["text"] == "Ağırlık 2.4 kg olarak ölçülmüştür."
    assert ilk["file_name"] == "DS.pdf"
    # The full path is DELIBERATELY surfaced to the panel.
    assert ilk["file_path"] == r"C:\korpus\DS.pdf"
    assert ilk["page"] == "4"


def test_source_chunk_id_evidence_ile_tekillestirilir(tmp_path):
    """`source_chunk_id` already exists in the `evidence` list too -- it must
    not be shown twice."""
    db, index = _db(tmp_path), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index=index)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_value_id_yoksa_urun_ozellik_ciftine_dusulur(tmp_path):
    """Aggregation/DISTINCT queries: `sql_citation` does NOT add columns; the
    `(product_code, key)` fallback takes over."""
    db, index = _db(tmp_path), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(
        db, [{"product_code": "PN1099", "key": "weight"}], chunk_index=index
    )
    assert chunks and chunks[0]["chunk_id"] == _CHUNK_A


def test_anahtarsiz_satir_bos_liste_doner(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    assert resolve_sql_evidence(db, [{"raw_text": "2.4 kg"}], chunk_index=index) == [[]]


def test_max_chunks_ust_siniri_uygulanir(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index=index, max_chunks=1)
    assert len(chunks) == 1


def test_metin_snippet_len_ile_kirpilir(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index=index, snippet_len=10)
    assert chunks[0]["text"] == "Ağırlık 2." + "…"


def test_korpusta_olmayan_chunk_metinsiz_ama_kimlikli_doner(tmp_path):
    """If specs.db and the corpus have diverged (re-chunked), the panel should
    still show which chunk was looked up -- don't vanish silently."""
    db = _db(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index={})
    assert chunks[0]["chunk_id"] == _CHUNK_A
    assert chunks[0]["text"] is None
    # doc_id is derived from the node_id prefix -> file name/path is STILL found.
    assert chunks[0]["file_path"] == r"C:\korpus\DS.pdf"


def test_bozuk_evidence_json_turu_dusurmez(tmp_path):
    db, index = _db(tmp_path, evidence_json="{bu json değil"), _korpus(tmp_path)
    (chunks,) = resolve_sql_evidence(db, [{"value_id": 7}], chunk_index=index)
    # `source_chunk_id` is still read -- only the JSON side degrades.
    assert [c["chunk_id"] for c in chunks] == [_CHUNK_A]


def test_okunamayan_db_bos_doner(tmp_path):
    assert resolve_sql_evidence(str(tmp_path / "yok.db"), [{"value_id": 7}]) == [[]]


# --- attach_sql_evidence ----------------------------------------------------


class _Sonuc:
    def __init__(self, metadata):
        self.metadata = metadata


def test_attach_yalniz_sql_satirlarini_zenginlestirir(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    panel = [
        {"source_kind": "DOC", "snippet": "serbest metin"},
        {"source_kind": "SQL", "snippet": "key=weight"},
    ]
    results = [_Sonuc({}), _Sonuc({"value_id": 7})]

    out = attach_sql_evidence(panel, results, db, chunk_index=index)

    assert "evidence_chunks" not in out[0]  # [DOC] already carries the text
    assert out[1]["evidence_chunks"][0]["text"].startswith("Ağırlık")
    # The input is not mutated.
    assert "evidence_chunks" not in panel[1]


def test_attach_uzunluk_uyusmazsa_aynen_doner(tmp_path):
    db, index = _db(tmp_path), _korpus(tmp_path)
    panel = [{"source_kind": "SQL"}]
    assert attach_sql_evidence(panel, [], db, chunk_index=index) == panel
