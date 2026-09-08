"""N-06 (I-08) kabul testleri: `forget_deleted_source` bir `doc_id`nin TUM
turevlerini (parse ciktisi, chunk kaydi, Qdrant noktalari, spec kanitlari)
temizler ve bunu IDEMPOTENT yapar. Gercek Qdrant baglantisi YOK -- sahte bir
`replace_scope` kayitcisiyla test edilir (Grup G'nin fake Redis/Queue
desenininAYNISI)."""
from __future__ import annotations

import json
import sqlite3

from medrag.pipeline.forget_deleted_source import forget_deleted_source


class FakeVectorStore:
    """Gercek Qdrant istemcisi yerine -- `replace_scope` cagrilarini kaydeder,
    aga hic cikmaz."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    def replace_scope(self, scope_id, points):
        self.calls.append((scope_id, list(points)))


def _specs_db():
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


def _insert_spec_value(con, value_id, doc_id, chunk_sha):
    evidence = [{"doc_id": doc_id, "file_name": "f.pdf", "chunk_sha256": chunk_sha}]
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, status, "
        "source_doc_id, source_chunk_id, evidence) VALUES (?, 'DE1', 'core', 'attr', "
        "'present', ?, ?, ?)",
        (value_id, doc_id, chunk_sha, json.dumps(evidence, ensure_ascii=False)),
    )
    con.commit()


def _write_all_chunks(path, **documents):
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")


def test_removes_parse_output_folder_by_doc_type(tmp_path):
    parsed_dir = tmp_path / "parsed"
    grup = parsed_dir / "DATASHEET_doc-1"
    grup.mkdir(parents=True)
    (grup / "doc-1.json").write_text("{}", encoding="utf-8")

    result = forget_deleted_source(
        doc_id="doc-1", doc_type="DATASHEET",
        parsed_output_dir=parsed_dir,
        all_chunks_path=tmp_path / "yok.json",
        vector_store=FakeVectorStore(),
        specs_con=_specs_db(),
    )

    assert not grup.exists(), "parse cikti klasoru SILINMELI"
    assert result.parse_dirs_removed == [str(grup)]


def test_removes_parse_output_folder_without_doc_type_via_glob(tmp_path):
    """doc_type bilinmiyorsa (None) `_<doc_id>` sonekiyle klasor bulunmali --
    tarama sirasinda doc_type'in degismis/kayip olmasina karsi guvenlik agi."""
    parsed_dir = tmp_path / "parsed"
    grup = parsed_dir / "BROCHURE_doc-2"
    grup.mkdir(parents=True)

    result = forget_deleted_source(
        doc_id="doc-2", doc_type=None,
        parsed_output_dir=parsed_dir,
        all_chunks_path=tmp_path / "yok.json",
        vector_store=FakeVectorStore(),
        specs_con=_specs_db(),
    )

    assert not grup.exists()
    assert result.parse_dirs_removed == [str(grup)]


def test_missing_parse_output_folder_is_a_noop(tmp_path):
    """Klasor hic yoksa (daha once silinmis ya da hic parse edilmemis)
    hata FIRLAMAMALI -- idempotent temizligin temeli."""
    result = forget_deleted_source(
        doc_id="doc-yok", doc_type="DATASHEET",
        parsed_output_dir=tmp_path / "parsed",
        all_chunks_path=tmp_path / "yok.json",
        vector_store=FakeVectorStore(),
        specs_con=_specs_db(),
    )
    assert result.parse_dirs_removed == []


def test_removes_chunk_entry_from_all_chunks_json(tmp_path):
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "text": "x"}]},
        "doc-2": {"nodes": [{"node_id": "doc-2::c0", "text": "y"}]},
    })

    result = forget_deleted_source(
        doc_id="doc-1", doc_type=None,
        parsed_output_dir=tmp_path / "parsed",
        all_chunks_path=all_chunks,
        vector_store=FakeVectorStore(),
        specs_con=_specs_db(),
    )

    assert result.chunk_entry_removed is True
    kalan = json.loads(all_chunks.read_text(encoding="utf-8"))["documents"]
    assert "doc-1" not in kalan
    assert "doc-2" in kalan, "baska dokumanin chunk'ina DOKUNULMAMALI"


def test_missing_all_chunks_file_is_a_noop(tmp_path):
    result = forget_deleted_source(
        doc_id="doc-1", doc_type=None,
        parsed_output_dir=tmp_path / "parsed",
        all_chunks_path=tmp_path / "hic-olusmadi.json",
        vector_store=FakeVectorStore(),
        specs_con=_specs_db(),
    )
    assert result.chunk_entry_removed is False


def test_calls_qdrant_replace_scope_with_doc_id_and_empty_list(tmp_path):
    """KARAR-004: bos liste = kapsamin TUM eski noktalarini sil, hic upsert
    etme -- gercek Qdrant baglantisi YOK, sahte kayitci cagriyi dogrular."""
    store = FakeVectorStore()

    forget_deleted_source(
        doc_id="doc-1", doc_type=None,
        parsed_output_dir=tmp_path / "parsed",
        all_chunks_path=tmp_path / "yok.json",
        vector_store=store,
        specs_con=_specs_db(),
    )

    assert store.calls == [("doc-1", [])]


def test_removes_spec_evidence_via_forget_source(tmp_path):
    """Spec tarafi O-11'in `forget_source`una DELEGE edilir -- ayri bir
    silme mantigi YOK. Tek kanitli satir dusuyor."""
    con = _specs_db()
    _insert_spec_value(con, 1, "doc-1", "sha256:" + "a" * 64)

    result = forget_deleted_source(
        doc_id="doc-1", doc_type=None,
        parsed_output_dir=tmp_path / "parsed",
        all_chunks_path=tmp_path / "yok.json",
        vector_store=FakeVectorStore(),
        specs_con=con,
    )

    assert result.spec_evidence["rows_deleted"] == 1
    assert con.execute("SELECT COUNT(*) FROM spec_value").fetchone()[0] == 0


def test_full_cleanup_is_idempotent_on_second_run(tmp_path):
    """I-08 kabulu: silme YARIDA KESILIP TEKRAR CALISTIRILABILIR -- ikinci
    cagri hicbir seye dokunmadan AYNI (bos) sonucu uretir, hata FIRLATMAZ."""
    parsed_dir = tmp_path / "parsed"
    grup = parsed_dir / "DATASHEET_doc-1"
    grup.mkdir(parents=True)
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{"doc-1": {"nodes": []}})
    con = _specs_db()
    _insert_spec_value(con, 1, "doc-1", "sha256:" + "a" * 64)
    store = FakeVectorStore()

    birinci = forget_deleted_source(
        doc_id="doc-1", doc_type="DATASHEET",
        parsed_output_dir=parsed_dir, all_chunks_path=all_chunks,
        vector_store=store, specs_con=con,
    )
    assert birinci.parse_dirs_removed == [str(grup)]
    assert birinci.chunk_entry_removed is True
    assert birinci.spec_evidence["rows_deleted"] == 1

    # Yarida kesilip TEKRAR calistirilan gece kosusu ayni cagriyi tekrarlar.
    ikinci = forget_deleted_source(
        doc_id="doc-1", doc_type="DATASHEET",
        parsed_output_dir=parsed_dir, all_chunks_path=all_chunks,
        vector_store=store, specs_con=con,
    )

    assert ikinci.parse_dirs_removed == [], "klasor zaten yok -- ikinci kez SILINECEK bir sey yok"
    assert ikinci.chunk_entry_removed is False, "kayit zaten yok"
    assert ikinci.spec_evidence["rows_deleted"] == 0
    assert ikinci.spec_evidence["rows_touched"] == 0
    # Qdrant her seferinde ayni sekilde cagrilir -- delete-then-upsert zaten
    # kendi basina idempotent (KARAR-004), bu yuzden ikinci cagri da olmali.
    assert store.calls == [("doc-1", []), ("doc-1", [])]
    assert not grup.exists()
    assert con.execute("SELECT COUNT(*) FROM spec_value").fetchone()[0] == 0
