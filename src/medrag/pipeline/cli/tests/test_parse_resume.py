"""N-04 (resume resilience): parse stage.

`run_parse_pipeline.py` ALREADY carries a phase-1 checkpoint
(`<doc_id>.stage1.json`, write-then-replace, see `phase1_one`/`_stage1_path`
docstrings) -- the current code already has this mechanism and uses the
same write-then-replace pattern as chunk/vectorize. This file's job is to
prove with a destructive test that the mechanism ACTUALLY works -- using
the task's suggested method (instead of real timing/subprocess, inject a
testable kill point): we replace `json.dump` with one that raises
`KeyboardInterrupt` in the MIDDLE of the checkpoint write.

No Ollama/VLM/LLM is touched -- input files are `.md` (MarkdownParser, no
images/tables), so `handle_images`/`apply_ocr_checks`/`describe_blocks` are
no-ops with no input.
"""

from __future__ import annotations

import json

import pytest

from medrag.pipeline.cli import run_parse_pipeline as rpp


def _rec(doc_id: str, urunler_dir, doc_type: str = "DATASHEET") -> dict:
    src = urunler_dir / f"{doc_id}.md"
    src.write_text(f"# {doc_id}\n\nmerhaba dunya\n", encoding="utf-8")
    return {
        "identity": {"doc_id": doc_id, "file_name": f"{doc_id}.md"},
        "location": {"rel_path": f"{doc_id}.md"},
        "scan": {"content_hash": f"sha256:{doc_id}"},
        "doc_type": doc_type,
        "is_active": True,
        "parse": {"status": "PENDING", "parsed_from_hash": None, "parser_version": None},
    }


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    urunler_dir = tmp_path / "urunler"
    urunler_dir.mkdir()
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    monkeypatch.setattr(rpp, "BELGELER_DIR", str(urunler_dir), raising=False)
    monkeypatch.setattr(rpp, "PARSED_OUTPUT_DIR", str(parsed_dir), raising=False)
    return urunler_dir, parsed_dir


def test_kill_mid_checkpoint_write_leaves_no_truncated_file(ortam, monkeypatch):
    urunler_dir, _parsed_dir = ortam
    rec_a = _rec("A", urunler_dir)
    rec_b = _rec("B", urunler_dir)

    # A, kesintiden ONCE basariyla checkpoint'lenir.
    rpp.phase1_one(rec_a)
    ckpt_a = rpp._stage1_path(rec_a)
    assert ckpt_a.is_file()
    onceki_icerik = ckpt_a.read_text(encoding="utf-8")
    json.loads(onceki_icerik)  # gecerli JSON

    # B islenirken -- checkpoint'in TAM json.dump'i sirasinda -- kosu
    # "oldurulur" (KeyboardInterrupt). Once dosyaya YARIM veri yazilir, SONRA
    # kesilir; write-then-replace bu yarim veriyi gecici dosyada hapsetmeli,
    # B'nin gercek checkpoint yoluna asla ULASMAMALI.
    def _patlayan_dump(obj, fp, **kw):
        fp.write('{"content_hash": "YARIM YAZILDI VE KESILDI')
        raise KeyboardInterrupt("simulated kill mid-write")

    # Ayri bir monkeypatch kapsami: `ortam` fixture'inin BELGELER_DIR/
    # PARSED_OUTPUT_DIR ayarlarini BOZMADAN, yalniz json.dump'i gecici olarak
    # degistirip `with` blogu bitince otomatik geri alir.
    with monkeypatch.context() as mp:
        mp.setattr(rpp.json, "dump", _patlayan_dump)
        with pytest.raises(KeyboardInterrupt):
            rpp.phase1_one(rec_b)

    ckpt_b = rpp._stage1_path(rec_b)
    assert not ckpt_b.is_file(), (
        "write-then-replace: yarim yazilmis gecici dosya B'nin nihai "
        "checkpoint yoluna asla os.replace edilmemis olmali")
    # A'nin checkpoint'i bu kesintiden hic etkilenmemis.
    assert ckpt_a.read_text(encoding="utf-8") == onceki_icerik

    # --- bir sonraki kosu: kaldigi yerden devam -----------------------------
    a_mtime = ckpt_a.stat().st_mtime_ns
    rpp.phase1_one(rec_a)
    assert ckpt_a.stat().st_mtime_ns == a_mtime, (
        "ayni content_hash -> checkpoint AYNEN korunur, phase 1 yeniden calismaz")
    rpp.phase1_one(rec_b)
    assert ckpt_b.is_file()
    json.loads(ckpt_b.read_text(encoding="utf-8"))  # artik gecerli JSON

    # phase 2: ikisi de nihai IR'a donusur, checkpoint'ler harcanir (silinir).
    rpp.phase2_one(rec_a)
    rpp.phase2_one(rec_b)
    assert rec_a["parse"]["status"] == "SUCCESS"
    assert rec_b["parse"]["status"] == "SUCCESS"
    assert not ckpt_a.is_file()
    assert not ckpt_b.is_file()
    assert rec_a["parse"]["parsed_json_path"] and rec_b["parse"]["parsed_json_path"]


def test_kill_between_documents_next_run_resumes_only_pending(ortam, monkeypatch):
    """document_nodes.json seviyesinde: `_run_phase` her dokumandan SONRA
    registry'i atomik yazar (bkz. modul docstring'i) -- bir kosu N. dokumanda
    kesilirse, bir SONRAKI `compute_pending()` cagrisi yalnizca HALA
    PENDING olanlari doner (bitenler yeniden islenmez)."""
    urunler_dir, _parsed_dir = ortam
    registry = urunler_dir.parent / "document_nodes.json"
    records = [_rec("A", urunler_dir), _rec("B", urunler_dir), _rec("C", urunler_dir)]
    registry.write_text(json.dumps({"documents": records}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    monkeypatch.setattr(rpp, "DOCUMENT_NODES_PATH", str(registry), raising=False)
    monkeypatch.setattr(rpp, "DOC_TYPES", set(), raising=False)

    todo, info = rpp.compute_pending()
    assert {r["identity"]["doc_id"] for r in todo} == {"A", "B", "C"}
    data = info["data"]  # compute_pending kendi TAZE kopyasini diskten okur

    # Kosu A ve B'yi bitirir (phase1+phase2), sonra C'ye gelmeden "oldurulur"
    # -- registry'e YALNIZ A ve B'nin SUCCESS'i yazilir (gercek `_run_phase`
    # her dokumandan sonra registry'i atomik kaydeder, bkz. `main()`'daki
    # `_run_phase._run`).
    for rec in todo:
        if rec["identity"]["doc_id"] == "C":
            break
        rpp.phase1_one(rec)
        rpp.phase2_one(rec)
        rpp._atomic_write_json(str(registry), data)

    todo2, info2 = rpp.compute_pending()
    assert {r["identity"]["doc_id"] for r in todo2} == {"C"}, (
        "kesintiden SONRAKI kosu yalniz kalan dokumani islemeli")
    assert info2["total"] == 3
