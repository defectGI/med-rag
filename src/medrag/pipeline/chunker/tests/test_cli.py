"""CLI (`chunker/cli.py`) offline tests.

Runs inject an env dict into `calistir(env)` — no real environment, no
.env file, no network; the tokenizer is always "fake". Input IRs are
hand-built v7 dicts (same approach as the adapter tests).

Output layout: two fixed JSON files —
all_chunks.json (doc_id -> ChunkSet), all_combined.json (wrapped form of
the same dictionary). No per-document file; hence no staleness gate —
every run regenerates the full input.
"""

import json
from datetime import datetime

import pytest

from medrag.pipeline.chunker import CHUNKER_VERSION
from medrag.pipeline.chunker.adapters.first_parse import ir
from medrag.pipeline.chunker.cli import calistir, kesfet, main
from medrag.pipeline.chunker.core.chunk import ChunkSet
from medrag.pipeline.chunker.layout import OutputLayout


def _ir(doc_id, bloklar=None):
    return {
        "ir_version": 7, "doc_id": doc_id, "source_path": f"{doc_id}.md",
        "fmt": "markdown",
        "blocks": bloklar if bloklar is not None else [
            {"type": "heading", "id": "b0", "text": "Kurulum", "level": 1,
             "anchor_id": "kurulum"},
            {"type": "paragraph", "id": "b1", "text": "Önce paketi indirin.",
             "heading_path": ["Kurulum"]},
        ],
    }


@pytest.fixture
def korpus(tmp_path):
    """A parsed_corpus-like nested layout: IR plus a stage1 wrapper,
    summary.json, and non-json side files."""
    girdi = tmp_path / "girdi"
    d1 = girdi / "01_A"
    d1.mkdir(parents=True)
    (d1 / "A.json").write_text(json.dumps(_ir("a")), encoding="utf-8")
    (d1 / "A.stage1.json").write_text(json.dumps({"ir": _ir("a")}),
                                      encoding="utf-8")
    (d1 / "A.md").write_text("# A", encoding="utf-8")
    (d1 / "run.log").write_text("ok", encoding="utf-8")
    d2 = girdi / "02_B"
    d2.mkdir()
    (d2 / "B.json").write_text(json.dumps(_ir("b")), encoding="utf-8")
    (girdi / "summary.json").write_text(json.dumps({"passed": 2}),
                                        encoding="utf-8")
    return girdi


def _env(korpus, tmp_path, **ekstra):
    e = {"CHUNKER_INPUT_DIR": str(korpus),
         "CHUNKER_OUTPUT_DIR": str(tmp_path / "cikti"),
         "TOKENIZER": "fake"}
    e.update(ekstra)
    return e


def _out(tmp_path) -> OutputLayout:
    return OutputLayout(tmp_path / "cikti")


def _belgeler(tmp_path) -> dict:
    """`documents` dict from all_chunks.json: doc_id -> ChunkSet."""
    veri = json.loads(_out(tmp_path).all_chunks_file.read_text(encoding="utf-8"))
    return {doc_id: ChunkSet.model_validate(d)
            for doc_id, d in veri["documents"].items()}


def _birlesik(tmp_path) -> dict:
    return json.loads(_out(tmp_path).all_combined_file.read_text(encoding="utf-8"))


# -- discovery -------------------------------------------------------------------


def test_kesif_ir_leri_bulur_yan_dosyalari_atlar(korpus):
    bulunan, atlanan = kesfet(korpus)
    assert [d["doc_id"] for _, d in bulunan] == ["a", "b"]
    # stage1 wrapper ({"ir": ...}) and summary.json fail the content check;
    # .md/.log aren't scanned anyway.
    assert atlanan == 2


def test_kesif_bozuk_json_ir_sayilmaz(tmp_path):
    (tmp_path / "x.json").write_text("{bozuk", encoding="utf-8")
    bulunan, atlanan = kesfet(tmp_path)
    assert bulunan == [] and atlanan == 1


# -- end to end -------------------------------------------------------------------


def test_uctan_uca_leaf_chunklar(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    belgeler = _belgeler(tmp_path)
    assert set(belgeler) == {"a", "b"}
    cs = belgeler["a"]
    assert cs.doc_id == "a"
    assert cs.nodes and all(n.is_leaf for n in cs.nodes)
    assert "Önce paketi indirin." in cs.nodes[0].text


def test_ust_uste_kosu_tum_girdiyi_yeniden_uretir(korpus, tmp_path):
    """No per-document file → no staleness gate: every run rewrites
    everything from scratch."""
    env = _env(korpus, tmp_path, CHUNKER_VIZ="off")
    assert calistir(env) == 0
    ilk = _out(tmp_path).all_chunks_file.stat().st_mtime_ns
    assert calistir(env) == 0
    assert _out(tmp_path).all_chunks_file.stat().st_mtime_ns >= ilk
    assert set(_belgeler(tmp_path)) == {"a", "b"}


def test_ust_uste_kosu_eski_ciktiyi_silmez_arsivler(korpus, tmp_path):
    """The previous run's output is NOT DELETED — it's moved into a dated
    archive folder so the provenance chain can retroactively find old
    chunk text."""
    env = _env(korpus, tmp_path, CHUNKER_VIZ="off")
    assert calistir(env) == 0
    ilk_icerik = _out(tmp_path).all_chunks_file.read_text(encoding="utf-8")
    ilk_damga = json.loads(ilk_icerik)["generated_at"]

    assert calistir(env) == 0

    out = _out(tmp_path)
    arsiv_klasoru = out.root / "archive" / ilk_damga.replace(":", "-")
    assert arsiv_klasoru.is_dir()
    arsivlenen = (arsiv_klasoru / "all_chunks.json").read_text(encoding="utf-8")
    assert arsivlenen == ilk_icerik  # old content preserved EXACTLY
    # current location still readable with the right docs (new run)
    assert set(_belgeler(tmp_path)) == {"a", "b"}


def test_ilk_kosuda_arsiv_klasoru_olusmaz(korpus, tmp_path):
    """When there's no previous output to archive (first run), archive/
    is never created."""
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    assert not (_out(tmp_path).root / "archive").exists()


# -- provenance ----------------------------------------------------------------


def test_provenance_damgalanir(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    p = _belgeler(tmp_path)["a"].provenance
    assert p is not None
    assert p.chunker_version == CHUNKER_VERSION
    # Offset local ISO (a naive timestamp must not pass silently).
    assert datetime.fromisoformat(p.generated_at).tzinfo is not None
    # Recorded version is the version chunks were PRODUCED under: input
    # file is v7 but the adapter migrates it to the current schema.
    assert p.source.ir_version == ir.IR_VERSION
    assert p.raptor is None


def test_alt_cizgi_onekli_doc_id_reddedilir(korpus, tmp_path):
    # `_` prefix is reserved for scope identifiers; such a doc_id makes
    # that document fail (run continues, exit code 1).
    (korpus / "02_B" / "rezerve.json").write_text(
        json.dumps(_ir("_gizli")), encoding="utf-8")
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 1
    belgeler = _belgeler(tmp_path)
    assert "_gizli" not in belgeler
    assert "a" in belgeler


# -- visualization -----------------------------------------------------------------


def test_viz_varsayilan_html_uretir(korpus, tmp_path):
    # Without CHUNKER_VIZ, the run produces .tree.html + index.html per document.
    assert calistir(_env(korpus, tmp_path)) == 0
    out = _out(tmp_path)
    adlar = sorted(p.name for p in out.viz.iterdir())
    assert set(adlar) == {"a.tree.html", "b.tree.html", "index.html"}
    idx = (out.viz / "index.html").read_text(encoding="utf-8")
    assert "a.tree.html" in idx and "b.tree.html" in idx


def test_viz_kapatilabilir(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    # viz/ folder is STILL created (the layout lives there every run) but stays empty.
    assert list(_out(tmp_path).viz.iterdir()) == []


# -- error model --------------------------------------------------------------------


def test_input_dir_tanimsiz(tmp_path):
    assert calistir({"CHUNKER_OUTPUT_DIR": str(tmp_path)}) == 2


def test_input_dir_klasor_degil(tmp_path):
    assert calistir({"CHUNKER_INPUT_DIR": str(tmp_path / "yok"),
                     "TOKENIZER": "fake"}) == 2


# -- --limit --------------------------------------------------------------------


def test_limit_ilk_n_iri_isler(korpus, tmp_path):
    # 2 IRs; --limit 1 processes only the first ("a" in discovery order).
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off"), limit=1) == 0
    assert set(_belgeler(tmp_path)) == {"a"}


def test_limit_toplamdan_buyukse_hepsi_islenir(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off"), limit=99) == 0
    assert set(_belgeler(tmp_path)) == {"a", "b"}


def test_limit_none_hepsi_islenir(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    assert len(_belgeler(tmp_path)) == 2


def test_main_limit_bayragini_gecirir(korpus, tmp_path, monkeypatch):
    # main() parses --limit from argv and forwards to calistir; env comes
    # from real environment, so we use monkeypatch (.env loading is no-op).
    monkeypatch.setenv("CHUNKER_INPUT_DIR", str(korpus))
    monkeypatch.setenv("CHUNKER_OUTPUT_DIR", str(tmp_path / "cikti"))
    monkeypatch.setenv("TOKENIZER", "fake")
    monkeypatch.setenv("CHUNKER_VIZ", "off")
    assert main(["--limit", "1"]) == 0
    assert set(_belgeler(tmp_path)) == {"a"}


def test_main_gecersiz_limit_reddedilir(korpus, tmp_path):
    # argparse rejects 0/negative/non-numeric --limit with SystemExit(2).
    with pytest.raises(SystemExit):
        main(["--limit", "0"])
    with pytest.raises(SystemExit):
        main(["--limit", "abc"])


def test_hic_ir_bulunamamasi_yapilandirma_hatasi(tmp_path):
    # An empty input folder almost always means a wrong path — NOT a
    # silent "0 processed" success, a configuration error.
    bos = tmp_path / "bos"
    bos.mkdir()
    assert calistir({"CHUNKER_INPUT_DIR": str(bos), "TOKENIZER": "fake"}) == 2


def test_cakisan_doc_id_hata_diger_dokuman_islenir(korpus, tmp_path):
    # A second file carrying the same doc_id makes that document fail
    # (no silent overwrite); the run continues, exit code 1.
    (korpus / "02_B" / "B_kopya.json").write_text(
        json.dumps(_ir("b")), encoding="utf-8")
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 1
    assert set(_belgeler(tmp_path)) == {"a", "b"}


def test_bozuk_ir_dokumani_digerlerini_dusurmez(korpus, tmp_path):
    # ir_version+blocks present (passes discovery) but doc_id missing →
    # adapter fails at boundary; only that document counts as failed.
    bozuk = {"ir_version": 7, "blocks": []}
    (korpus / "02_B" / "bozuk.json").write_text(json.dumps(bozuk),
                                                encoding="utf-8")
    assert calistir(_env(korpus, tmp_path)) == 1
    belgeler = _belgeler(tmp_path)
    assert "a" in belgeler and "b" in belgeler


def test_gelecek_versiyon_dosyasi_reddedilir(korpus, tmp_path):
    # The version gate also works on the CLI path: a file from the
    # reader's future makes that document fail (no silent
    # "best-effort read"). 999 must stay greater than IR_VERSION from
    # parser/parsers/base.py — bump this constant if IR_VERSION moves.
    (korpus / "02_B" / "gelecek.json").write_text(
        json.dumps(_ir("c") | {"ir_version": 999}), encoding="utf-8")
    assert calistir(_env(korpus, tmp_path)) == 1


def test_cross_ref_cli_yolunda_cozulur(tmp_path):
    # Document-internal link (#kurulum) → anchor_id match → target chunk id.
    bloklar = [
        {"type": "heading", "id": "b0", "text": "Kurulum", "level": 1,
         "anchor_id": "kurulum"},
        {"type": "paragraph", "id": "b1", "text": "Önce paketi indirin.",
         "heading_path": ["Kurulum"]},
        {"type": "paragraph", "id": "b2",
         "text": "Ayrıntı için kurulum bölümüne bakın.",
         "heading_path": ["Diğer"],
         "runs": [{"text": "kurulum bölümüne", "link": "#kurulum"}]},
    ]
    girdi = tmp_path / "girdi"
    girdi.mkdir()
    (girdi / "d.json").write_text(json.dumps(_ir("d", bloklar)),
                                  encoding="utf-8")
    assert calistir(_env(girdi, tmp_path)) == 0
    cs = _belgeler(tmp_path)["d"]
    refler = [r for n in cs.nodes for r in n.cross_refs]
    assert refler and all(r.target_chunk_id for r in refler)


def test_birlesik_dosya_belgeleri_icerir(korpus, tmp_path):
    assert calistir(_env(korpus, tmp_path, CHUNKER_VIZ="off")) == 0
    birlesik = _birlesik(tmp_path)
    assert set(birlesik["documents"]) == {"a", "b"}