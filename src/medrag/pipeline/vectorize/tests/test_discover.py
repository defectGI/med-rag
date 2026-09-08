import json

import pytest

from medrag.pipeline.vectorize.discover import discover_chunk_scopes


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _node(node_id, doc_id):
    return {"node_id": node_id, "doc_id": doc_id, "tree_level": 0, "text": "x"}


def test_bulunmayan_klasor_yukselir(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_chunk_scopes(tmp_path / "yok")


def test_tek_dosya_yolu_dogrudan_verilebilir(tmp_path):
    dosya = tmp_path / "all_chunks.json"
    _write(dosya, {
        "documents": {
            "DOC1": {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]},
            "DOC2": {"doc_id": "DOC2", "nodes": [_node("DOC2::c0", "DOC2")]},
        },
    })
    kapsamlar = discover_chunk_scopes(dosya)
    assert sorted(k.scope_id for k in kapsamlar) == ["DOC1", "DOC2"]


def test_tek_dosya_yolu_bare_chunkset(tmp_path):
    dosya = tmp_path / "DOC1.chunks.json"
    _write(dosya, {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]})
    kapsamlar = discover_chunk_scopes(dosya)
    assert [k.scope_id for k in kapsamlar] == ["DOC1"]


def test_tekil_chunkset_dosyasi_bulunur(tmp_path):
    _write(tmp_path / "DOC1.chunks.json",
          {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]})
    kapsamlar = discover_chunk_scopes(tmp_path)
    assert [k.scope_id for k in kapsamlar] == ["DOC1"]
    assert kapsamlar[0].chunk_set.nodes[0].node_id == "DOC1::c0"


def test_all_chunks_bicimi_bulunur(tmp_path):
    _write(tmp_path / "all_chunks.json", {
        "generated_at": "x",
        "documents": {
            "DOC1": {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]},
            "DOC2": {"doc_id": "DOC2", "nodes": [_node("DOC2::c0", "DOC2")]},
        },
    })
    kapsamlar = discover_chunk_scopes(tmp_path)
    assert sorted(k.scope_id for k in kapsamlar) == ["DOC1", "DOC2"]


def test_all_combined_bicimi_documents_ve_trees_birlestirir(tmp_path):
    _write(tmp_path / "all_combined.json", {
        "documents": {"DOC1": {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]}},
        "raptor": {"_corpus": {"doc_id": "_corpus", "scope": "corpus",
                              "member_doc_ids": ["DOC1"],
                              "nodes": [{"node_id": "_corpus::s1.0", "doc_id": "_corpus",
                                        "tree_level": 1, "text": "özet"}]}},
    })
    # not: cli.py'nin gerçek yazdığı anahtar "raptor" değil "trees"tir (bkz.
    # chunker/chunker/cli.py); burada kasıtlı yanlış anahtarla atlanmayı test
    # etmiyoruz, doğru anahtarla ayrı test yapılır aşağıda.
    kapsamlar = discover_chunk_scopes(tmp_path)
    assert [k.scope_id for k in kapsamlar] == ["DOC1"]


def test_all_combined_trees_anahtariyla_kapsam_kimligi_de_gelir(tmp_path):
    _write(tmp_path / "all_combined.json", {
        "documents": {"DOC1": {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]}},
        "trees": {"_corpus": {"doc_id": "_corpus", "scope": "corpus",
                              "member_doc_ids": ["DOC1"],
                              "nodes": [{"node_id": "_corpus::s1.0", "doc_id": "_corpus",
                                        "tree_level": 1, "text": "özet"}]}},
    })
    kapsamlar = discover_chunk_scopes(tmp_path)
    assert sorted(k.scope_id for k in kapsamlar) == ["DOC1", "_corpus"]


def test_ilgisiz_json_atlanir(tmp_path):
    _write(tmp_path / "summary.json", {"anlamsiz": True})
    assert discover_chunk_scopes(tmp_path) == []


def test_bozuk_json_atlanir(tmp_path):
    (tmp_path / "bozuk.json").write_text("{bu json degil", encoding="utf-8")
    assert discover_chunk_scopes(tmp_path) == []


def test_gecersiz_chunkset_atlanir_digerleri_bulunur(tmp_path):
    _write(tmp_path / "all_chunks.json", {
        "documents": {
            "DOC1": {"doc_id": "DOC1", "nodes": [_node("DOC1::c0", "DOC1")]},
            "DOC2": {"nodes": "gecersiz-cunku-liste-degil"},
        },
    })
    kapsamlar = discover_chunk_scopes(tmp_path)
    assert [k.scope_id for k in kapsamlar] == ["DOC1"]


def test_archive_alt_agaci_taranmaz(tmp_path):
    """`chunker/cli.py::_archive_existing_output` her kosuda bir onceki
    ciktiyi `<kok>/archive/<damga>/` altina tasir (KARAR-041) ve uretimde
    `VECTORIZE_INPUT_DIR` o kokun TA KENDISI -- arsiv taranirsa Temmuz'dan
    kalma chunk'lar Qdrant'a indekslenir (2026-08-26 sunucu olcumu: 198
    dokumana karsilik 1188 kapsam)."""
    _write(tmp_path / "all_chunks.json", {
        "documents": {"GUNCEL": {"doc_id": "GUNCEL", "nodes": [_node("GUNCEL::c0", "GUNCEL")]}},
    })
    _write(tmp_path / "archive" / "2026-07-27T08-53-27+03-00" / "all_chunks.json", {
        "documents": {"BAYAT": {"doc_id": "BAYAT", "nodes": [_node("BAYAT::c0", "BAYAT")]}},
    })

    kapsamlar = discover_chunk_scopes(tmp_path)

    assert [k.scope_id for k in kapsamlar] == ["GUNCEL"]


def test_kokun_kendisi_arsivse_taranir(tmp_path):
    """Eski bir kosuyu BILEREK yeniden indekslemek isteyen operatorun yolu
    kapanmaz -- atlanan yalniz kokun ALTINDAKI `archive/` agaci."""
    arsiv_koku = tmp_path / "archive" / "2026-07-27T08-53-27+03-00"
    _write(arsiv_koku / "all_chunks.json", {
        "documents": {"BAYAT": {"doc_id": "BAYAT", "nodes": [_node("BAYAT::c0", "BAYAT")]}},
    })

    kapsamlar = discover_chunk_scopes(arsiv_koku)

    assert [k.scope_id for k in kapsamlar] == ["BAYAT"]
