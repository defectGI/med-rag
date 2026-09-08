"""Chunk output schema (`chunker.core.chunk`) tests — offline, dependency-free."""

import pytest
from pydantic import ValidationError

from medrag.pipeline.chunker.core.chunk import (
    CHUNK_SCHEMA_VERSION,
    CORPUS_SCOPE_ID,
    ChunkNode,
    ChunkProvenance,
    ChunkSet,
    CrossRef,
    ImageRef,
    RaptorProvenance,
    SourceProvenance,
    _compute_content_sha256,
    leaf_node_id,
    profile_scope_id,
    summary_node_id,
)
from medrag.pipeline.chunker.core.document import Provenance


def ornek_leaf(n: int = 0, **degisiklik) -> ChunkNode:
    alanlar = {
        "node_id": leaf_node_id("doc1", n),
        "doc_id": "doc1",
        "tree_level": 0,
        "text": "Genel Bakış\n\nAyrıntı için kurulum bölümüne bakın.",
        "source_path": "raw/kilavuz.pdf",
        "fmt": "pdf",
        "source_block_ids": ["b0", "b1"],
        "heading_path": ["Genel Bakış"],
        "page_start": 1,
        "page_end": 1,
        "token_count": 42,
        "token_limit": 512,
        "images": [ImageRef(image_id="sha256:abc", ocr_text="ŞEKİL 1",
                            alt_text="Bağlantı şeması")],
        "cross_refs": [CrossRef(source_block_id="b1", link="#kurulum")],
        "provenance_summary": Provenance.VERIFIED,
    }
    alanlar.update(degisiklik)
    return ChunkNode(**alanlar)


def test_node_id_formatlari():
    assert leaf_node_id("doc1", 3) == "doc1::c3"
    assert summary_node_id("doc1", 2, 0) == "doc1::s2.0"


def test_leaf_kurulur_enrichment_alanlari_bos():
    leaf = ornek_leaf()
    assert leaf.is_leaf
    assert leaf.summary is None
    assert leaf.keywords == []
    assert leaf.parent_id is None
    assert leaf.cross_refs[0].target_chunk_id is None


def test_ozet_dugum_kurulur():
    ozet = ChunkNode(
        node_id=summary_node_id("doc1", 1, 0),
        doc_id="doc1",
        tree_level=1,
        text="Doküman genel bakış ve kurulum adımlarını anlatır.",
        summary="Genel bakış + kurulum.",
        keywords=["kurulum", "genel bakış"],
        child_ids=[leaf_node_id("doc1", 0), leaf_node_id("doc1", 1)],
    )
    assert not ozet.is_leaf


def test_ozet_dugumde_leaf_alani_reddedilir():
    with pytest.raises(ValidationError, match="source_block_ids"):
        ChunkNode(node_id="doc1::s1.0", doc_id="doc1", tree_level=1,
                  text="özet", source_block_ids=["b0"])
    with pytest.raises(ValidationError, match="token_count"):
        ChunkNode(node_id="doc1::s1.0", doc_id="doc1", tree_level=1,
                  text="özet", token_count=5)


def test_ozet_dugumde_heading_path_ve_sayfa_agregeli_dolabilir():
    # v2: these three are NO LONGER leaf-only — summary nodes may carry
    # the aggregation over their children (common heading prefix, min/max
    # page).
    ozet = ChunkNode(node_id="doc1::s1.0", doc_id="doc1", tree_level=1,
                      text="özet", heading_path=["Giriş"],
                      page_start=2, page_end=5)
    assert ozet.heading_path == ["Giriş"]
    assert ozet.page_start == 2 and ozet.page_end == 5


def test_negatif_tree_level_reddedilir():
    with pytest.raises(ValidationError):
        ornek_leaf(tree_level=-1)


def test_split_kontrati():
    parca = ornek_leaf(split_kind="table", split_index=1, split_total=3,
                       overlap_units=2)
    assert parca.split_total == 3
    # split_kind accepts only known block types
    with pytest.raises(ValidationError):
        ornek_leaf(split_kind="heading", split_index=1, split_total=2)
    # split_index/total required when split_kind is set
    with pytest.raises(ValidationError, match="split_index"):
        ornek_leaf(split_kind="paragraph")
    # index > total not allowed
    with pytest.raises(ValidationError, match="split_index"):
        ornek_leaf(split_kind="table", split_index=4, split_total=3)
    # split fields meaningless in an unsplit chunk
    with pytest.raises(ValidationError, match="split_kind"):
        ornek_leaf(split_index=1, split_total=2)


def test_flex_kontrati():
    esnek = ornek_leaf(flex_applied=True, flex_amount=64,
                       flex_reason="tablo bölünmesin diye")
    assert esnek.flex_amount == 64
    with pytest.raises(ValidationError, match="flex_amount"):
        ornek_leaf(flex_amount=64)  # flex_applied=False here


def test_sayfa_araligi_tutarliligi():
    with pytest.raises(ValidationError, match="page_start"):
        ornek_leaf(page_start=5, page_end=2)


def test_content_sha256_otomatik_hesaplanir():
    """If `content_sha256` is not given, it's computed from `text` automatically."""
    dugum = ornek_leaf()
    assert dugum.content_sha256 == _compute_content_sha256(dugum.text)
    assert dugum.content_sha256.startswith("sha256:")


def test_content_sha256_tutarsizlik_reddedilir():
    """If the given `content_sha256` doesn't match the recomputed one from
    `text`, it's rejected silently — drift is caught at construction."""
    with pytest.raises(ValidationError, match="content_sha256"):
        ornek_leaf(content_sha256="sha256:" + "0" * 64)


def test_content_sha256_dogru_deger_kabul_edilir():
    """A pre-computed correct value (e.g. read from disk) is accepted when
    it matches the recomputation."""
    metin = "Genel Bakış\n\nAyrıntı için kurulum bölümüne bakın."
    dogru = _compute_content_sha256(metin)
    dugum = ornek_leaf(content_sha256=dogru)
    assert dugum.content_sha256 == dogru


def test_chunkset_json_gidis_donus(tmp_path):
    cs = ChunkSet(doc_id="doc1", nodes=[
        ornek_leaf(0, next_node_id=leaf_node_id("doc1", 1)),
        ornek_leaf(1, prev_node_id=leaf_node_id("doc1", 0)),
    ])
    yol = cs.save(tmp_path / "doc1.chunks.json")
    tekrar = ChunkSet.load(yol)
    assert tekrar == cs
    assert tekrar.schema_version == CHUNK_SCHEMA_VERSION


def test_chunkset_none_alanlari_jsona_yazmaz():
    cs = ChunkSet(doc_id="doc1", nodes=[ornek_leaf()])
    assert '"summary"' not in cs.to_json()
    assert '"flex_reason"' not in cs.to_json()


def test_chunkset_yardimcilari():
    leaf = ornek_leaf()
    ozet = ChunkNode(node_id="doc1::s1.0", doc_id="doc1", tree_level=1,
                     text="özet", child_ids=[leaf.node_id])
    cs = ChunkSet(doc_id="doc1", nodes=[leaf, ozet])
    assert cs.leaves() == [leaf]
    assert cs.node_by_id("doc1::s1.0") is ozet
    assert cs.node_by_id("yok") is None


def test_gelecek_surum_reddedilir():
    with pytest.raises(ValidationError, match="schema"):
        ChunkSet(doc_id="doc1", schema_version=CHUNK_SCHEMA_VERSION + 1)


# -- scope contract — v3 --------------------------------------------------------


def kapsam_dugumu(kapsam_id: str = CORPUS_SCOPE_ID, n: int = 0) -> ChunkNode:
    return ChunkNode(node_id=summary_node_id(kapsam_id, 1, n),
                     doc_id=kapsam_id, tree_level=1, text="özet",
                     child_ids=[leaf_node_id("doc1", n)])


def test_kapsam_kimlik_yardimcilari():
    assert CORPUS_SCOPE_ID == "_corpus"
    assert profile_scope_id("p1") == "_profile.p1"
    assert summary_node_id(CORPUS_SCOPE_ID, 1, 0) == "_corpus::s1.0"


def test_kapsam_seti_gidis_donus(tmp_path):
    cs = ChunkSet(doc_id=CORPUS_SCOPE_ID, scope="corpus",
                  member_doc_ids=["doc1", "doc2"], nodes=[kapsam_dugumu()])
    yol = cs.save(tmp_path / "_corpus.chunks.json")
    tekrar = ChunkSet.load(yol)
    assert tekrar == cs
    assert tekrar.scope == "corpus"
    assert tekrar.member_doc_ids == ["doc1", "doc2"]


def test_v2_dosya_document_scope_ile_okunur():
    # An old (v2) file has no scope/member_doc_ids — reads with defaults.
    cs = ChunkSet.from_json(
        '{"schema_version": 2, "doc_id": "doc1", "nodes": []}')
    assert cs.scope == "document" and cs.member_doc_ids == []


def test_kapsam_seti_leaf_iceremez():
    with pytest.raises(ValidationError, match="not contain leaves"):
        ChunkSet(doc_id=CORPUS_SCOPE_ID, scope="corpus",
                 member_doc_ids=["doc1"], nodes=[ornek_leaf()])


def test_kapsam_setinde_dugum_doc_id_kapsam_kimligi_olmali():
    with pytest.raises(ValidationError, match="doc_id"):
        ChunkSet(doc_id=CORPUS_SCOPE_ID, scope="corpus",
                 member_doc_ids=["doc1"],
                 nodes=[kapsam_dugumu(profile_scope_id("p1"))])


def test_kapsam_seti_bos_uye_listesi_reddedilir():
    with pytest.raises(ValidationError, match="member_doc_ids"):
        ChunkSet(doc_id=CORPUS_SCOPE_ID, scope="corpus")


def test_dokuman_seti_uye_listesi_tasiyamaz():
    with pytest.raises(ValidationError, match="member_doc_ids"):
        ChunkSet(doc_id="doc1", member_doc_ids=["doc1"])


# -- provenance — v4 ------------------------------------------------------------


def ornek_provenance(**degisiklik) -> ChunkProvenance:
    alanlar = {
        "chunker_version": "1.0.0",
        "generated_at": "2026-07-17T14:03:22+03:00",
        "source": SourceProvenance(raw_sha256="sha256:abc",
                                   parser_version="1.4.0", ir_version=9),
        "raptor": RaptorProvenance(mode="real", model="ollama:qwen2.5:32b",
                                   prompt_version="1"),
    }
    alanlar.update(degisiklik)
    return ChunkProvenance(**alanlar)


def test_provenance_gidis_donus(tmp_path):
    cs = ChunkSet(doc_id="doc1", provenance=ornek_provenance(),
                  nodes=[ornek_leaf()])
    tekrar = ChunkSet.load(cs.save(tmp_path / "doc1.chunks.json"))
    assert tekrar == cs
    assert tekrar.provenance.source.raw_sha256 == "sha256:abc"
    assert tekrar.provenance.raptor.model == "ollama:qwen2.5:32b"


def test_provenance_v3_dosyada_yok_sayilir():
    # In a pre-v4 file the provenance field is absent — reads as None
    # (and the CLI's staleness gate treats that as "stale" and regenerates).
    cs = ChunkSet.from_json(
        '{"schema_version": 3, "doc_id": "doc1", "nodes": []}')
    assert cs.provenance is None


def test_same_inputs_generated_at_farkini_yok_sayar():
    a = ornek_provenance()
    b = ornek_provenance(generated_at="2026-07-18T09:00:00+03:00")
    assert a.same_inputs(b)


@pytest.mark.parametrize("degisiklik", [
    {"chunker_version": "1.1.0"},
    {"source": SourceProvenance(raw_sha256="sha256:BASKA",
                                parser_version="1.4.0", ir_version=9)},
    {"source": SourceProvenance(raw_sha256="sha256:abc",
                                parser_version="1.5.0", ir_version=9)},
    {"raptor": RaptorProvenance(mode="real", model="ollama:baska",
                                prompt_version="1")},
    {"raptor": RaptorProvenance(mode="real", model="ollama:qwen2.5:32b",
                                prompt_version="2")},
    {"raptor": None},
])
def test_same_inputs_her_girdi_degisiminde_bayat(degisiklik):
    assert not ornek_provenance().same_inputs(ornek_provenance(**degisiklik))