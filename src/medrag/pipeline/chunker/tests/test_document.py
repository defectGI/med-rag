"""Inner document model (`chunker.core.document`) tests — offline,
dependency-free."""

import pytest
from pydantic import ValidationError

from medrag.pipeline.chunker.core.document import (
    Code,
    Document,
    Heading,
    Image,
    LinkRef,
    ListBlock,
    ListItem,
    Paragraph,
    Provenance,
    Table,
)


def ornek_dokuman() -> Document:
    """Representative document covering all block types."""
    return Document(
        doc_id="doc1",
        source_path="raw/kilavuz.pdf",
        fmt="pdf",
        page_count=3,
        blocks=[
            Heading(id="b0", text="Genel Bakış", level=1,
                    page_start=1, page_end=1),
            Paragraph(
                id="b1",
                text="Ayrıntı için kurulum bölümüne bakın.",
                heading_path=["Genel Bakış"],
                page_start=1, page_end=1,
                provenance=Provenance.VERIFIED,
                links=[LinkRef(text="kurulum bölümüne", target="#kurulum")],
            ),
            Table(
                id="b2",
                heading_path=["Genel Bakış"],
                cells=[["Parametre", "Değer"], ["Sıcaklık", "0-40 C"]],
                header_rows=1,
                description="Çalışma koşulları tablosu.",
                page_start=1, page_end=2,
            ),
            ListBlock(
                id="liste-1",
                heading_path=["Genel Bakış"],
                items=[
                    ListItem(level=0, ordered=True,
                             blocks=[Paragraph(id="b3", text="Kutuyu açın.")]),
                    ListItem(level=1, ordered=False,
                             blocks=[
                                 Paragraph(id="b4", text="İçindekiler:"),
                                 Table(id="b5", cells=[["Adet", "Parça"]],
                                       header_rows=0),
                             ]),
                ],
            ),
            Image(id="b6", image_id="sha256:abc", ocr_text="ŞEKİL 1",
                  alt_text="Bağlantı şeması", page_start=2, page_end=2,
                  provenance=Provenance.UNVERIFIED),
            Code(id="b7", text="pip install chunker", language="bash"),
        ],
    )


def test_json_gidis_donus_kayipsiz():
    doc = ornek_dokuman()
    tekrar = Document.model_validate_json(doc.model_dump_json())
    assert tekrar == doc


def test_discriminator_dogruu_tide_cozer():
    doc = Document.model_validate({
        "doc_id": "d", "source_path": "s", "fmt": "md",
        "blocks": [{"id": "b0", "kind": "code", "text": "x = 1"}],
    })
    assert isinstance(doc.blocks[0], Code)


def test_bilinmeyen_kind_reddedilir():
    with pytest.raises(ValidationError):
        Document.model_validate({
            "doc_id": "d", "source_path": "s", "fmt": "md",
            "blocks": [{"id": "b0", "kind": "footnote", "text": "?"}],
        })


def test_fazladan_alan_reddedilir():
    with pytest.raises(ValidationError):
        Paragraph(id="b0", text="x", boyle_bir_alan_yok=1)


def test_baslik_seviyesi_en_az_1():
    with pytest.raises(ValidationError):
        Heading(id="b0", text="x", level=0)


def test_liste_ogesi_liste_konteyneri_iceremez():
    # A nested list is represented as ListItem.level, not as a container.
    with pytest.raises(ValidationError):
        ListItem(blocks=[{"id": "x", "kind": "list", "items": []}])


def test_tablo_boyutlari():
    t = Table(id="t", cells=[["a", "b", "c"], ["d", "e"]])
    assert t.n_rows == 2
    assert t.n_cols == 3
    assert Table(id="t2").n_rows == 0
    assert Table(id="t2").n_cols == 0


def test_provenance_lowest_en_dusugu_bulur():
    assert Provenance.lowest(
        [Provenance.VERIFIED, Provenance.UNVERIFIED, Provenance.CONSENSUS]
    ) is Provenance.UNVERIFIED
    assert Provenance.lowest(
        [Provenance.VERIFIED, None, Provenance.CONSENSUS]
    ) is Provenance.CONSENSUS


def test_provenance_lowest_bilinmeyende_none():
    assert Provenance.lowest([]) is None
    assert Provenance.lowest([None, None]) is None


def test_block_by_id():
    doc = ornek_dokuman()
    assert doc.block_by_id("b2").kind == "table"
    assert doc.block_by_id("yok") is None