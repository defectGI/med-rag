"""ImageParser offline tests: IR yapısı, VLM metin akışı ve degrade davranışı.

`get_vlm_client()` monkeypatch ile taklit edilir (ağ yok). Bir üretim Görsel
belgesinin:
  1. parse'ta ParagraphBlock (VLM metni) + ImageBlock (provenance) ürettiğini,
  2. metnin chunk gövdesine girmek için doc.block metni olarak durduğunu,
  3. VLM yok/hata ise "degrade" ile yine geçerli bir ParsedDocument döndüğünü,
  4. registry `parser_for` ile .jpg/.png seçimini yaptığını doğrular.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medrag.pipeline.parser.parsers.base import ParsedDocument
from medrag.pipeline.parser.parsers.image_parser import ImageParser
from medrag.pipeline.parser.parsers.registry import parser_for

PNG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class _FakeVLM:
    def __init__(self, text: str = "Beyaz zemin üzerinde eczane logosu ve ilaç adı.") -> None:
        self.text = text
        self.calls = 0

    def complete_vision(self, *, system: str, user: str, images,
                        max_tokens: int = 2048) -> str:
        self.calls += 1
        return self.text


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (120, 60), (255, 255, 255))
    ImageDraw.Draw(im).text((5, 10), "PARAMAX 1000mg", fill=(0, 0, 0))
    p = tmp_path / "recel.png"
    im.save(p)
    return p


def test_registry_selects_image_parser(sample_png):
    for ext in PNG_EXTENSIONS:
        p = Path("/tmp/dosya") / f"r{ext}"
        assert isinstance(parser_for(p), ImageParser), ext


def test_parse_produces_text_and_image_blocks(sample_png, monkeypatch):
    fake = _FakeVLM("Beyaz zemin üzerinde 'PARAMAX 1000mg' etiketi.")
    monkeypatch.setattr(
        "medrag.pipeline.parser.llm.get_vlm_client",
        lambda role="primary", model=None: fake,
    )
    parser = ImageParser()
    doc = parser.parse(sample_png, "doc-img-1")

    assert isinstance(doc, ParsedDocument)
    assert doc.fmt == "image"
    assert doc.raw_sha256 and doc.raw_sha256.startswith("sha256:")
    assert doc.mimetype == "image/png"
    assert doc.parser_version  # PARSER_VERSION dolu

    texts = [b.text for b in doc.blocks if hasattr(b, "text")]
    assert any("PARAMAX" in t for t in texts), "VLM metni ParagraphBlock'a yazılmalı"
    img = [b for b in doc.blocks if hasattr(b, "image_id")]
    assert img, "ImageBlock korunmalı (provenance)"
    assert img[0].image_id == doc.raw_sha256
    assert img[0].mime == "image/png"
    assert img[0].width == 120 and img[0].height == 60
    assert fake.calls == 1


def test_parse_degrades_without_vlm(sample_png, monkeypatch):
    def _boom(role="primary", model=None):
        raise RuntimeError("VLM yapılandırılmamış")

    monkeypatch.setattr("medrag.pipeline.parser.llm.get_vlm_client", _boom)
    parser = ImageParser()
    doc = parser.parse(sample_png, "doc-img-2")

    # Yine de geçerli bir doc; degrade metni mevcut, ImageBlock duruyor.
    assert isinstance(doc, ParsedDocument)
    assert any("Görsel belge" in b.text for b in doc.blocks if hasattr(b, "text"))
    assert [b for b in doc.blocks if hasattr(b, "image_id")]


def test_parse_roundtrip_todict_fromdict(sample_png, monkeypatch):
    fake = _FakeVLM("Etiket: 500 mg.")
    monkeypatch.setattr(
        "medrag.pipeline.parser.llm.get_vlm_client",
        lambda role="primary", model=None: fake,
    )
    doc = ImageParser().parse(sample_png, "doc-img-3")
    data = doc.to_dict()
    doc2 = ParsedDocument.from_dict(data)
    assert doc2.fmt == "image"
    assert len(doc2.blocks) == len(doc.blocks)
    assert [b.type.value for b in doc2.blocks] == [b.type.value for b in doc.blocks]
