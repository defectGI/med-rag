"""`tesseract_lang_override()` is the single point both
`parsers/pdf_parser.py::TesseractDetector` and `images/image_handler.py`'s
OCR fallback now read `PDF_TESSERACT_LANG` through, instead of each calling
`os.getenv` independently."""

from __future__ import annotations

from medrag.pipeline.parser.storage_paths import tesseract_lang_override


def test_tesseract_lang_override_unset_is_none(monkeypatch):
    monkeypatch.delenv("PDF_TESSERACT_LANG", raising=False)
    assert tesseract_lang_override() is None


def test_tesseract_lang_override_empty_is_none(monkeypatch):
    monkeypatch.setenv("PDF_TESSERACT_LANG", "")
    assert tesseract_lang_override() is None


def test_tesseract_lang_override_reads_value(monkeypatch):
    monkeypatch.setenv("PDF_TESSERACT_LANG", "deu")
    assert tesseract_lang_override() == "deu"
