"""Central storage-path configuration: where raw input files, parsed IR
output, and the image/crop blob store live. All three default to the
dev-time `storage/` layout (see storage/README.md) but are overridable via
environment variables (and `.env`, loaded the same way llm/ loads model
config) so a deployment isn't stuck with paths baked into scripts.

Environment (all optional):
    STORAGE_RAW_DIR      raw input files (default storage/raw)
    STORAGE_OUTPUT_DIR   parsed IR JSON, one file per doc_id (default storage/output)
    STORAGE_IMAGES_DIR   image/crop blob store (default storage/images). Files are
                         named with the BARE sha256 hex + mime extension; the IR
                         field (`image_id`/`source_crop`) says `sha256:<hex>` --
                         `hash_hex()` in parsers/base.py bridges the two
                         (`:` is illegal in a Windows filename, hence the split).
    STORAGE_LABELS_DIR   visual-type classification cache, keyed by
                         sha256(crop bytes + prompt version), so a prompt/logic
                         change invalidates the cache (default storage/labels) --
                         see images/visual_classify.py.
                         Deliberately outside STORAGE_OUTPUT_DIR/STORAGE_IMAGES_DIR:
                         it must survive both an IR_VERSION bump (which regenerates
                         every doc's IR from scratch) and PARSER_VERSION-triggered
                         re-parses, so a crop already labeled is never reclassified.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if there's no .env file; never overrides a set env var


def raw_dir() -> Path:
    return Path(os.getenv("STORAGE_RAW_DIR", "storage/raw"))


def output_dir() -> Path:
    return Path(os.getenv("STORAGE_OUTPUT_DIR", "storage/output"))


def images_dir() -> Path:
    return Path(os.getenv("STORAGE_IMAGES_DIR", "storage/images"))


def labels_dir() -> Path:
    return Path(os.getenv("STORAGE_LABELS_DIR", "storage/labels"))


def tesseract_lang_override() -> str | None:
    """`PDF_TESSERACT_LANG` -- deploy-specific pytesseract language override,
    read in exactly one place: both
    `parsers/pdf_parser.py::TesseractDetector` and
    `images/image_handler.py`'s OCR fallback go through here."""
    return os.getenv("PDF_TESSERACT_LANG") or None
