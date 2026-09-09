"""Standalone image (JPEG/PNG) -> ParsedDocument via VLM (K6 extension).

A user uploading a photo/scanned image as a document wants it to become a
searchable, citable piece of the corpus. A raster image has no deterministic
text layer, so the only source of content is a vision model: this parser
calls `get_vlm_client()` to transcribe + describe the image and writes the
result as normal `ParagraphBlock` body text, so it flows through
chunk -> vectorize -> retrieval exactly like any other block text. An
`ImageBlock` is kept too, for provenance (blob store + evidence `[n]` badge).

Degradation (offline / no VLM): if no vision client can be built or the call
fails/returns nothing usable, the parser still returns a valid `ParsedDocument`
carrying a short placeholder paragraph -- an upload never hard-fails, the
document simply has no searchable text until VLM is available (a re-parse
after configuring VLM recovers it).
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from PIL import Image
except (
    ImportError
):  # pragma: no cover - Pillow is a parse extra; guard keeps import offline
    Image = None  # type: ignore[assignment]

from .base import (
    IR_VERSION,
    PARSER_VERSION,
    BaseParser,
    ImageBlock,
    ParagraphBlock,
    ParsedDocument,
    Span,
    sha256_id,
)

logger = logging.getLogger("medrag.pipeline.parser.image")

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_MIME_BY_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_SYSTEM = (
    "You are reading a standalone image that was uploaded as a document. "
    "Transcribe and describe what is actually in the image so that someone "
    "who cannot see it can still find it via text search.\n"
    "Rules:\n"
    "- Plain text only: no markdown, no bullet points, no preamble or quotes.\n"
    "- Transcribe clearly legible text verbatim (labels, headings, numbers, "
    "dosage, patient/drug names). Never invent text you cannot read.\n"
    "- Then describe the subject and any visible structure in a couple of "
    "sentences.\n"
    "- If the image is too small, blurry, or empty to make out, say so plainly "
    "rather than guessing.\n"
    "- Write in the language of whichever text you see; if there is none, "
    "respond in Turkish."
)

#: IR_VERSION is unaffected by adding new blocks/types; it was only the
#: format tag and parser version. The next PARSER_VERSION bump follows this
#: prompt change.
_PROMPT_VERSION = 1

_FALLBACK_TEXT = "Görsel belge — bu içerik için VLM açıklaması alınamadı."


class ImageParser(BaseParser):
    """`jpg/jpeg/png` -> IR. Body text comes from a VLM transcription."""

    extensions = _IMAGE_EXTENSIONS
    mimetypes = tuple(_MIME_BY_EXT.values())
    fmt = "image"

    def parse(self, raw_path: str | Path, doc_id: str) -> ParsedDocument:
        raw_path = Path(raw_path)
        data = raw_path.read_bytes()
        ext = raw_path.suffix.lower()
        mime = _MIME_BY_EXT.get(ext, "image/*")
        raw_sha256 = sha256_id(data)

        width = height = None
        if Image is not None:
            try:
                with Image.open(raw_path) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001 -- size info optional
                width = height = None

        text = self._describe(data, mime)
        if not text or not text.strip():
            text = _FALLBACK_TEXT

        # Single block id generation (the next_id discipline from markdown).
        blk = "b0"
        blocks = [
            ParagraphBlock(
                id=blk,
                span=Span(byte_start=0, byte_end=len(data)),
                text=text.strip(),
            ),
            ImageBlock(
                id="img1",
                span=Span(byte_start=0, byte_end=len(data)),
                image_index=1,
                # locator: the image_handler/blob store resolves this; the
                # standalone image itself is both source and blob.
                locator={"file": raw_path.name, "source": "standalone-image"},
                image_id=raw_sha256,
                mime=mime,
                width=width,
                height=height,
                # This parser produced the description itself (not enrichment).
                description=text.strip(),
                description_source="vlm-vision",
            ),
        ]

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(raw_path),
            fmt=self.fmt,
            raw_sha256=raw_sha256,
            mimetype=mime,
            parser_version=PARSER_VERSION,
            ir_version=IR_VERSION,
            metadata={"image_prompt_version": _PROMPT_VERSION},
            blocks=blocks,
        )

    @staticmethod
    def _describe(data: bytes, mime: str) -> str:
        """Analyze the image with the VLM; return "" if it fails/returns nothing."""
        try:
            from medrag.pipeline.parser.llm import get_vlm_client

            client = get_vlm_client()
        except Exception as exc:  # noqa: BLE001 -- no VLM: degrade
            logger.info("VLM istemcisi kurulamadı (%s) — görsel metni boş", exc)
            return ""
        try:
            return client.complete_vision(
                system=_SYSTEM,
                user="Describe this image.",
                images=[(mime, data)],
            )
        except Exception as exc:  # noqa: BLE001 -- network/key error: degrade
            logger.warning("VLM görsel açıklaması başarısız (%s)", exc)
            return ""


__all__ = ["ImageParser"]
