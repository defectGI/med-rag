"""Render a PDF's pages to PNG bytes for a vision model.

The judge sees the PDF as the source of truth. Since the provider-agnostic
`VLMClient` takes images (not a PDF document part), pages are rasterized the
same way the parser itself does it -- pdfplumber's `page.to_image(...).original`
(pypdfium2 under the hood), so no new dependency is introduced.
"""

from __future__ import annotations

import io
from pathlib import Path


def render_pdf(pdf: Path, *, dpi: int = 150, max_pages: int | None = None
               ) -> tuple[list[tuple[str, bytes]], int]:
    """Return (images, total_pages).

    `images` is a list of ("image/png", bytes), one per rendered page, capped at
    `max_pages` when set. `total_pages` is the PDF's real page count so the
    caller can report how many pages went ungraded rather than truncating
    silently.
    """
    import pdfplumber

    images: list[tuple[str, bytes]] = []
    with pdfplumber.open(pdf) as doc:
        total = len(doc.pages)
        pages = doc.pages if max_pages is None else doc.pages[:max_pages]
        for page in pages:
            pil = page.to_image(resolution=dpi).original
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images.append(("image/png", buf.getvalue()))
    return images, total
