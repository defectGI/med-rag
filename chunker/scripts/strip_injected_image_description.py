"""Strip an injected IMAGE DESCRIPTION (VLM `Image.description`) from a
generated chunk set — `[visual] inject_image_description = false`.

WHY A SEPARATE SCRIPT (when `strip_injected_ocr.py` exists): that script
reads the text to strip from `images[].<field>` and finds it verbatim in
the body. In this set that path is CLOSED — measured against
`chunker/storage/all_chunks.json`:

    image refs carrying OCR         : 1068
    image refs carrying description  :    0   <-- absent from structured channel

The set was produced BEFORE schema v6: descriptions were injected into the
body but never written to the structured field (the sibling script's
docstring warns about this). So reversal has to work from the text's OWN
PATTERN.

WHY WE STRIP: VLM descriptions do not separate from the chunk body's real
paragraph/table text and read like spec entries. From `specs.db`: **188
fact rows** were produced from such blocks — e.g. `PN1113 connector_types`
= "On the left side (labeled REAR PANEL), there are input ports for..."
(this is a photo description, not a spec), while the product's REAL
connector list lives in the same chunk's normal text. Same rationale as
`[images] inject_ocr = false`.

SAFETY — pattern-based stripping is only legitimate once these three
conditions are measured and verified (690-block scan):
  1. The block sits on a FULL BLOCK BOUNDARY (preceded by start of text or `\\n\\n`)
  2. The chunk's `images[]` list is NON-EMPTY
  3. The block opens with a pattern that could not be the document's own sentence
This script RE-CHECKS all three per block; blocks that fail are LEFT
ALONE and reported. No silent cuts.

TABLE description is OUT OF SCOPE (`The table outlines/lists...`): it
summarizes a table that already lives in the chunk — not commentary.
63 rows measured and deliberately left in place
(`[visual] inject_description` stays on).

Usage:
    python scripts/strip_injected_image_description.py <all_chunks.json> [--dry-run]

Backup: <file>.pre_desc_strip_<date>.json (except --dry-run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

# Opening pattern for a VLM description. Kept narrow: the document's own
# text does not begin with "The image shows" (a spec text does not
# describe images in third person). Add new openers here if observed —
# RE-RUN the conditions-2/3 scan BEFORE broadening.
OPENERS = re.compile(
    r"^The image (?:shows|depicts|displays|is|contains|presents)\b",
    re.MULTILINE,
)


def _blok_sinirinda(text: str, i: int) -> bool:
    """Is `i` the start of a block: start of text or right after `\\n\\n`?

    `render_blocks` joins every chunk with `\\n\\n`, so an injected
    description is ALWAYS at a block boundary. Without this check, a
    sentence inside the document's own text could be cut."""
    return i == 0 or text[i - 2:i] == "\n\n"


def strip_image_descriptions(text: str, *, has_image_ref: bool) -> tuple[str, int, int]:
    """Returns (new_text, stripped, skipped).

    Block = from opener pattern to the next `\\n\\n`. When `has_image_ref`
    is False NOTHING is stripped (condition 2) — in a chunk with no image
    ref that sentence could be the document's own text."""
    cikarilan = atlanan = 0
    while True:
        m = OPENERS.search(text)
        if m is None:
            break
        i = m.start()
        if not has_image_ref or not _blok_sinirinda(text, i):
            atlanan += 1
            # Advance the search so we don't re-hit the same block.
            nxt = OPENERS.search(text, m.end())
            if nxt is None:
                break
            # To keep the loop simple, break out for skipped blocks (0
            # observed in the measured scan).
            break
        j = text.find("\n\n", m.end())
        j = len(text) if j < 0 else j + 2
        text = text[:i] + text[j:]
        cikarilan += 1
    return text, cikarilan, atlanan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="don't write, only report")
    a = ap.parse_args(argv)

    data = json.loads(a.path.read_text(encoding="utf-8"))
    docs = data["documents"]
    doc_iter = docs.values() if isinstance(docs, dict) else docs

    n_blok = n_cik = n_atla = n_node = 0
    bosalan: list[str] = []
    atlanan_ids: list[str] = []

    for doc in doc_iter:
        for node in doc["nodes"]:
            metin = node["text"]
            if not OPENERS.search(metin):
                continue
            n_blok += len(OPENERS.findall(metin))
            yeni, cik, atla = strip_image_descriptions(
                metin, has_image_ref=bool(node.get("images")))
            n_cik += cik
            n_atla += atla
            if atla:
                atlanan_ids.append(node["node_id"])
            if not cik:
                continue
            yeni = yeni.strip("\n")
            if not yeni.strip():
                # Description was this node's ONLY content source — emptying
                # it deletes real information. Node is LEFT ALONE (same
                # principle as the sibling script), reported.
                bosalan.append(node["node_id"])
                n_cik -= cik
                continue
            n_node += 1
            node["text"] = yeni
            # Schema v5+: a node declares the hash of its own text — if
            # text changed the declaration must change too, otherwise
            # the `facts/` pass-1 cross-check breaks.
            if "content_sha256" in node:
                node["content_sha256"] = hashlib.sha256(yeni.encode("utf-8")).hexdigest()

    print(f"image-description blocks      : {n_blok}")
    print(f"stripped                      : {n_cik}")
    print(f"skipped (condition unmet)     : {n_atla}")
    print(f"chunks whose text changed     : {n_node}")
    if bosalan:
        print(f"description was sole content (left alone): {len(bosalan)}")
        for nid in bosalan[:5]:
            print(f"  {nid}")
    if atlanan_ids:
        print("SKIPPED nodes (inspect manually):")
        for nid in atlanan_ids[:10]:
            print(f"  {nid}")

    if a.dry_run:
        print("\n--dry-run: file was not written")
        return 0
    if not n_node:
        print("\nNo changes: file was not written.")
        return 0

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    yedek = a.path.with_name(f"{a.path.name}.pre_desc_strip_{stamp}.json")
    shutil.copy2(a.path, yedek)
    a.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {a.path}\nbackup   -> {yedek}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())