"""Strip OCR text injected into chunk bodies from a generated chunk set
(`all_chunks.json`). Decisions: `[images] inject_ocr = false`.

Why this is a migration: `render_image` no longer injects OCR into the body,
but on-disk sets were produced under the old behavior. Re-chunking
(re-parse + re-chunk) is expensive on this machine; injection is REVERSIBLE
(the OCR text is preserved verbatim in `ChunkNode.images[].ocr_text`), so
removing it from the body by exact match is sufficient.

Scope is deliberately narrow: only `ocr_text`. `alt_text`/`description`
stay in the body (deliberate) — before schema v6 `description` was not
carried in the structured channel, so reversing those is not possible for
every set.

Safety: a single occurrence is removed per `ocr_text` (if an image's OCR
overlaps exactly with a real table in the chunk, the table survives); only
the separator at the cut point is cleaned up (no global whitespace
normalization — the rest of the body is preserved bit-for-bit).

Usage:
    python scripts/strip_injected_ocr.py <all_chunks.json> [--dry-run]

Backup: written as <file>.pre_ocr_strip_<date>.json (except --dry-run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Label that `render_image` puts in front of unverified OCR
# (kept identical to `core/markdown.py::_OCR_UNVERIFIED_LABEL`; duplicated so
# this script can run independently — if it drifts, only the label line is
# left behind, the OCR body is still cleaned).
OCR_LABEL = "*(OCR, doğrulanmamış):*"

# Fields that could be injected into the body AND that survive verbatim in
# the structured channel — only these can be stripped. `alt_text` is
# deliberately OUT: it is short and the document's own label, intended to
# stay in the body.
_ALANLAR = ("ocr_text", "description")


def _blok_sinirinda(text: str, i: int, j: int) -> bool:
    """Is the [i, j) range in the body an ENTIRE BLOCK: preceded by start of
    text or `\\n\\n`, followed by end of text or `\\n\\n`.

    This check is required: `render_blocks`/`render_image` joins every chunk
    with `\\n\\n`, so injected OCR is ALWAYS at a block boundary. A lax
    (bare `find`) match would, for short OCR like "STUB" or "BUS", chop a
    real piece of the document's own text."""
    return ((i == 0 or text[i - 2:i] == "\n\n")
            and (j == len(text) or text[j:j + 2] == "\n\n"))


def _strip_one(text: str, ocr: str) -> tuple[str, bool]:
    """Remove the single block-boundary occurrence of `ocr` (with its label
    line if any) from `text`. Returns (new_text, removed)."""
    ocr = ocr.strip()
    if not ocr:
        return text, False
    for aday in (f"{OCR_LABEL}\n{ocr}", ocr):
        i = text.find(aday)
        while i != -1:
            j = i + len(aday)
            if _blok_sinirinda(text, i, j):
                # Also take the block separator at the cut: take the one
                # before first, else the one after (taking both would merge
                # neighbors).
                if text[max(0, i - 2):i] == "\n\n":
                    i -= 2
                elif text[j:j + 2] == "\n\n":
                    j += 2
                return text[:i] + text[j:], True
            i = text.find(aday, i + 1)
    return text, False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write, only report")
    ap.add_argument("--also-description", action="store_true",
                    help="also strip the image description "
                         "(`images[].description`) — migration for "
                         "`[visual] inject_image_description = false`. "
                         "OFF by default — legacy call shape is preserved. "
                         "TABLE description is OUT OF SCOPE for this script "
                         "(not carried in the structured channel, so it "
                         "cannot be reversed).")
    ap.add_argument("--force", action="store_true",
                    help="write even when the set already looks clean "
                         "(dangerous: see idempotency note)")
    ap.add_argument("--allow-empty", action="store_true",
                    help="also empty chunks whose body became completely "
                         "empty after OCR removal (default: the node is "
                         "RESTORED untouched — OCR was its only content "
                         "source, e.g. a brochure page printed as a full-"
                         "page image)")
    a = ap.parse_args(argv)

    data = json.loads(a.path.read_text(encoding="utf-8"))
    docs = data["documents"]
    if isinstance(docs, dict):
        doc_iter = docs.values()
    else:  # some sets are lists
        doc_iter = docs

    toplam_ocr = cikarilan = bulunamayan = 0
    dokunulan_node = korunan = 0
    bosalan: list[str] = []
    kalinti: list[tuple[str, str]] = []

    for doc in doc_iter:
        for node in doc["nodes"]:
            metin = node["text"]
            once = metin
            for ref in node.get("images") or []:
                for alan in _ALANLAR:
                    if alan == "description" and not a.also_description:
                        continue
                    parca = (ref.get(alan) or "").strip()
                    if not parca:
                        continue
                    toplam_ocr += 1
                    metin, oldu = _strip_one(metin, parca)
                    if oldu:
                        cikarilan += 1
                    else:
                        bulunamayan += 1
            # If, after processing ALL of a node's images, an OCR copy still
            # sits AS A BLOCK (more copies than image references): leave it
            # alone and report it. A free substring inside the text (like
            # "STUB") is NOT a residue — it is the document's own text.
            for ref in node.get("images") or []:
                for alan in _ALANLAR:
                    if alan == "description" and not a.also_description:
                        continue
                    parca = (ref.get(alan) or "").strip()
                    if parca and _strip_one(metin, parca)[1]:
                        kalinti.append((node["node_id"], parca[:60]))
            metin = metin.strip("\n")
            if metin != once:
                if not metin.strip():
                    # OCR was this node's ONLY content source: emptying it
                    # deletes real information. Default: leave the node
                    # alone, report it.
                    bosalan.append(node["node_id"])
                    if not a.allow_empty:
                        cikarilan -= sum(
                            1 for r in (node.get("images") or [])
                            for alan in _ALANLAR
                            if (alan != "description" or a.also_description)
                            and (r.get(alan) or "").strip())
                        korunan += 1
                        continue
                dokunulan_node += 1
                node["text"] = metin
                # Schema v5+: a node declares the hash of its own text —
                # if text changed the declaration must change too,
                # otherwise the `facts/` pass-1 cross-check breaks. The
                # formula is fixed there as well:
                # sha256(text.encode("utf-8")), no normalize/trim.
                if "content_sha256" in node:
                    node["content_sha256"] = (
                        "sha256:"
                        + hashlib.sha256(metin.encode("utf-8")).hexdigest())

    print(f"Image refs carrying OCR       : {toplam_ocr}")
    print(f"stripped from body            : {cikarilan}")
    print(f"not found in body             : {bulunamayan}")
    print(f"chunks whose text changed     : {dokunulan_node}")
    etiket = "emptied" if a.allow_empty else f"LEFT ALONE ({korunan})"
    print(f"OCR was sole content (chunk would have emptied): {len(bosalan)} -> {etiket}")
    for nid in bosalan:
        print(f"  {nid}")
    if kalinti:
        print(f"WARNING: {len(kalinti)} leftover matches after stripping "
              f"(duplicate occurrence or collision with real text):")
        for nid, parca in kalinti[:10]:
            print(f"  {nid}: {parca!r}")

    # NOT idempotent: on an already-clean set, only collisions with the
    # document's REAL text remain (e.g. a "www.example.com" footer) — a
    # second run would delete them. So when the table shows "most OCR not
    # in body" the write is refused.
    zaten_temiz = bulunamayan > cikarilan
    if zaten_temiz:
        print("\nSTOPPED: most OCR (%d/%d) is not in the body — this set "
              "looks already clean. Re-running could delete real document "
              "text." % (bulunamayan, toplam_ocr))
        if not a.force:
            return 1
        print("--force: writing anyway")

    if not dokunulan_node:
        print("\nNo changes: file was not written (and no backup was created).")
        return 0

    if a.dry_run:
        print("\n--dry-run: file was not written")
        return 0

    damga = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yedek = a.path.with_suffix(f".pre_ocr_strip_{damga}.json")
    yedek.write_text(a.path.read_text(encoding="utf-8"), encoding="utf-8")
    a.path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nbackup  : {yedek}")
    print(f"written : {a.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())