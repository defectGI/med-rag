"""Find the (PDF, Markdown) pairs a benchmark run should score.

The parser pipeline writes one folder per document, each holding the source
`<name>.pdf` next to the parser's rendered `<name>.md` (see
`pipeline/run_parse_pipeline.py`). A benchmark target is therefore just a
folder; this module turns a path into the concrete list of documents to grade:

* a folder that itself contains one `.pdf` + one `.md`  -> a single document
* a folder of such folders                              -> a batch (one per subfolder)

Anything ambiguous (a subfolder with two PDFs, a stray `.md` with no PDF) is
skipped with a reason rather than guessed at, so a batch score can never be
quietly computed over the wrong file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One gradable unit: the source PDF and the Markdown to score against it."""

    doc_id: str
    pdf: Path
    md: Path
    # The parser's own IR JSON, if one sits next to the .md at <md-stem>.json
    # (the pipeline's own output layout puts it there -- see
    # pipeline/run_parse_pipeline.py / run_e2e_test.py). None for a hand-built
    # or otherwise IR-less folder; entirely optional, never required to grade.
    ir_json: Path | None = None


@dataclass(frozen=True)
class Skipped:
    """A folder that looked like a target but couldn't be resolved to a pair."""

    where: Path
    reason: str


def _pair_in(folder: Path) -> Document | Skipped | None:
    """Resolve one folder to a Document, a Skipped (explained), or None.

    None means "this folder isn't a document target at all" (no PDF in it) --
    used to tell a plain container folder apart from a broken document folder.
    """
    pdfs = sorted(folder.glob("*.pdf"))
    mds = sorted(folder.glob("*.md"))
    if not pdfs:
        return None
    if len(pdfs) > 1:
        return Skipped(folder, f"{len(pdfs)} PDFs found, expected exactly one")
    pdf = pdfs[0]
    if not mds:
        return Skipped(folder, "a PDF but no .md to score")

    # Prefer the .md whose name matches the PDF (the pipeline's convention);
    # fall back to the sole .md, and refuse to guess when several are present.
    md = next((m for m in mds if m.stem == pdf.stem), None)
    if md is None:
        if len(mds) == 1:
            md = mds[0]
        else:
            return Skipped(folder, f"{len(mds)} .md files, none matching {pdf.stem}.pdf")
    ir_json = md.with_suffix(".json")
    return Document(doc_id=folder.name if folder.name else pdf.stem, pdf=pdf, md=md,
                    ir_json=ir_json if ir_json.is_file() else None)


def discover(root: Path) -> tuple[list[Document], list[Skipped]]:
    """Return (documents, skipped) for a benchmark root.

    Single-document folders and batch (folder-of-folders) layouts are both
    accepted; the caller decides how to present the results. `skipped` carries
    an explained entry for every folder that looked like a target but wasn't a
    clean pair, so nothing is dropped silently.
    """
    root = root.expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")

    # A PDF directly inside root -> treat root itself as the single document.
    direct = _pair_in(root)
    if isinstance(direct, Document):
        return [direct], []
    if isinstance(direct, Skipped):
        return [], [direct]

    # Otherwise fan out over immediate subfolders (batch layout).
    documents: list[Document] = []
    skipped: list[Skipped] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        result = _pair_in(child)
        if isinstance(result, Document):
            documents.append(result)
        elif isinstance(result, Skipped):
            skipped.append(result)
        # None -> not a document folder, ignore silently

    if not documents and not skipped:
        raise FileNotFoundError(
            f"no PDF/Markdown pair found in {root} or its immediate subfolders")
    return documents, skipped
