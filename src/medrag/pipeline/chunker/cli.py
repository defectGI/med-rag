"""End-to-end entrypoint: IR JSON folder → chunk JSON folder.

Usage: ``python -m medrag.pipeline.chunker`` — configured via environment
variables (env/.env). The SOLE exception is the ``--limit N`` flag: process
at most ``N`` IRs (first N in discovery order; for trial/sampling). This
is a "how much to run" control, not a persistent setting — hence a flag
rather than env.

    CHUNKER_INPUT_DIR   required — root for IR JSONs (scanned recursively)
    CHUNKER_OUTPUT_DIR  default ./storage — output ROOT; under it two JSON
                        files: all_chunks.json (chunk sets for all docs),
                        all_combined.json (wrapped form of the same) +
                        viz/ (HTML, separate). Layout is defined in one
                        place: `chunker/layout.py`. RAPTOR was removed —
                        only leaf chunks are produced.
    CHUNKER_CONFIG      optional — partial override TOML applied on top of
                        default.toml
    CHUNKER_VIZ         on | off (default on) — for each document, produce
                        an interactive offline HTML tree
                        (`viz/{doc_id}.tree.html`) and an `viz/index.html`
                        linking them. Path is printed at the end of the
                        log; open in a browser to click through nodes and
                        see text/metadata. Visualization does not affect
                        chunk production.
    CHUNKER_PROGRESS    on | off (default on) — terminal progress bar over
                        the document loop (tqdm; if installed and stderr
                        is a real terminal, otherwise silently no-op).
    TOKENIZER           see tokenization/factory.py (default cl100k_base)

Error model: per-document failures are logged and the run continues with
the others; exit code 0 = all OK, 1 = at least one document failed,
2 = configuration error (including missing/incorrect env or no IR found —
an empty input folder almost always means a wrong path, not a silent
"0 processed" success).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from medrag.pipeline.chunker import CHUNKER_VERSION
from medrag.pipeline.chunker.adapters import first_parse
from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.chunk import (
    ChunkProvenance,
    ChunkSet,
    SourceProvenance,
)
from medrag.pipeline.chunker.core.engine import chunk_document
from medrag.pipeline.chunker.enrichment.cross_ref import resolve_cross_refs
from medrag.pipeline.chunker.layout import DEFAULT_OUTPUT_ROOT, OutputLayout
from medrag.pipeline.chunker.tokenization.factory import get_tokenizer
from medrag.pipeline.chunker.viz import (
    IndexEntry,
    write_document_html,
    write_index_html,
)

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _redirect_tqdm
except ImportError:  # optional dependency -- bar just turns off
    _tqdm = None
    _redirect_tqdm = None

log = logging.getLogger("chunker.cli")

_KAPALI = ("0", "off", "false", "no", "hayir", "hayır")


def _viz_acik(env: Mapping[str, str]) -> bool:
    """CHUNKER_VIZ value (default on); only an explicit "off" value
    (0/off/false/no — and the Turkish "hayır"/"hayir" in `_KAPALI`)
    disables visualization."""
    return (env.get("CHUNKER_VIZ") or "on").strip().lower() not in _KAPALI


def _progress_acik(env: Mapping[str, str]) -> bool:
    """CHUNKER_PROGRESS value (default on) + tqdm installed + stderr is a
    real terminal — if any of the three is missing the bar silently becomes
    a no-op (so \\r control chars do not end up in redirected logs/CI)."""
    if (env.get("CHUNKER_PROGRESS") or "on").strip().lower() in _KAPALI:
        return False
    try:
        return _tqdm is not None and sys.stderr.isatty()
    except Exception:  # noqa: BLE001 -- replaced/broken stderr = no TTY
        return False


def _ir_veri(path: Path) -> dict | None:
    """Top-level dict if the file is IR, else None. Check is content-based:
    int `ir_version` + list `blocks` at the top level (not a name pattern)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if (isinstance(data, dict) and isinstance(data.get("ir_version"), int)
            and isinstance(data.get("blocks"), list)):
        return data
    return None


def kesfet(input_dir: Path) -> tuple[list[tuple[Path, dict]], int]:
    """(path, data) pairs for IR JSONs under `input_dir`; second value is the
    count of skipped .json files. Order is deterministic (path order)."""
    bulunan: list[tuple[Path, dict]] = []
    atlanan = 0
    for path in sorted(input_dir.rglob("*.json")):
        data = _ir_veri(path)
        if data is None:
            atlanan += 1
            log.info("skipped (not IR): %s", path)
        else:
            bulunan.append((path, data))
    return bulunan, atlanan


def _simdi() -> str:
    """Offset local ISO-8601 — repo-wide single timestamp standard."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _provenance_for(doc) -> ChunkProvenance:
    """Provenance this IR + this run's config will produce.

    The source identifier is read from the adapter boundary
    (`doc.metadata`) — not from the raw dict: the adapter migrates the IR
    to the current schema first; we want to record the version the chunks
    were ACTUALLY PRODUCED under. A v7 file on disk may produce v9
    semantics; when IR_VERSION bumps to 10 and migration changes, this
    field changes too and the staleness gate regenerates the set. Had we
    recorded the file's own version, that change would silently go stale
    — exactly what the staleness gate is designed to prevent.

    The staleness gate uses the same value: when the existing file's
    provenance matches `same_inputs` to this one, the file is current.
    `raptor` is always None (RAPTOR was removed)."""
    return ChunkProvenance(
        chunker_version=CHUNKER_VERSION,
        generated_at=_simdi(),
        source=SourceProvenance(
            raw_sha256=doc.metadata.get("raw_sha256"),
            parser_version=doc.metadata.get("parser_version"),
            ir_version=doc.metadata.get("ir_version")))


def _set_dict(chunk_set: ChunkSet) -> dict:
    """JSON-compatible form of a ChunkSet for embedding into the combined
    output files."""
    return chunk_set.model_dump(mode="json", exclude_none=True)


def _archive_existing_output(out: OutputLayout) -> None:
    """Before a new run writes, the existing output triple (if any) is NOT
    DELETED — it is moved into a dated archive folder. The archive key is
    the previous run's own `generated_at` (or the file's mtime if that
    can't be read), so the folder name says which run produced it. No-op
    if `all_chunks.json` is absent (first run)."""
    if not out.all_chunks_file.exists():
        return
    damga = None
    try:
        eski = json.loads(out.all_chunks_file.read_text(encoding="utf-8"))
        damga = eski.get("generated_at")
    except (OSError, json.JSONDecodeError):
        pass
    if not damga:
        damga = (datetime.fromtimestamp(out.all_chunks_file.stat().st_mtime)
                 .astimezone().isoformat(timespec="seconds"))
    klasor = out.root / "archive" / damga.replace(":", "-")
    klasor.mkdir(parents=True, exist_ok=True)
    for dosya in (out.all_chunks_file, out.all_combined_file):
        if dosya.exists():
            dosya.replace(klasor / dosya.name)
    log.info("previous run archived (not deleted): %s", klasor)


def calistir(env: Mapping[str, str] = os.environ, limit: int | None = None) -> int:
    """The full run; `env` is injectable (tests pass a dict). If `limit` is
    given, only the first `limit` IRs in discovery order are processed
    (>=1; `main` forwards from --limit). Returns the exit code
    (0/1/2 — see error model in the module docstring)."""
    girdi_ham = env.get("CHUNKER_INPUT_DIR")
    if not girdi_ham:
        log.error("CHUNKER_INPUT_DIR is not set (see .env.example)")
        return 2
    input_dir = Path(girdi_ham)
    if not input_dir.is_dir():
        log.error("CHUNKER_INPUT_DIR is not a directory: %s", input_dir)
        return 2

    try:
        cfg = load_config(env.get("CHUNKER_CONFIG") or None)
        tokenizer = get_tokenizer(env.get("TOKENIZER"))
    except Exception as exc:  # noqa: BLE001 -- config/tokenizer load failure is logged at error and surfaced as exit 2
        log.error("failed to load configuration: %s", exc)
        return 2

    bulunan, atlanan = kesfet(input_dir)
    if not bulunan:
        log.error("no IR JSONs found under %s (%d .json skipped) — "
                  "is CHUNKER_INPUT_DIR correct?", input_dir, atlanan)
        return 2

    if limit is not None and limit < len(bulunan):
        log.info("--limit %d: processing the first %d of %d IRs",
                 limit, limit, len(bulunan))
        bulunan = bulunan[:limit]

    # Output root + viz folder. The two JSON files are written at the END
    # of the run (write_outputs) — ChunkSets only live in memory during
    # this loop; per-document files are no longer written.
    out = OutputLayout(env.get("CHUNKER_OUTPUT_DIR") or DEFAULT_OUTPUT_ROOT).ensure()
    viz = _viz_acik(env)
    progress = _progress_acik(env)
    dongu = (_tqdm(bulunan, desc="chunking", unit="doc", dynamic_ncols=True,
                   leave=False) if progress else bulunan)
    # While the tqdm bar is on the screen, a plain `log.info`/`log.exception`
    # (writes to stderr) would overwrite the bar — the redirect temporarily
    # routes logs above the bar (tqdm's own standard solution, no extra
    # dependency).
    ctx = _redirect_tqdm() if progress else nullcontext()
    islenen: dict[str, Path] = {}  # doc_id -> source file (collision detection)
    viz_girdileri: list[IndexEntry] = []
    doc_setler: dict[str, ChunkSet] = {}  # doc_id -> ChunkSet (embedded in all_chunks.json)
    hatali = 0
    with ctx:
        for path, data in dongu:
            try:
                doc = first_parse.adapt_dict(data, config=cfg)
                if doc.doc_id in islenen:
                    raise ValueError(
                        f"doc_id {doc.doc_id!r} already processed in this run "
                        f"({islenen[doc.doc_id]}) — output would be silently overwritten")
                if doc.doc_id.startswith("_"):
                    raise ValueError(
                        f"doc_id {doc.doc_id!r} cannot start with '_' — that "
                        f"prefix is reserved for scope identifiers")
                beklenen = _provenance_for(doc)
                chunk_set = chunk_document(doc, cfg, tokenizer)
                chunk_set = resolve_cross_refs(chunk_set, doc)
                chunk_set.provenance = beklenen
                log.info("%s processed (%d leaf, %d nodes)", path.name,
                         len(chunk_set.leaves()), len(chunk_set.nodes))
                doc_setler[doc.doc_id] = chunk_set
                islenen[doc.doc_id] = path
                if viz:
                    # Visualization is best-effort: if a document's HTML
                    # build fails the chunks are already in memory, the
                    # run doesn't fall over.
                    try:
                        html = write_document_html(chunk_set, out.viz)
                        viz_girdileri.append(IndexEntry(
                            doc_id=doc.doc_id, html_name=html.name,
                            leaf_count=len(chunk_set.leaves()),
                            node_count=len(chunk_set.nodes)))
                    except Exception:
                        log.exception("could not produce visualization: %s", doc.doc_id)
            except Exception:
                log.exception("failed to process: %s", path)
                hatali += 1

    # Two fixed files (RAPTOR/all_raptor.json was removed): all_chunks.json,
    # all_combined.json. Write order is intentionally write-then-replace
    # to a separate TEMP file (so a mid-write interruption leaves the old
    # pair intact and uncorrupted) — but only AFTER both temp files are
    # complete do we archive the previous run (it is NOT deleted) and
    # move the temp files into their final locations with `os.replace`.
    # This shrinks the "neither old nor new exists" window to a few rename
    # calls (not the json.dump time proportional to data size) — previously
    # these two files were opened directly with `open(..., "w")` at the
    # final path and archiving happened BEFORE writing, so an interruption
    # lost both old and new.
    documents_dict = {doc_id: _set_dict(cs) for doc_id, cs in doc_setler.items()}
    simdi = _simdi()
    payload = {"generated_at": simdi, "documents": documents_dict}
    tmp_chunks_fd, tmp_chunks_path = tempfile.mkstemp(
        dir=str(out.root), prefix=".tmp_all_chunks_", suffix=".json")
    tmp_combined_fd, tmp_combined_path = tempfile.mkstemp(
        dir=str(out.root), prefix=".tmp_all_combined_", suffix=".json")
    try:
        with os.fdopen(tmp_chunks_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with os.fdopen(tmp_combined_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except BaseException:  # includes KeyboardInterrupt/SystemExit — cleanup
        # must run on Ctrl+C/kill too, so use BaseException not Exception.
        for tmp in (tmp_chunks_path, tmp_combined_path):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise

    _archive_existing_output(out)
    os.replace(tmp_chunks_path, out.all_chunks_file)
    log.info("%s → %d documents", out.all_chunks_file, len(documents_dict))
    os.replace(tmp_combined_path, out.all_combined_file)
    log.info("%s → %d documents", out.all_combined_file, len(documents_dict))

    log.info("done: %d processed, %d skipped, %d failed (tokenizer=%s)",
             len(islenen), atlanan, hatali, tokenizer.name)
    if viz and viz_girdileri:
        indeks = write_index_html(viz_girdileri, out.viz)
        log.info("visualization ready → %s (open in browser: file:///%s)",
                 indeks, indeks.resolve().as_posix())
    return 1 if hatali else 0


def _parse_argv(argv: list[str] | None) -> int | None:
    """Parse the `--limit N` flag; None if absent. N>=1 required (0/negative
    is an argparse error → exit code 2)."""
    import argparse

    def _pozitif(deger: str) -> int:
        n = int(deger)
        if n < 1:
            raise argparse.ArgumentTypeError("must be at least 1")
        return n

    parser = argparse.ArgumentParser(
        prog="python -m medrag.pipeline.chunker",
        description="Convert an IR JSON folder to chunk JSON "
                    "(configured via env/.env; see .env.example).")
    parser.add_argument(
        "--limit", type=_pozitif, metavar="N", default=None,
        help="process at most N IRs (first N in discovery order; for trial runs)")
    return parser.parse_args(argv).limit


def main(argv: list[str] | None = None) -> int:
    """``python -m medrag.pipeline.chunker`` entry: log + argv + .env, then `calistir`."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    limit = _parse_argv(argv)
    from dotenv import load_dotenv
    # The package is now installed via `pip install -e .`; the caller's cwd
    # is no longer guaranteed to be the chunker's own directory, so `.env`
    # is located explicitly relative to __file__ (previously the
    # no-argument `load_dotenv()` relied on cwd).
    load_dotenv(Path(__file__).resolve().parent / ".env")  # no-op if .env missing; does not overwrite a real env var
    return calistir(os.environ, limit=limit)