"""Glue script: runs `parser` over the documents listed in corpus's
`document_nodes.json` and writes the result back into each record's `parse`
block.

Neither `medrag.pipeline.parser` nor `chatbot-corpus` import from each other or
from this script. This is the only file that knows about both; it imports
`medrag.pipeline.parser` as a real package (D-39 Faz B, editable install --
no more sys.path hack) and reads/writes corpus's `document_nodes.json`
directly.

Documents run in THREE PHASES by model affinity. VLM_CLASSIFY can now be
configured as a genuinely different model from VLM (see
parser/images/visual_classify.py, parser/llm/__init__.py's
get_vlm_classify_client) -- with only two phases, classify calls were
interleaved with VLM calls throughout phase 1 (per image in
images/image_handler.py, and per document via pdf_parser.py's own
triage-then-read split), so a single-GPU Ollama swapped models constantly
instead of once per run, exactly the problem the two-phase split was
introduced to avoid for VLM vs LLM. So:

  * phase 0 classifies every document's visual regions ONLY (VLM_CLASSIFY):
    pdf_parser.py's own PdfParser.classify_only() pre-warms its table-region
    triage cache without reading a single page; every other format calls
    parser_obj.parse() (cheap -- no model calls of its own) then
    images.image_handler.handle_images(doc, classify_only=True) to classify
    its plain images. No checkpoint needed here: classify_crop's own on-disk
    cache (keyed by crop sha256, see visual_classify.py) already makes this
    step idempotent/resumable on its own -- interrupting phase 0 and
    re-running it just re-hits already-classified crops for free.
  * phase 1 does every document's VLM-only work (parse -- including
    pdf_parser.py's triage, which now hits phase 0's warm classify cache
    instead of calling VLM_CLASSIFY again -- + image OCR, LLM check deferred
    via handle_images(raw_sink=...), + describe_blocks(stage="vlm") for
    charts/diagrams/drawings),
  * phase 2 does every document's LLM-only work (apply_ocr_checks +
    describe_blocks(stage="llm") for tables).

A PDF's own plain images (full-page renders, unpaired figures) are only
discovered once phase 1's page read has already run, so those specific
crops still classify inside phase 1, not phase 0 -- unavoidable without
restructuring pdf_parser.py's fused triage/page-read, but a small remainder
next to what phase 0 now covers up front (every table-region candidate in
every PDF, and every plain image in every non-PDF document).

Progress is preserved at two levels:
  * document_nodes.json is atomically re-written after every document (as
    before), so completed documents never re-parse (parsed_from_hash).
  * phase 1 checkpoints each document atomically to its group dir's
    <doc_id>.stage1.json (tagged with the source content_hash so a stale
    checkpoint of an older file version is ignored). After a crash/Ctrl-C,
    a re-run skips the phase-1 work of every document whose checkpoint
    exists and redoes only phase 2 for them. The checkpoint is deleted once
    the final IR is saved. Phase 0 has no checkpoint of its own (see above).

Output layout mirrors run_e2e_test.py: one folder per document under
PARSED_OUTPUT_DIR keeps the raw copy, IR and Markdown together --
    PARSED_OUTPUT_DIR/<doc_type>_<doc_id>/
        <original file name>    <- raw copy of the source
        <doc_id>.json            <- parsed IR (this is parse.parsed_json_path)
        <doc_id>.md              <- rendered Markdown
(no NN_ index prefix: incremental runs would renumber; the folder name must
be stable across runs). SKIPPED formats (.stp/.png/...) get no folder.

A file whose extension has no registered parser (.stp, .png, ...) is marked
SKIPPED rather than FAILED -- that mirrors the "deliberately not parsed"
case in document_node_schema.md, not an error.

Settings are read from the .env file (see .env.example):
    PARSER_DIR           root of medrag.pipeline.parser (used to load its own .env)
    BELGELER_DIR          root that document_nodes.json's rel_path values are relative to
    DOCUMENT_NODES_PATH  full path to document_info/document_nodes.json
    PARSED_OUTPUT_DIR    folder the generated IR JSON files are written to
    DOC_TYPES            optional comma-separated doc_type filter (e.g.
                         "DATASHEET,BROCHURE"); unset/empty = every type

The one CLI flag is `--limit N`: parse at most N of the documents that need
parsing (the first N in registry order) and leave the rest pending. It is a
"how much of the corpus should I run right now" control -- a small trial run
before committing the GPU to the whole corpus -- not a persistent setting, so
it is a flag rather than an env var (CONFIG.md's taxonomy puts run input on
the CLI; same reasoning as chunker's own `--limit`). It never changes WHICH
documents are eligible (that's DOC_TYPES) and never truncates the registry:
every record is still written back, the unparsed ones simply stay PENDING and
the next run picks them up.

Concurrency (all optional, configured in parser/config -- see images/
image_handler.py and describe/core.py):
    DOC_CONCURRENCY      number of documents processed at once (default 4)
    IMAGE_CONCURRENCY    number of images OCR'd in parallel within a document
    DESCRIBE_CONCURRENCY number of describable blocks (tables, then visuals)
                         processed in parallel within a document (config.py's
                         `[describe] concurrency`)

Running:
    python run_parse_pipeline.py              # the whole pending corpus
    python run_parse_pipeline.py --limit 3    # a 3-document trial run

This script is one stage of the corpus chain; `run_full_corpus.py` runs the
whole chain (scan -> parse -> chunk) in order.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Paths in .env are relative to the folder this file lives in
# (src/medrag/pipeline/cli/, D-58 moved it here from repo-root pipeline/) --
# resolved against BASE_DIR instead of CWD so this doesn't break no matter
# where the repo gets moved to or where the script is run from. An absolute
# path (the old behavior) is used as-is.
BASE_DIR = Path(__file__).resolve().parent


def _bootstrap() -> None:
    """Resolve .env-driven paths/settings into module globals. Not called at
    import time (D-43/K-45) -- call it explicitly before touching PARSER_DIR,
    BELGELER_DIR, DOCUMENT_NODES_PATH, PARSED_OUTPUT_DIR or DOC_TYPES. main()
    calls it first thing; tests call it directly after monkeypatch.setenv(...)
    instead of reimporting the whole module."""
    global PARSER_DIR, BELGELER_DIR, DOCUMENT_NODES_PATH, PARSED_OUTPUT_DIR, DOC_TYPES

    load_dotenv(BASE_DIR / ".env")

    def _resolve(env_var: str) -> str:
        return str((BASE_DIR / os.environ[env_var]).resolve())

    PARSER_DIR = _resolve("PARSER_DIR")
    BELGELER_DIR = _resolve("BELGELER_DIR")
    DOCUMENT_NODES_PATH = _resolve("DOCUMENT_NODES_PATH")
    PARSED_OUTPUT_DIR = _resolve("PARSED_OUTPUT_DIR")

    # parser's own .env carries the LLM/VLM/storage settings; it's loaded before
    # importing the parser modules so that get_client()/get_vlm_client() see
    # these settings regardless of the working directory.
    load_dotenv(Path(PARSER_DIR) / ".env")

    # Optional doc_type filter, e.g. "DATASHEET,BROCHURE" -- unset/empty parses
    # every type, same as before this existed. Known values (see
    # chatbot-corpus/document_info/document_node_schema.md): BROCHURE, CATALOGUE,
    # DATASHEET, CE_DECLARATION, TECHNICAL_DRAWING, USER_MANUAL,
    # QUICK_START_GUIDE, STP, PRODUCT_IMAGE (the last two have no parser --
    # selecting them just yields SKIPPED records, not an error). Read AFTER both
    # .env loads above (this package's own .env via load_dotenv(BASE_DIR / ".env"), parser/.env
    # here) so it works no matter which of the two files it was actually put in.
    DOC_TYPES = {t.strip().upper() for t in os.getenv("DOC_TYPES", "").split(",") if t.strip()}


# parser is now a real package (under src/medrag/pipeline/parser, D-39 Phase B);
# no sys.path hack / unqualified import needed.
from medrag.pipeline.cli.pipeline_config import (
    get_config as _pipeline_config,
)
from medrag.pipeline.parser import progress
from medrag.pipeline.parser.config import get_config as _parser_config
from medrag.pipeline.parser.describe.core import describe_blocks
from medrag.pipeline.parser.images.image_handler import (
    apply_ocr_checks,
    handle_images,
)
from medrag.pipeline.parser.llm import (
    LLMError,
    describe_configured_models,
    get_client,
    get_vlm_client,
    vlm_classify_configured,
)
from medrag.pipeline.parser.llm.health import probe_llm, probe_vlm
from medrag.pipeline.parser.parsers.base import (
    PARSER_VERSION,
    ParsedDocument,
    migrate_dict,
    sha256_id,
)
from medrag.pipeline.parser.parsers.registry import (
    UnsupportedFormatError,
    parser_for,
)
from medrag.pipeline.parser.render.markdown import to_markdown

PARSER_LABEL = "run_parse_pipeline.py"
# PARSER_VERSION is now defined in the parser package and imported here
# (PROTOCOL.md KARAR-006): _needs_parse's re-parse gate, the parser_version
# written into the IR, and chunk provenance all see the same single value.


def _now() -> str:
    # Local system time WITH utc offset (e.g. ...+03:00): folder names and
    # log lines match the wall clock, yet the stamp stays unambiguous if
    # document_nodes.json is ever read on a machine in another timezone.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _model_label(*env_names: str) -> str | None:
    for name in env_names:
        val = os.getenv(name)
        if val:
            return val
    return None


def _atomic_write_json(path: str, data: dict) -> None:
    """Write-then-replace so a crash mid-run never leaves a truncated/corrupt
    document_nodes.json -- this file is hundreds of records and re-scanning
    isn't free."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _needs_parse(record: dict) -> bool:
    if not record.get("is_active", True):
        return False
    parse = record["parse"]
    if parse.get("parsed_from_hash") != record["scan"]["content_hash"]:
        return True
    # Same bytes, but parsed by an older parser: the output on disk no longer
    # reflects what the current code would produce, so it must be redone.
    # SKIPPED records are exempt -- no parser ran on them (unsupported
    # format), so no parser change can affect their (non-)output.
    if parse.get("status") == "SKIPPED":
        return False
    return parse.get("parser_version") != PARSER_VERSION


def _group_dir(record: dict) -> Path:
    """Per-document output folder (raw copy + IR + Markdown together, like
    run_e2e_test.py's layout). No index prefix: incremental runs would
    renumber, and the name must be stable for resume to find checkpoints."""
    doc_type = record.get("doc_type", "UNKNOWN")
    return Path(PARSED_OUTPUT_DIR) / f"{doc_type}_{record['identity']['doc_id']}"


# --- model health ------------------------------------------------------------
#
# "Server reachable" is not "model usable": an endpoint can return HTTP 200
# with empty or non-contract replies for every single page (hidden-thinking
# budget burn, truncated multimodal prompts, a router serving the wrong
# model...). The July 2026 full run lost 509 pages exactly this way and the
# outputs alone couldn't say why. Two cheap defenses, both env-tunable:
#
#   HEALTH_CHECK=0        skip the known-answer probes entirely
#   HEALTH_FAIL_STREAK=N  consecutive fully-VLM-failed documents that trigger
#                         a mid-run re-probe (default 3)
#
# A failed probe aborts the run: parsing hundreds of documents against a
# broken model only produces silent image-fallbacks that all need re-parsing.


def _health_on() -> bool:
    return _pipeline_config().health.enabled


#: Types NOT requiring VLM-vision (see the `[describe]` note in parser
#: config/default.toml): tables are produced from cells via LLM, charts
#: deterministically from their own XML. If this list grows, `_vlm_work_configured()`
#: runs a needless VLM canary -- the fail-safe side is this.
_VLMSIZ_DESCRIBE_TIPLERI = frozenset({"table", "chart"})


def _vlm_work_configured() -> bool:
    """Does this run actually put any work on the VLM client.

    The canary's purpose is to avoid "silently turning hundreds of documents
    into image-fallbacks with a broken model" (see the block above). If no call
    goes to the VLM that risk does not exist -- but the probe still ran and
    stopped the run. We set up the pattern `vlm_classify_configured()` uses for
    phase 0 (line ~633) for the VLM role too.

    Conservative: if ANY of the four listed consumers is on, the probe runs.
    With `describe.enabled` on, `types` can only be table/chart (neither uses
    VLM) but we do not try to resolve that here -- falling to the wrong side
    would mean not noticing a broken VLM."""
    cfg = _parser_config()
    return bool(
        cfg.pdf.vlm              # page reading (hybrid/scanned pages)
        or cfg.image_ocr.enabled  # image OCR
        # note: the strategy differs BY TYPE -- `table` is generated from cells
        # via LLM, `chart` deterministically from its own XML (both VLM-free);
        # every other type goes to VLM-vision. So if only table/chart are in the
        # list, no work falls to the VLM.
        or bool(cfg.describe.enabled
                and set(cfg.describe.types) - _VLMSIZ_DESCRIBE_TIPLERI)
        # classification uses the VLM_CLASSIFY client, which falls back to the
        # VLM's own client if it is not configured (get_vlm_client("classify"))
        or cfg.visual.classify
    )


def _probe_or_abort(role: str) -> None:
    """Known-answer probe for one role ("VLM", "VLM_CLASSIFY", or "LLM");
    raises on failure.

    An unconfigured role (get_*_client raising LLMError at build time) is
    fine — the parser degrades deliberately without it; only a configured
    endpoint that can't answer the canary is fatal. "VLM_CLASSIFY" is probed
    with the same generic transcription canary as "VLM" (probe_vlm) — it only
    needs to confirm the endpoint is up and answers sanely, not exercise
    classify_crop's actual prompt, same as every other role's canary here."""
    is_vlm_role = role in ("VLM", "VLM_CLASSIFY")
    try:
        client = get_vlm_client("classify") if role == "VLM_CLASSIFY" \
            else get_vlm_client() if role == "VLM" else get_client()
    except LLMError as exc:
        progress.write(f"health: {role} not configured ({exc}); probe skipped")
        return
    ok, detail = (probe_vlm if is_vlm_role else probe_llm)(client)
    if ok:
        progress.write(f"health: {role} canary PASS" + (f" ({detail})" if detail else ""))
        return
    raise RuntimeError(
        f"health: {role} canary FAILED — endpoint is up but unusable: {detail}\n"
        f"Fix the model/server before parsing (set HEALTH_CHECK=0 to override).")


class _VlmHealthBreaker:
    """Trips after HEALTH_FAIL_STREAK consecutive documents whose every VLM
    page read failed (per metadata["vlm_health"]), then re-probes the canary:
    canary dead -> abort the run; canary alive -> loud warning (the failures
    are page-specific, not systemic) and the streak restarts."""

    def __init__(self) -> None:
        self.threshold = max(1, _pipeline_config().health.fail_streak)
        self._streak = 0
        self._lock = threading.Lock()

    def record(self, vlm_health: dict | None) -> None:
        if not vlm_health:  # no VLM pages in this document: no signal
            return
        attempts = sum(vlm_health.values())
        failed_all = attempts > 0 and vlm_health.get("ok", 0) == 0
        with self._lock:
            self._streak = self._streak + 1 if failed_all else 0
            if self._streak < self.threshold:
                return
            self._streak = 0
            kinds = {k: v for k, v in vlm_health.items() if k != "ok"}
            progress.write(f"health: {self.threshold} consecutive documents with every "
                           f"VLM page read failed (last doc: {kinds}); re-probing")
        _probe_or_abort("VLM")
        progress.write("health: canary still passes — failures look page-specific, "
                       "continuing (inspect vlm_error in the page metadata)")


def _stage1_path(record: dict) -> Path:
    return _group_dir(record) / f"{record['identity']['doc_id']}.stage1.json"


def phase0_classify_one(record: dict) -> None:
    """Classify-only pass: warms visual_classify.py's on-disk classification
    cache with VLM_CLASSIFY before VLM/LLM work runs (see module docstring).
    Never raises and never touches record["parse"] -- a failure here just
    means those crops get classified for the first time during phase 1
    instead (fail-soft; classify_crop itself is already fail-open when no
    classifier is configured/reachable). No checkpoint: classify_crop's own
    on-disk cache already makes re-running this idempotent."""
    doc_id = record["identity"]["doc_id"]
    src_path = Path(BELGELER_DIR) / record["location"]["rel_path"]
    if not src_path.is_file():
        return
    try:
        parser_obj = parser_for(src_path)
    except UnsupportedFormatError:
        return  # e.g. .stp/.png -- no parser, so nothing to classify

    try:
        classify_only = getattr(parser_obj, "classify_only", None)
        if classify_only is not None:
            classify_only(src_path, doc_id)  # e.g. PdfParser: triage-only, no page read
            return
        doc = parser_obj.parse(src_path, doc_id)  # cheap: no model calls of its own
        handle_images(doc, classify_only=True)
    except Exception as exc:  # noqa: BLE001
        progress.write(f"  !! phase 0 classify failed for {doc_id}, continuing: {exc!r}")


def phase1_one(record: dict) -> dict | None:
    """VLM-only half of one document: parse (incl. visual-region
    classification) + image classify/OCR (LLM check deferred), checkpointed
    atomically to <doc_id>.stage1.json. Mutates record["parse"]
    only on failure/skip (success is decided in phase 2). Never raises.
    Returns the document's metadata["vlm_health"] counters (None when the
    document produced no VLM signal) for the run-level health breaker."""
    doc_id = record["identity"]["doc_id"]
    src_path = Path(BELGELER_DIR) / record["location"]["rel_path"]
    content_hash = record["scan"]["content_hash"]
    parse = record["parse"]
    now = _now()

    if not src_path.is_file():
        parse.update(status="FAILED", error=f"file not found: {src_path}", last_parsed=now)
        return

    try:
        parser_obj = parser_for(src_path)
    except UnsupportedFormatError:
        # e.g. .stp / .png -- not a parser bug, this format has no text to extract.
        parse.update(parser=None, parser_version=None, status="SKIPPED",
                     parsed_from_hash=content_hash, parsed_json_path=None,
                     last_parsed=now, error=None, stages=None)
        return

    # Resume: a checkpoint from an interrupted run is trusted iff it was made
    # from THIS version of the file (content_hash match) -- existence alone is
    # enough integrity-wise because the write below is atomic.
    ckpt_path = _stage1_path(record)
    if ckpt_path.is_file():
        try:
            if json.loads(ckpt_path.read_text(encoding="utf-8")
                          ).get("content_hash") == content_hash:
                progress.write(f"  resume: checkpoint kept for {doc_id}")
                return
        except Exception:  # noqa: BLE001, S110 -- unreadable -> just redo phase 1
            pass

    try:
        doc = parser_obj.parse(src_path, doc_id)
        doc.raw_sha256 = sha256_id(src_path.read_bytes())

        raw_ocr: dict[str, str] = {}
        try:
            handle_images(doc, raw_sink=raw_ocr)
            ocr_status, ocr_error = "SUCCESS", None
        except Exception as exc:  # noqa: BLE001
            ocr_status, ocr_error = "FAILED", str(exc)

        # Describe the visual regions that have pixels (charts/diagrams/
        # drawings) while the VLM is the loaded model -- same model-affinity
        # reason the OCR half runs here. Tables are described in phase 2 by
        # the text LLM. Fail-soft: an undescribed visual keeps its type and
        # renders as a placeholder, exactly as before this pass existed.
        try:
            describe_blocks(doc, stage="vlm")
            vdesc_status, vdesc_error = "SUCCESS", None
        except Exception as exc:  # noqa: BLE001
            vdesc_status, vdesc_error = "FAILED", str(exc)

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        raw_copy = ckpt_path.parent / src_path.name
        if not raw_copy.is_file():  # keep the source next to its outputs
            shutil.copy2(src_path, raw_copy)
        _atomic_write_json(str(ckpt_path), {
            "content_hash": content_hash,
            "ir": doc.to_dict(),
            "raw_ocr": raw_ocr,
            "ocr_stage": {"model": _model_label("VLM_MODEL", "LLM_MODEL"),
                           "status": ocr_status, "at": now, "error": ocr_error},
            "visual_desc_stage": {"model": _model_label("VLM_MODEL", "LLM_MODEL"),
                                   "status": vdesc_status, "at": now,
                                   "error": vdesc_error},
        })
        return (doc.metadata or {}).get("vlm_health")
    except Exception as exc:  # noqa: BLE001
        parse.update(status="FAILED", error=str(exc), last_parsed=now)
    return None


def phase2_one(record: dict) -> None:
    """LLM-only half: finish a phase-1 checkpoint (deferred OCR checks + table
    descriptions), save the final IR, fill record["parse"], drop the
    checkpoint. A document without a checkpoint (phase 1 failed/skipped it)
    is left as phase 1 recorded it. Never raises."""
    doc_id = record["identity"]["doc_id"]
    content_hash = record["scan"]["content_hash"]
    parse = record["parse"]
    now = _now()

    ckpt_path = _stage1_path(record)
    if not ckpt_path.is_file():
        return

    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        doc = ParsedDocument.from_dict(migrate_dict(ckpt["ir"]))
        raw_ocr: dict[str, str] = ckpt.get("raw_ocr") or {}
        stages: dict = {"ocr": ckpt["ocr_stage"]}
        # Absent in a checkpoint written before the visual describe pass
        # existed -- an old checkpoint must still finish, not crash here.
        if ckpt.get("visual_desc_stage"):
            stages["visual_desc"] = ckpt["visual_desc_stage"]
        ocr_status = stages["ocr"]["status"]

        if ocr_status == "SUCCESS" and raw_ocr:
            try:
                apply_ocr_checks(doc, raw_ocr)
                check_status, check_error = "SUCCESS", None
            except Exception as exc:  # noqa: BLE001
                check_status, check_error = "FAILED", str(exc)
        else:
            # nothing transcribed (or ocr itself failed) -> nothing to check
            check_status = "SUCCESS" if ocr_status == "SUCCESS" else "SKIPPED"
            check_error = None
        stages["ocr_check"] = {"model": _model_label("LLM_MODEL"),
                                "status": check_status, "at": now, "error": check_error}

        if ocr_status == "SUCCESS" and doc.tables():
            try:
                describe_blocks(doc, stage="llm")
                table_status, table_error = "SUCCESS", None
            except Exception as exc:  # noqa: BLE001
                table_status, table_error = "FAILED", str(exc)
        else:
            table_status, table_error = "SKIPPED", None
        stages["table_desc"] = {"model": _model_label("LLM_MODEL"),
                                 "status": table_status, "at": now, "error": table_error}

        # Records which model(s)/context window actually produced this
        # output -- both inside the IR itself and, below, as a top-level
        # parse.models field so it's visible straight from document_nodes.json
        # without opening the IR JSON. Best-effort: never block the actual
        # save over a metadata snapshot.
        try:
            models = describe_configured_models()
            doc.metadata["models"] = models
        except Exception as exc:  # noqa: BLE001
            models = None
            progress.write(f"  !! could not record model metadata for {doc_id}, continuing: {exc!r}")

        group_dir = _group_dir(record)
        out_path = group_dir / f"{doc_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_path)
        try:  # Markdown is a rendering of the saved IR -- best-effort
            (group_dir / f"{doc_id}.md").write_text(to_markdown(doc),
                                                    encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            progress.write(f"  !! markdown render failed for {doc_id}: {exc}")

        if ocr_status == "FAILED":
            overall = "FAILED"
        elif (check_status == "FAILED" or table_status == "FAILED"
                or stages.get("visual_desc", {}).get("status") == "FAILED"):
            # non-critical stage failed but ocr succeeded -- see schema's
            # "overall parse.status derivation rule". A table_desc SKIPPED because
            # the doc simply has no tables is not a problem, so it stays SUCCESS.
            overall = "PARTIAL"
        else:
            overall = "SUCCESS"

        parse.update(parser=PARSER_LABEL, parser_version=PARSER_VERSION,
                     status=overall, parsed_from_hash=content_hash,
                     parsed_json_path=str(out_path), last_parsed=now,
                     error=None, stages=stages, models=models)
        ckpt_path.unlink(missing_ok=True)  # final IR saved -> checkpoint spent
    except Exception as exc:  # noqa: BLE001
        parse.update(status="FAILED", error=str(exc), last_parsed=now)


def _parse_argv(argv: list[str] | None = None) -> int | None:
    """`--limit N` (N >= 1) or None. See the module docstring for why this is
    the one flag."""
    import argparse

    def _positive(value: str) -> int:
        n = int(value)
        if n < 1:
            raise argparse.ArgumentTypeError("must be at least 1")
        return n

    ap = argparse.ArgumentParser(
        prog="python run_parse_pipeline.py",
        description="Parse the corpus listed in document_nodes.json "
                    "(configured via .env -- see .env.example).")
    ap.add_argument("--limit", type=_positive, metavar="N", default=None,
                    help="parse at most N pending documents (the first N in "
                         "registry order) and leave the rest PENDING -- for a "
                         "trial run before committing to the whole corpus")
    return ap.parse_args(argv).limit


def compute_pending(limit: int | None = None) -> tuple[list[dict], dict]:
    """Pure/read-only: reads `document_nodes.json` and returns the `todo` list
    that passes the `_needs_parse`/`DOC_TYPES` filters. Writes NOTHING to disk
    and makes no network calls (health probe/VLM/LLM) -- this is why N-03's
    dry-run reporting calls this function instead of `main()` triggering real
    work. `_bootstrap()` having been called is a PRECONDITION (it uses
    DOCUMENT_NODES_PATH/DOC_TYPES). The second value carries summary info the
    caller can report (`data`, `pending`, `total`)."""
    with open(DOCUMENT_NODES_PATH, encoding="utf-8") as f:
        data = json.load(f)

    todo = [r for r in data["documents"]
           if _needs_parse(r) and (not DOC_TYPES or r.get("doc_type") in DOC_TYPES)]
    pending = len(todo)
    if limit is not None and limit < pending:
        # Cut AFTER the _needs_parse/DOC_TYPES filter: "N of the documents that
        # actually need work", not "look at the first N records and maybe do
        # nothing". The rest keep their PENDING state and the next run takes
        # them -- nothing about the registry is truncated.
        todo = todo[:limit]
    return todo, {"data": data, "pending": pending, "total": len(data["documents"])}


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    limit = _parse_argv(argv)

    if DOC_TYPES:
        print(f"doc_type filter: {', '.join(sorted(DOC_TYPES))}")
    todo, info = compute_pending(limit)
    data = info["data"]
    if limit is not None and limit < info["pending"]:
        print(f"--limit {limit}: {info['pending']} pending documents, parsing the "
              f"first {limit}; the rest stay PENDING for a later run")
    print(f"{len(todo)} / {info['total']} documents will be parsed")

    # DOC_CONCURRENCY documents in flight at once. phase1_one()/phase2_one()
    # never raise (see their docstrings) and each call only mutates its own
    # record dict, so concurrent documents don't step on each other; the
    # write_lock just keeps two threads from calling _atomic_write_json at
    # the same instant.
    max_workers = max(1, _pipeline_config().run.doc_concurrency)
    write_lock = threading.Lock()

    def _run_phase(label: str, fn, after_doc=None) -> None:
        # `after_doc` (fed each document's fn() return value) may raise to
        # abort the whole run -- the health breaker uses this when the VLM
        # endpoint goes systemically bad mid-run. The phase-level bar counts
        # whole documents; the per-page/image/table bars inside come from the
        # parser's own stages (see parser/progress.py).
        pbar = progress.bar(len(todo), label, unit="doc")

        def _run(i: int, record: dict) -> None:
            progress.write(f"[{label} {i}/{len(todo)}] {record['identity']['file_name']}")
            result = fn(record)
            with write_lock:
                _atomic_write_json(DOCUMENT_NODES_PATH, data)  # save after every document, so a crash doesn't lose work
            pbar.update()
            if after_doc is not None:
                after_doc(result)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run, i, record) for i, record in enumerate(todo, 1)]
                try:
                    for f in futures:
                        f.result()
                except BaseException:
                    # Abort means abort: without cancel_futures the `with` exit's
                    # shutdown(wait=True) would still drain the ~whole queued
                    # corpus against the very endpoint the breaker just declared
                    # dead. In-flight documents finish; queued ones are dropped.
                    pool.shutdown(cancel_futures=True)
                    raise
        finally:
            pbar.close()

    # All VLM_CLASSIFY work first, then all VLM work, then all LLM work -- one
    # model swap per run instead of several per document (see module
    # docstring). Each phase starts with a known-answer canary against the
    # model it is about to lean on, and phase 1 keeps watching for systemic
    # VLM failure while it runs. Phase 0 is skipped with a note when
    # VLM_CLASSIFY isn't configured at all (get_vlm_client("classify") then
    # falls back to VLM itself, so there is nothing distinct to pre-warm --
    # phase 1 would just re-do the identical classify calls for free).
    if vlm_classify_configured():
        if _health_on():
            _probe_or_abort("VLM_CLASSIFY")
        _run_phase("phase 0/3 classify", phase0_classify_one)
    else:
        print("VLM_CLASSIFY not configured -- classify reuses VLM itself, "
              "nothing distinct to pre-warm; phase 0 skipped")
    vlm_isi_var = _vlm_work_configured()
    if not vlm_isi_var:
        print("VLM'e is dusmuyor (pdf.vlm / image_ocr.enabled / "
              "describe.enabled / visual.classify hepsi kapali) -- "
              "VLM canary'si ve saglik kesicisi atlandi")
    if vlm_isi_var and _health_on():
        _probe_or_abort("VLM")
    breaker = _VlmHealthBreaker() if (vlm_isi_var and _health_on()) else None
    _run_phase("phase 1/3 parse+ocr", phase1_one,
               after_doc=breaker.record if breaker else None)
    if _health_on():
        _probe_or_abort("LLM")
    _run_phase("phase 2/3 check+tables", phase2_one)

    counts: dict[str, int] = {}
    for r in data["documents"]:
        s = r["parse"]["status"]
        counts[s] = counts.get(s, 0) + 1
    print("summary:", counts)

    # N-21: `run_nightly.py::stage_parse` writes this into ParseSection.processed_count
    # -- before this the value was computed and DISCARDED (the function returned
    # `None`), so the report had to invent a made-up number for "documents
    # parsed"; it was exposed out so it wouldn't have to.
    return len(todo)


if __name__ == "__main__":
    main()  # return value (N-21: number of documents parsed) NOT the CLI exit code
