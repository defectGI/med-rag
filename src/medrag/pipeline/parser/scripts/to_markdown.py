"""Parse a document to IR, enrich it (image OCR + table descriptions), and render
the result to Markdown -- the CLI face of render/markdown.py.

Usage:
    python scripts/to_markdown.py <path-to-document> [doc_id] [--stage STAGE]

Writes <output-dir>/<doc_id>.json (the IR) and <output-dir>/<doc_id>.md (the
rendered Markdown). Output dir defaults to storage/output (STORAGE_OUTPUT_DIR).

Enrichment (image OCR, table descriptions) uses the configured LLM/VLM (.env,
see llm/). Each enrichment stage is best-effort: if a model call fails the parse
+ render still complete, just without that enrichment -- so you always get a .md.

Stages (--stage, default "all") -- the model-affinity split. "all" runs
everything in one process (the original behavior). A phased pipeline (see
pipeline/run_e2e_test.py) instead runs "parse" for EVERY document first, then
"finish" for every document, so a single-GPU Ollama loads the VLM once and the
LLM once per run instead of swapping them in and out of VRAM per document (and,
worst of all, per image -- OCR transcription is VLM work but its meaningfulness
check is LLM work):

    parse   VLM-only work: parse the document + fetch/store/OCR its images
            (LLM meaningfulness check deferred). Writes <doc_id>.stage1.json --
            {"ir": <IR dict>, "raw_ocr": {"<imageN>": raw transcription}} --
            atomically (tmp + os.replace), so an existing stage1 file is always
            a COMPLETE checkpoint: a killed/crashed run never leaves a torn one
            behind, and a re-run can trust existence alone.
    finish  LLM-only work: load the checkpoint, apply the deferred OCR checks,
            describe tables, then write the final .json + .md. The stage1 file
            is kept (it makes "re-run finish only", e.g. after a prompt tweak,
            free -- no re-parse of minutes of VLM work).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from medrag.pipeline.parser.describe.core import describe_blocks
from medrag.pipeline.parser.images.image_handler import apply_ocr_checks, handle_images
from medrag.pipeline.parser.llm import describe_configured_models
from medrag.pipeline.parser.parsers.base import ParsedDocument, migrate_dict
from medrag.pipeline.parser.parsers.registry import parser_for
from medrag.pipeline.parser.render.markdown import to_markdown
from medrag.pipeline.parser.storage_paths import output_dir
from medrag.pipeline.parser.tables.header_infer import infer_headers


def _parse_document(raw_path: Path, doc_id: str) -> ParsedDocument:
    parser = parser_for(raw_path)
    print(f"parsing with {type(parser).__name__} ...", flush=True)
    t0 = time.time()
    doc = parser.parse(raw_path, doc_id)
    print(f"parsed in {time.time() - t0:.1f}s: fmt={doc.fmt} "
          f"blocks={len(doc.blocks)} tables={len(doc.tables())} "
          f"images={len(doc.images())}", flush=True)
    return doc


def _describe_visuals(doc: ParsedDocument) -> None:
    """VLM half of the describe pass: charts/diagrams/drawings read from their
    crops. Runs next to the image stage so the VLM stays loaded."""
    try:
        print("describing visuals (VLM) ...", flush=True)
        describe_blocks(doc, stage="vlm")
    except Exception as e:  # noqa: BLE001 -- best-effort: keep the parse + render; the error is printed to stdout as "!! visual describe error"
        print(f"!! visual describe error, continuing: {e!r}", flush=True)


def _describe_tables(doc: ParsedDocument) -> None:
    """LLM half of the describe pass: tables, written from their real cells."""
    if not doc.tables():
        return
    try:  # LLM fallback for tables whose source stated no header semantics
        infer_headers(doc)
    except Exception as e:  # noqa: BLE001 -- best-effort: header_rows just stays unknown; the error is printed as "!! header inference error"
        print(f"!! header inference error, continuing: {e!r}", flush=True)
    try:
        print("describing tables (LLM) ...", flush=True)
        describe_blocks(doc, stage="llm")
        print("tables described", flush=True)
    except Exception as e:  # noqa: BLE001 -- best-effort: keep the parse + render; the error is printed to stdout as "!! table stage error"
        print(f"!! table stage error, continuing: {e!r}", flush=True)


def _write_final(doc: ParsedDocument, doc_id: str) -> None:
    # Records which model(s)/context window actually produced this output --
    # otherwise that's only ever visible in stdout/env at run time, lost the
    # moment the process exits. Best-effort: never let a metadata snapshot
    # block getting the actual parse result saved.
    try:
        doc.metadata["models"] = describe_configured_models()
    except Exception as e:  # noqa: BLE001
        print(f"!! could not record model metadata, continuing: {e!r}", flush=True)

    out_json = output_dir() / f"{doc_id}.json"
    doc.save(out_json)
    md = to_markdown(doc)
    out_md = output_dir() / f"{doc_id}.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"saved IR -> {out_json}", flush=True)
    print(f"saved MD -> {out_md}  ({len(md)} chars, {md.count(chr(10)) + 1} lines)",
          flush=True)


def stage1_path(doc_id: str) -> Path:
    return output_dir() / f"{doc_id}.stage1.json"


def stage_parse(raw_path: Path, doc_id: str) -> None:
    """VLM-only half: parse + image fetch/OCR, checkpoint to stage1 json."""
    doc = _parse_document(raw_path, doc_id)

    raw_ocr: dict[str, str] = {}
    try:
        print("resolving images (classify + VLM OCR, check deferred) ...", flush=True)
        handle_images(doc, raw_sink=raw_ocr)
        print(f"images resolved ({len(raw_ocr)} transcription(s) pending check)",
              flush=True)
    except Exception as e:  # noqa: BLE001 -- best-effort: checkpoint what we have; the error is printed to stdout as "!! image stage error"
        print(f"!! image stage error, continuing: {e!r}", flush=True)

    _describe_visuals(doc)  # VLM work belongs in the VLM half

    # Atomic checkpoint: whatever reads this file later (finish stage, resume
    # scan) must never see a half-written one, so write-then-replace.
    out = stage1_path(doc_id)
    tmp = out.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps({"ir": doc.to_dict(), "raw_ocr": raw_ocr},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"saved stage1 checkpoint -> {out}", flush=True)


def stage_finish(doc_id: str) -> None:
    """LLM-only half: load the checkpoint, run deferred OCR checks + table
    descriptions, write the final .json + .md."""
    ckpt_file = stage1_path(doc_id)
    ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
    doc = ParsedDocument.from_dict(migrate_dict(ckpt["ir"]))
    raw_ocr: dict[str, str] = ckpt.get("raw_ocr") or {}
    print(f"loaded stage1 checkpoint: blocks={len(doc.blocks)} "
          f"tables={len(doc.tables())} pending_ocr={len(raw_ocr)}", flush=True)

    if raw_ocr:
        try:
            print("checking OCR transcriptions (LLM) ...", flush=True)
            apply_ocr_checks(doc, raw_ocr)
            ocr = sum(1 for im in doc.images() if im.ocr_meaningful)
            print(f"OCR checks done ({ocr} meaningful)", flush=True)
        except Exception as e:  # noqa: BLE001 -- best-effort: keep the parse + render; the error is printed to stdout as "!! ocr check error"
            print(f"!! ocr check error, continuing: {e!r}", flush=True)

    _describe_tables(doc)
    _write_final(doc, doc_id)


def stage_all(raw_path: Path, doc_id: str) -> None:
    """Original single-process flow: everything, models interleaved."""
    doc = _parse_document(raw_path, doc_id)

    try:
        print("resolving images (classify + OCR) ...", flush=True)
        handle_images(doc)
        ocr = sum(1 for im in doc.images() if im.ocr_meaningful)
        print(f"images resolved ({ocr} with meaningful OCR)", flush=True)
    except Exception as e:  # noqa: BLE001 -- best-effort: keep the parse + render; the error is printed to stdout as "!! image stage error"
        print(f"!! image stage error, continuing: {e!r}", flush=True)

    _describe_visuals(doc)
    _describe_tables(doc)
    _write_final(doc, doc_id)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="document to parse")
    ap.add_argument("doc_id", nargs="?", default=None)
    ap.add_argument("--stage", choices=("parse", "finish", "all"), default="all",
                    help="'parse' = VLM work only, checkpoint to "
                         "<doc_id>.stage1.json; 'finish' = LLM work from that "
                         "checkpoint to final .json/.md; 'all' (default) = both "
                         "in one go")
    args = ap.parse_args()

    raw_path = Path(args.path)
    doc_id = args.doc_id or raw_path.stem

    t0 = time.time()
    if args.stage == "parse":
        stage_parse(raw_path, doc_id)
    elif args.stage == "finish":
        stage_finish(doc_id)
    else:
        stage_all(raw_path, doc_id)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
