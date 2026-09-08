"""Chunk pipeline: parsed IR JSONs -> chunk JSON, via the `chunker` package.

The chunking counterpart of run_parse_pipeline.py. Where that script turns
corpus documents into IR JSON (PARSED_OUTPUT_DIR), this one turns those IR
JSONs into chunk sets under chunker's output home (`chunker/storage/` by
default): `all_chunks.json` (every document's chunk set, doc_id -> ChunkSet)
and `all_combined.json` (the same dict, wrapped) -- plus `viz/` (per-document
HTML, unaffected). Neither `parser` nor `chunker` imports the other; only
pipeline/ knows about both.

Deliberately dropped: the old per-document/per-scope file layout, and
with it the staleness gate -- every chunk run now reprocesses the whole
corpus from scratch, trading incremental cost for output simplicity.

RAPTOR (embedding-based summary tree / corpus-profile clustering) was
removed entirely -- `all_raptor.json` and `CHUNKER_RAPTOR` no longer exist.

chunker's interface is env variables ONLY (its decision, no CLI flags), so
this script is a thin adapter: it resolves paths from its own .env
(src/medrag/pipeline/cli/.env), sets CHUNKER_INPUT_DIR / CHUNKER_OUTPUT_DIR,
and runs `python -m medrag.pipeline.chunker` as a subprocess. Discovery,
per-document error handling and exit codes (0 = all ok, 1 = >=1 document
failed, 2 = config error) live in chunker's own `cli.py` -- not duplicated
here.

The `chunker` package is installed via `pip install -e .`, so it no longer
needs `CHUNKER_DIR` as a subprocess cwd/import-path trick --
`python -m medrag.pipeline.chunker` resolves the same everywhere.
`CHUNKER_DIR` is kept only as the DATA root (chunker's `storage/` output
home stayed at the old `chunker/` location, same "data doesn't move with
code" precedent as the vectorize move).

After the subprocess, this script writes each document's `chunk` block back
into document_nodes.json: status/chunked_at/chunked_from_hash/chunks_path/
chunker_version, read straight from the chunk file's own provenance rather
than re-derived here. Without this, "which part of the corpus is chunked?"
could only be answered by counting files in a folder. The registry is
optional: no DOCUMENT_NODES_PATH (or no file there) simply skips this step,
so chunking an ad-hoc IR folder still works.

Settings (this package's own .env; see .env.example):
    CHUNKER_DIR         root of chunker's DATA home (default ../../../../chunker,
                        i.e. repo-root chunker/, 4 levels up from cli/) --
                        NOT the code location any more; only used to default
                        CHUNKS_OUTPUT_DIR to CHUNKER_DIR/storage
    CHUNKER_INPUT_DIR   IR JSON root; default: PARSED_OUTPUT_DIR (the parse
                        pipeline's output, so the two scripts chain naturally)
    CHUNKS_OUTPUT_DIR   output ROOT; default CHUNKER_DIR/storage. chunker lays
                        out all_chunks.json + all_combined.json + viz/
                        underneath it -- the layout belongs to chunker
                        (its layout.py), this script only picks the root
    DOCUMENT_NODES_PATH the registry to write `chunk` blocks into (optional)
    CHUNKER_CONFIG, TOKENIZER   optional pass-through to chunker (if unset
                        here, chunker's own .env values apply)

Usage:
    python run_chunk_pipeline.py
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve(raw: str) -> Path:
    """Relative paths are relative to pipeline/ (same rule as the other
    pipeline scripts / .env.example); absolute paths are used as-is."""
    p = Path(raw)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def resolve_all_chunks_path() -> Path:
    """Path that this run ACTUALLY WRITES TO / READS FROM for
    `all_chunks.json` -- same logic as `main()` (CHUNKS_OUTPUT_DIR >
    CHUNKER_DIR/storage). `run_nightly.py::stage_chunk` reads this same file
    before and after to compute `produced_count`/`removed_count` -- kept
    here (it used to be a local in main()) so the path-resolution logic
    lives in one place."""
    chunker_dir = _resolve(os.environ.get("CHUNKER_DIR") or "../../../../chunker")
    out_root = _resolve(os.environ.get("CHUNKS_OUTPUT_DIR") or str(chunker_dir / "storage"))
    return out_root / "all_chunks.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write-then-replace: a crash mid-write never truncates the registry
    (same guard as run_parse_pipeline.py)."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_",
                                    suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _all_documents(all_chunks_path: Path) -> dict[str, dict]:
    """The `documents` dict of all_chunks.json (doc_id -> ChunkSet dict);
    empty if file is missing/unreadable -- caller treats this as "no
    document has been chunked" (single-file layout; no per-document files)."""
    if not all_chunks_path.is_file():
        return {}
    try:
        data = json.loads(all_chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("documents") or {}


def _chunk_block(doc_data: dict | None, all_chunks_path: Path) -> dict | None:
    """A document's registry `chunk` block, read off its entry inside
    all_chunks.json's own provenance. None = no entry for this doc_id.

    The values are copied, never re-derived: `chunked_from_hash` is the IR's
    `raw_sha256` as the chunker recorded it, so registry and chunk data
    can never disagree about which bytes a chunk set came from. `chunks_path`
    points at the ONE shared file -- every SUCCESS record in the registry
    shares the same path; the document's own doc_id is the key to find it
    inside `documents`.
    """
    if doc_data is None:
        return None
    prov = doc_data.get("provenance") or {}
    source = prov.get("source") or {}
    return {
        "status": "SUCCESS",
        "chunked_at": prov.get("generated_at"),
        "chunked_from_hash": source.get("raw_sha256"),
        "chunks_path": str(all_chunks_path),
        "chunker_version": prov.get("chunker_version"),
    }


def update_registry(registry_path: Path, all_chunks_path: Path) -> tuple[int, int]:
    """Write each record's `chunk` block from all_chunks.json. Returns
    (updated, missing) counts.

    A document with no entry is NOT silently left alone: its block is
    reset to PENDING, so a doc that used to be chunked and no longer is
    (source deleted, chunker failed) stops claiming SUCCESS forever.
    """
    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)

    belgeler = _all_documents(all_chunks_path)
    guncel = eksik = 0
    for record in data.get("documents", []):
        doc_id = record["identity"]["doc_id"]
        block = _chunk_block(belgeler.get(doc_id), all_chunks_path)
        if block is None:
            eksik += 1
            record["chunk"] = {"status": "PENDING", "chunked_at": None,
                               "chunked_from_hash": None, "chunks_path": None,
                               "chunker_version": None}
        else:
            guncel += 1
            record["chunk"] = block
    _atomic_write_json(registry_path, data)
    return guncel, eksik


def main() -> int:
    # chunker_dir is now DATA-only (storage/ etc.), not code -- the code
    # being importable as `medrag.pipeline.chunker` is verified separately via
    # importlib; directory existence alone is not sufficient.
    try:
        importlib.import_module("medrag.pipeline.chunker")
    except ImportError as exc:
        print(f"HATA: medrag.pipeline.chunker import edilemedi ({exc!r}) -- "
              "`pip install -e .` çalıştırıldı mı?", file=sys.stderr)
        return 2
    input_raw = (os.environ.get("CHUNKER_INPUT_DIR")
                 or os.environ.get("PARSED_OUTPUT_DIR"))
    if not input_raw:
        print("HATA: CHUNKER_INPUT_DIR de PARSED_OUTPUT_DIR de tanımsız "
              "(bkz. .env.example)", file=sys.stderr)
        return 2
    input_dir = _resolve(input_raw)
    if not input_dir.is_dir():
        print(f"HATA: girdi klasörü yok: {input_dir}", file=sys.stderr)
        return 2

    # Output ROOT (single-file layout): chunker writes two fixed files
    # (all_chunks.json, all_combined.json) + viz/. The layout owner is
    # chunker's own layout.py; here we only choose the root, file names
    # are a shared contract with chunker.
    all_chunks_file = resolve_all_chunks_path()
    out_root = all_chunks_file.parent
    out_root.mkdir(parents=True, exist_ok=True)

    # Paths must be absolute here: previously the subprocess cwd was
    # CHUNKER_DIR (relative paths resolved against it); now chunker is an
    # installed package so cwd doesn't matter -- but keeping them absolute
    # is harmless and safer.
    env = os.environ.copy()
    env["CHUNKER_INPUT_DIR"] = str(input_dir)
    env["CHUNKER_OUTPUT_DIR"] = str(out_root)
    # Bug fix during subprocess wiring verification: pipeline/.env's own
    # `PARSER_DIR=../parser` (meant for run_parse_pipeline.py, relative to
    # pipeline/'s own directory) leaked through `os.environ.copy()` into
    # this subprocess -- chunker's `_parser_path.py` also reads `PARSER_DIR`,
    # but resolves it relative to ITS OWN package directory (now 4 levels
    # deep under src/medrag/pipeline/chunker/). Before the move the two
    # components' directories happened to sit at the same depth from the
    # repo root, so the same relative value coincidentally pointed at the
    # right place; after the move it silently pointed at a nonexistent
    # directory and broke `import parsers.base` (ModuleNotFoundError).
    # chunker's own PARSER_DIR default is already correct without this
    # override, so it is simply not forwarded.
    env.pop("PARSER_DIR", None)

    print(f"chunker: {input_dir} -> {out_root}")
    proc = subprocess.run([sys.executable, "-m", "medrag.pipeline.chunker"],
                          env=env, check=False)  # exit code is reported below; the run is not aborted on non-zero

    n_belge = len(_all_documents(all_chunks_file))
    print(f"bitti: exit={proc.returncode}, {out_root} altında {n_belge} doküman")

    # registry's `chunk` block. Best-effort -- chunk output is already on
    # disk, the run must not abort if the registry update fails. No
    # staleness anymore -- every run rewrites all_chunks.json from scratch.
    registry_raw = os.environ.get("DOCUMENT_NODES_PATH")
    registry = _resolve(registry_raw) if registry_raw else None
    if registry is None or not registry.is_file():
        print("registry güncellenmedi (DOCUMENT_NODES_PATH tanımsız/dosya yok)"
              " — ad-hoc koşu")
    else:
        try:
            guncel, eksik = update_registry(registry, all_chunks_file)
            print(f"registry: {registry} — {guncel} kayıt chunk'lı, "
                  f"{eksik} kayıt PENDING")
        except Exception as exc:  # noqa: BLE001
            print(f"!! registry güncellenemedi ({registry}): {exc!r}",
                  file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())