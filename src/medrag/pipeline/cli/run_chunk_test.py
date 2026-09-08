"""Chunker integration smoke test: real parser IR -> chunker -> validated chunks.

The chunking counterpart of run_e2e_test.py, with one big difference: chunking
is offline (no LLM/VLM), so this runs anywhere in seconds -- no Ollama
machine, no model preflight. It does NOT duplicate chunker's own unit suite
(src/medrag/pipeline/chunker/tests/, offline, run with pytest); this checks
the INTEGRATION: real IR JSONs produced by the parser go through adapter ->
engine -> serialized chunk files, and the results hold the contract.

Input (first match wins):
    1. CHUNK_TEST_INPUT_DIR (pipeline/.env or real env) -- any folder that
       contains parser IR JSONs (scanned recursively, content-checked).
    2. The newest run under pipeline/e2e_test_output/ -- so the natural flow
       is `run_e2e_test.py` once on the GPU machine, then this anywhere.

What is forced, for determinism (same idea as run_e2e_test.py force-blanking
VLM2_*): CHUNKER_CONFIG="" always -- a stray value in chunker/.env can never
change what this test runs. Lookups below are by doc_id inside
all_chunks.json's own `documents` dict. TOKENIZER is taken from the env
(default cl100k_base; set TOKENIZER=fake if the tiktoken encoding cannot be
fetched/cached on this machine).

RAPTOR (embedding-based summary tree) was removed entirely; this test never
referenced it beyond forcing it off.

Checks per document (FAIL = any of these):
    - chunker produced an entry for this doc_id inside all_chunks.json's
      `documents` dict
    - the entry round-trips through ChunkSet.model_validate() -- i.e. pydantic
      re-runs the schema version gate and the leaf/summary field contract on
      what was actually written to disk
    - the entry carries provenance, and its `source.raw_sha256` matches the IR
      sitting next to it: this is what proves the chunks were built from THIS
      IR and not left over from an older one -- without it, block-id references
      (`source_block_ids`) point into a file that may no longer exist as it
      was
    - >= 1 leaf chunk, and no leaf has empty text
    - the prev/next leaf chain is a single unbroken walk over all leaves
      (reading order survived chunking)
Reported as info, not failure: token stats (min/median/max), leaves over
their own token_limit (a legitimately unsplittable unit may exceed it),
split/flex counts, images, unresolved cross-refs (expected: PDF carries no
anchor_id and numbered "Table N" references are deliberately not resolved).

Output (pipeline/chunk_test_output/<timestamp>/ -- this is the chunker output
ROOT for the test run, so its layout is the real one):
    all_chunks.json            <- doc_id -> ChunkSet (what this test checks)
    viz/                       <- empty here
    chunker_run.log            <- the subprocess' stdout+stderr
    summary.json / summary.md  <- one record per document + totals

Exit code: 0 = all documents PASS, 1 = >=1 FAIL, 2 = setup/config problem.

Usage:
    python run_chunk_test.py

Requires chunker's deps (pip install -e . at the repo root now also installs
`medrag.pipeline.chunker`) -- ChunkSet is imported straight from the installed
package for the round-trip validation (no more sys.path bootstrap).
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def _is_ir(path: Path) -> dict | None:
    """Same content check as chunker/cli.py's discovery (int ir_version +
    list blocks) -- duplicated on purpose: the test must know what chunker
    WILL pick up without importing its internals for it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if (isinstance(data, dict) and isinstance(data.get("ir_version"), int)
            and isinstance(data.get("blocks"), list)):
        return data
    return None


def find_input_dir() -> Path | None:
    raw = os.environ.get("CHUNK_TEST_INPUT_DIR")
    if raw:
        p = _resolve(raw)
        return p if p.is_dir() else None
    e2e_root = BASE_DIR / "e2e_test_output"
    if not e2e_root.is_dir():
        return None
    runs = sorted((d for d in e2e_root.iterdir() if d.is_dir()), reverse=True)
    return runs[0] if runs else None


def _ir_raw_sha256(ir_path: Path) -> str | None:
    """The IR's own `raw_sha256` (the source file's hash, `sha256:<hex>`).
    None = unreadable or absent -- then there is nothing to compare against
    and the provenance cross-check below simply doesn't fire."""
    try:
        data = json.loads(ir_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("raw_sha256")


def check_document(doc_id: str, ir_path: Path, documents: dict,
                   chunk_set_cls) -> dict:
    """One summary record per IR document; rec['status'] is PASS or FAIL."""
    rec: dict = {"doc_id": doc_id, "ir_file": str(ir_path),
                 "status": "FAIL", "errors": []}
    entry = documents.get(doc_id)
    if entry is None:
        rec["errors"].append("all_chunks.json'da bu doc_id yok")
        return rec

    try:
        cs = chunk_set_cls.model_validate(entry)  # pydantic: schema version gate + contract
    except Exception as exc:  # noqa: BLE001 -- validation error is written verbatim into rec["errors"] and the record is reported as FAIL
        rec["errors"].append(f"ChunkSet.model_validate: {exc}")
        return rec

    if cs.doc_id != doc_id:
        rec["errors"].append(f"doc_id uyuşmuyor: {cs.doc_id!r}")

    # Was the chunk set produced from the IR sitting next to it? If the hash
    # is missing or points at a different IR, the file is stale -- meaning an
    # old file is sitting around when it should have been produced by this run.
    if cs.provenance is None:
        rec["errors"].append("provenance yok (v4 öncesi/bayat dosya)")
    else:
        rec["chunker_version"] = cs.provenance.chunker_version
        ir_hash = _ir_raw_sha256(ir_path)
        chunk_hash = (cs.provenance.source.raw_sha256
                      if cs.provenance.source else None)
        if ir_hash and chunk_hash and ir_hash != chunk_hash:
            rec["errors"].append(
                f"provenance başka bir IR'ı işaret ediyor: chunk "
                f"{chunk_hash!r} != IR {ir_hash!r}")

    leaves = cs.leaves()
    if not leaves:
        rec["errors"].append("hiç leaf chunk yok")
        return rec
    empty = [n.node_id for n in leaves if not n.text.strip()]
    if empty:
        rec["errors"].append(f"boş metinli leaf: {empty[:3]}")

    # prev/next chain: a single walk from the start should cover every leaf
    # exactly once (did reading order survive chunking?).
    by_id = {n.node_id: n for n in leaves}
    starts = [n for n in leaves if n.prev_node_id is None]
    if len(starts) != 1:
        rec["errors"].append(f"{len(starts)} zincir başlangıcı (1 beklenir)")
    else:
        seen, cur = set(), starts[0]
        while cur is not None and cur.node_id not in seen:
            seen.add(cur.node_id)
            cur = by_id.get(cur.next_node_id) if cur.next_node_id else None
        if len(seen) != len(leaves):
            rec["errors"].append(
                f"zincir {len(seen)}/{len(leaves)} leaf kapsıyor")

    tokens = [n.token_count for n in leaves if n.token_count is not None]
    rec.update({
        "leaves": len(leaves),
        "tokens_min": min(tokens) if tokens else None,
        "tokens_median": int(statistics.median(tokens)) if tokens else None,
        "tokens_max": max(tokens) if tokens else None,
        # informational: an unsplittable unit may legitimately exceed the limit
        "over_limit": sum(1 for n in leaves
                          if n.token_count is not None
                          and n.token_limit is not None
                          and n.token_count > n.token_limit),
        "split_chunks": sum(1 for n in leaves if n.split_kind),
        "flex_chunks": sum(1 for n in leaves if n.flex_applied),
        "images": sum(len(n.images) for n in leaves),
        "cross_refs": sum(len(n.cross_refs) for n in leaves),
        "unresolved_refs": sum(1 for n in leaves for r in n.cross_refs
                               if r.target_chunk_id is None),
    })
    if not rec["errors"]:
        rec["status"] = "PASS"
    return rec


def write_summaries(out_dir: Path, records: list[dict],
                    totals: dict) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps({"totals": totals, "documents": records},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ("status", "leaves", "tokens_min", "tokens_median", "tokens_max",
            "over_limit", "split_chunks", "flex_chunks", "images",
            "cross_refs", "unresolved_refs")
    lines = ["# chunk test summary", "",
             "| doc_id | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for r in records:
        lines.append("| " + r["doc_id"] + " | "
                     + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", f"**Totals:** {totals}", ""]
    for r in records:
        for err in r["errors"]:
            lines.append(f"- FAIL `{r['doc_id']}`: {err}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


def main() -> int:
    # chunker runs from the installed package, not from a directory on disk;
    # importability is what is verified (same pattern as run_chunk_pipeline.py).
    try:
        importlib.import_module("medrag.pipeline.chunker")
    except ImportError as exc:
        print(f"HATA: medrag.pipeline.chunker import edilemedi ({exc!r}) -- "
              "`pip install -e .` çalıştırıldı mı?", file=sys.stderr)
        return 2

    input_dir = find_input_dir()
    if input_dir is None:
        print("HATA: girdi bulunamadı — CHUNK_TEST_INPUT_DIR tanımla veya "
              "önce run_e2e_test.py ile bir koşu üret.", file=sys.stderr)
        return 2

    ir_docs: dict[str, Path] = {}
    for path in sorted(input_dir.rglob("*.json")):
        data = _is_ir(path)
        if data is None:
            continue
        doc_id = data.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            print(f"HATA: IR dosyasında doc_id yok: {path}", file=sys.stderr)
            return 2
        ir_docs.setdefault(doc_id, path)
    if not ir_docs:
        print(f"HATA: {input_dir} altında hiç IR JSON yok.", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005 -- output directory name; local wall-clock is intentional so it matches a human reading the folder
    out_dir = BASE_DIR / "chunk_test_output" / stamp
    out_dir.mkdir(parents=True)
    # CHUNKER_OUTPUT_DIR is the output ROOT; chunker puts its two fixed
    # files + viz/ under it.
    all_chunks_path = out_dir / "all_chunks.json"

    env = os.environ.copy()
    env["CHUNKER_INPUT_DIR"] = str(input_dir)
    env["CHUNKER_OUTPUT_DIR"] = str(out_dir)
    # chunker's own .env moved to src/medrag/pipeline/chunker/.env (it shipped
    # with the package) -- but since "no override leaks in" is guaranteed
    # by the env var below, the name/meaning hasn't changed.
    env["CHUNKER_CONFIG"] = ""      # no override may leak in from chunker's own .env
    # Bug fix during subprocess wiring verification: pipeline/.env's own
    # PARSER_DIR (relative to pipeline/'s own directory, meant for
    # run_parse_pipeline.py) otherwise leaks in via os.environ.copy() and
    # gets re-resolved relative to chunker's package directory (now 4 levels
    # deep under src/medrag/pipeline/chunker/) -- broke `import parsers.base`
    # after the move (see run_chunk_pipeline.py's matching comment).
    env.pop("PARSER_DIR", None)
    # On Windows the child process logs in the local code page (cp1254) by
    # default; pinning both ends to UTF-8 lets the pipe be read cleanly.
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"chunker: {input_dir} ({len(ir_docs)} IR) -> {all_chunks_path}")
    proc = subprocess.run([sys.executable, "-m", "medrag.pipeline.chunker"],  # noqa: PLW1510 -- exit code is checked explicitly below via `proc.returncode == 2`
                          env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    (out_dir / "chunker_run.log").write_text(
        (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode == 2:
        print("HATA: chunker yapılandırma hatasıyla çıktı (exit 2) — "
              f"bkz. {out_dir / 'chunker_run.log'}", file=sys.stderr)
        return 2

    # ChunkSet is imported here, on purpose: reading back the subprocess'
    # output through the SAME pydantic contract is the test. chunker is now
    # an installed package -- no sys.path bootstrap needed.
    from medrag.pipeline.chunker.core.chunk import ChunkSet

    if all_chunks_path.is_file():
        documents = json.loads(all_chunks_path.read_text(encoding="utf-8")).get("documents", {})
    else:
        documents = {}
    records = [check_document(doc_id, ir_path, documents, ChunkSet)
               for doc_id, ir_path in sorted(ir_docs.items())]
    n_fail = sum(1 for r in records if r["status"] == "FAIL")
    totals = {
        "documents": len(records),
        "pass": len(records) - n_fail,
        "fail": n_fail,
        "chunker_exit": proc.returncode,
        "leaves": sum(r.get("leaves", 0) for r in records),
        "over_limit": sum(r.get("over_limit", 0) for r in records),
        "unresolved_refs": sum(r.get("unresolved_refs", 0) for r in records),
    }
    write_summaries(out_dir, records, totals)

    print(f"bitti: {totals['pass']}/{totals['documents']} PASS "
          f"(leaf={totals['leaves']}, over_limit={totals['over_limit']}) "
          f"-> {out_dir / 'summary.md'}")
    if n_fail:
        for r in records:
            for err in r["errors"]:
                print(f"  FAIL {r['doc_id']}: {err}", file=sys.stderr)
    return 1 if n_fail or proc.returncode else 0


if __name__ == "__main__":
    sys.exit(main())