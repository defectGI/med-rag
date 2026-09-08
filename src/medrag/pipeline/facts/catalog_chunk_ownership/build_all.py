"""Builds the chunk->product OWNERSHIP table for the WHOLE corpus (NOT a
fact -- the generalisation of the PN1253 single-document trial (see
run_test.py + git history, 2026-08-04) to the full corpus, user decision).

Two paths, by the document's `document.product_codes` length:
  - SINGLE-owner document (724/772): the LLM is NOT NEEDED AT ALL --
    doc_id -> single product is already deterministic (the
    `product_codes` field of KARAR-055). Each chunk gets confidence=1.0,
    source="deterministic" with that single code.
  - MULTI-owner document (27/772, after `exclude_doc_types`): a SMALL
    LLM request per chunk (no dictionary, just the candidate list +
    chunk text -- see prompts/chunk_owner_system.md), sent IN PARALLEL
    (RTX 5090, user decision -- the earlier 262k ctx run on the same
    GPU LOCKED the GPU; here the prompt is small, num_ctx stays small,
    throughput comes from PARALLEL requests, not context size).

Confidence threshold 0.8 (user decision, validated on the PN1253 pilot:
sub-threshold assignments were either genuine family-wide edge cases or
"orphan" chunks cut at page boundaries; in both cases the model did NOT
assign HIGH confidence to a WRONG code). Sub-threshold assignments are
NOT DELETED -- they remain visible in the table with raw confidence +
`accepted=false` (same principle as discover.py's
`docs_excluded_fanout` / `docs_missing_text` "no silent skipping").

This script does NOT WRITE to specs.db (read-only); it only writes to
results/chunk_ownership_all.json. Hooking it into downstream (fan-out
narrowing) is a separate step.

PAUSABLE / RESUMABLE (2026-08-06): every chunk's LLM response is
appended to `results/chunk_owner_progress.jsonl` immediately. Ctrl+C
stops cleanly and does NOT OVERWRITE `chunk_ownership_all.json` (the
old full table is preserved); re-running the same command resumes
from where it stopped. This is for the use case "start in the morning,
stop when the GPU is needed for something else, resume later" (user
request).

Usage:
  cd facts/experiments/catalog_chunk_ownership
  python build_all.py              # from scratch OR resume from checkpoint
  python build_all.py --restart    # delete the checkpoint and start over
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from medrag.core.paths import resolve_specs_db_path

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/catalog_chunk_ownership/ (code, O-03)
REPO_ROOT = HERE.parents[4]  # medrag/ (catalog_chunk_ownership/facts/pipeline/urun/src/<repo>)
# DATA (db/, snapshots/) was NOT moved with the code -- O-03 scope was
# code only; same principle as `discover.py::_FACTS_DATA_ROOT`: lives in
# `facts/` at the repo root.
_FACTS_DATA_ROOT = REPO_ROOT / "facts"
DB_PATH = resolve_specs_db_path()
SNAPSHOTS_DIR = _FACTS_DATA_ROOT / "snapshots"
ATOMS_BRIDGE = Path(r"C:\\path\\to\\data\\atoms.ndjson")
# Same priority order as discover.py (set on this machine 2026-08-04 --
# 198/198 in-scope documents FULLY covered, FAR better than the
# ATOMS_BRIDGE's 123/772 partial coverage). Can be overridden via
# CHUNKER_OUTPUT_DIR; same env var as discover.py::CHUNKER_OUTPUT_ROOT
# (KARAR-012).
CHUNKER_OUTPUT_ROOT = Path(os.environ.get("CHUNKER_OUTPUT_DIR", REPO_ROOT / "chunker" / "storage"))
ALL_CHUNKS_PATH = CHUNKER_OUTPUT_ROOT / "all_chunks.json"

from medrag.core.db.issues import connect_issues
from medrag.core.paths import resolve_chunk_ownership_dir, resolve_issues_db_path
from medrag.pipeline.facts.facts_config import load_config
from medrag.pipeline.facts.issues_bridge import record_unresolved_docs


# N-22: RESULTS_DIR now uses the SAME `resolve_chunk_ownership_dir()`
# resolver as `discover.py::DEFAULT_CHUNK_OWNERSHIP_PATH` (the reader);
# it is resolved at CALL TIME (not frozen at import) so it is reread
# on every `main()` call (required for tests' `monkeypatch.setenv(
# "CHUNK_OWNERSHIP_DIR", ...)`). Default is UNCHANGED: env undefined ->
# STILL `HERE / "results"` (this file's own directory).
def _results_dir() -> Path:
    return resolve_chunk_ownership_dir()


# N-22: this script actually CALLS the LLM via `_chat_ollama` (for
# multi-owner documents). Model/URL used to be HARD-CODED (NOT read
# from env) -- the container's `http://localhost:11434` cannot reach
# the GPU machine, so `OWNERSHIP_LLM_MODEL`/`OWNERSHIP_LLM_BASE_URL`
# env vars now let it be overridden. Defaults are UNCHANGED (today's
# behaviour on this machine when env is unset).
MODEL = os.environ.get("OWNERSHIP_LLM_MODEL", "gemma4:31b")
NUM_CTX = 16384  # 2026-08-06: 8192 -> 16384. Measured: on a 45-candidate
# catalogue chunk the prompt alone is ~4967 tokens (system 1402 +
# candidate list 3122 + chunk 443); the span rule requests 45
# assignments so total exceeds 8192 and Ollama CUTS the JSON in HALF.
# Model was producing the RIGHT answer ("evidence": "3. ENVIRONMENTAL
# SP...") -- it was a window error, not a schema error. Just raising
# `num_predict` is NOT enough: the bottleneck is the total window.
# 16384 is a deliberately measured increase (262144 locked the GPU on
# 2026-08-04 -- target real demand, not model ceiling).
NUM_PREDICT = 8192  # 2026-08-06: 4096 -> 8192. The span rule (KARAR-064)
# roughly doubled the per-chunk output size; on 45+ candidate chunks
# the model hit 4096 and CUT JSON in half -- 2 chunks (`PN5085_CATALOGUE::c42`,
# `PN5078_Catalog::c0`) started with the CORRECT schema but dropped
# with "cannot parse". Schema error, NO -- ceiling error.
OLLAMA_URL = os.environ.get("OWNERSHIP_LLM_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
TIMEOUT_S = 400.0  # 2026-08-06: 180 -> 400. The span rule (KARAR-064)
# roughly doubled per-chunk assignment count (median 5 -> 10, up to 47 on
# 45+ candidate catalogue chunks); 180s timed out on these big outputs
# -- error rate rose from 1.1% to 9.2% and timeouts clustered exactly on
# the highest-owner documents (PN1253 catalogue). Model/GPU fault, NO
# -- generated token count grew.
MAX_WORKERS = 8  # RTX 5090, user decision -- small prompt / small num_ctx,
# throughput comes from parallelism
CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (HERE / "prompts" / "chunk_owner_system.md").read_text(encoding="utf-8")


class ProviderError(RuntimeError):
    pass


def _connect_ro() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _load_documents(con: sqlite3.Connection, exclude_doc_types: frozenset[str]) -> list[dict]:
    docs = []
    for row in con.execute("SELECT doc_id, file_name, doc_type, product_codes FROM document"):
        if row["doc_type"] in exclude_doc_types:
            continue
        codes = json.loads(row["product_codes"] or "[]")
        if not codes:
            continue
        docs.append({"doc_id": row["doc_id"], "file_name": row["file_name"],
                     "doc_type": row["doc_type"], "product_codes": codes})
    return docs


def _leaf_ordinal(node_id: str) -> int:
    # node_id = f"{doc_id}::c{n}" (chunker/chunker/core/chunk.py::leaf_node_id)
    # 'n' is the page/read order; numeric ordering is REQUIRED (string
    # ordering would put c9 before c10 -- wrong).
    try:
        return int(node_id.rsplit("::c", 1)[1])
    except (IndexError, ValueError):
        return 0


HEADING_PATH_BY_CHUNK: dict[str, list[str]] = {}
"""`{chunk_id: heading_path}` -- populated by
`_load_chunks_from_all_chunks_json`. The side index is deliberate: the
chunk tuple's shape (page_start, page_end, id, text) does NOT CHANGE,
so the bridge path and all call sites stay untouched."""


def _load_chunks_from_all_chunks_json() -> dict[str, list[tuple]]:
    """`{doc_id: [(page_start, page_end, node_id, text), ...]}` -- chunker's
    REAL output, with text EMBEDDED (no separate snapshot read needed).
    discover.py's PRIMARY source (ALL_CHUNKS_PATH); now available on this
    machine and covers 198/198 in-scope documents (2026-08-04 measurement)."""
    if not ALL_CHUNKS_PATH.is_file():
        return {}
    data = json.loads(ALL_CHUNKS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[tuple]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        leaves = [n for n in doc.get("nodes", []) if n.get("tree_level") == 0]
        leaves.sort(key=lambda n: _leaf_ordinal(n["node_id"]))
        out[doc_id] = [(n.get("page_start"), n.get("page_end"), n["node_id"], n.get("text") or "") for n in leaves]
        for n in leaves:
            # KARAR-064 clause 3: heading_path rides in the side index
            # so the chunk tuple's SHAPE does not change -- bridge path
            # has none -> empty.
            HEADING_PATH_BY_CHUNK[n["node_id"]] = list(n.get("heading_path") or [])
    return out


def _load_chunk_bridge() -> dict[str, list[tuple]]:
    """FALLBACK source -- for `doc_id`s that
    `_load_chunks_from_all_chunks_json` does NOT cover (discover.py's
    SAME priority: all_chunks.json missing or not covering -> falls
    back to atoms.ndjson+snapshots). `{doc_id: [(page_start, page_end,
    chunk_sha256, text), ...]}`."""
    if not ATOMS_BRIDGE.is_file():
        return {}
    per_doc: dict[str, dict[str, tuple]] = {}
    with ATOMS_BRIDGE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = obj.get("doc_id")
            sha = obj.get("chunk_sha256")
            if not did or not sha:
                continue
            per_doc.setdefault(did, {})[sha] = (obj.get("page_start"), obj.get("page_end"), sha)
    out: dict[str, list[tuple]] = {}
    for did, shamap in per_doc.items():
        rows = []
        for ps, pe, sha in sorted(shamap.values()):
            text = _snapshot_text(sha)
            if text is not None:
                rows.append((ps, pe, sha, text))
        if rows:
            out[did] = rows
    return out


def _snapshot_text(chunk_sha256: str) -> str | None:
    fname = chunk_sha256.split(":")[1] + ".txt"
    path = SNAPSHOTS_DIR / fname
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _load_product_info(con: sqlite3.Connection, codes: set[str]) -> dict[str, dict]:
    if not codes:
        return {}
    codes = sorted(codes)
    q = f"SELECT product_code, display_name, family, subfamily, subfamily_2, acme_code FROM product WHERE product_code IN ({','.join('?' * len(codes))})"
    out = {}
    for row in con.execute(q, codes):
        out[row["product_code"]] = {
            "product_code": row["product_code"], "display_name": row["display_name"],
            "family": row["family"], "subfamily": row["subfamily"],
            "subfamily_2": row["subfamily_2"], "acme_code": row["acme_code"],
        }
    return out


def _chat_ollama(system: str, user: str) -> tuple[str, dict]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "format": "json", "think": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            yanit = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"cannot reach {OLLAMA_URL}: {exc}") from exc
    try:
        content = yanit["message"]["content"] or ""
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"unexpected chat response shape: {str(yanit)[:500]}") from exc
    return content, {}


def _parse_assignments(content: str) -> list[dict]:
    for candidate in (content, content[content.find("{"): content.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("assignments"), list):
            return data["assignments"]
    raise ProviderError(f"cannot parse assignments JSON: {content[:500]}")


_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def headings_in(text: str) -> list[str]:
    """Markdown heading LINES in the chunk body (verbatim, `##` included).

    PROTOCOL KARAR-064 clause 3: the closed set the model is allowed to
    COPY as `heading_anchor` -- code validates the anchor against this
    list, so the model cannot invent a heading the slice cannot extract."""
    return [line.strip() for line in _HEADING_RE.findall(text or "")]


def normalize_anchor(assignment: dict, headings: list[str]) -> str | None:
    """Validates the model's `heading_anchor` against the closed set.

    No exact match -> `None` -- the assignment covers the WHOLE chunk
    (today's behaviour). No silent acceptance: caller counts and reports."""
    anchor = assignment.get("heading_anchor")
    if not isinstance(anchor, str):
        return None
    anchor = anchor.strip()
    return anchor if anchor in headings else None


def _tag_chunk_llm(
    candidates: list[dict], chunk_text: str, page_str: str, heading_path: list[str] | None = None,
) -> tuple[list[dict], str | None]:
    """`heading_path` + body headings also enter the prompt (KARAR-064 clause 3).

    An EMPTY `heading_path` is NOT HIDDEN, its meaning is WRITTEN -- in
    the corpus 22.1% of leaf chunks have empty heading_path and that
    means "sections with no common ancestor were merged"; it is the
    strongest deterministic signal that the chunk is likely MIXED (2026-
    08-06 measurement; PN1253 catalogue's `c16` was missed exactly this
    way)."""
    cand_json = json.dumps(candidates, ensure_ascii=False)
    path = heading_path or []
    path_line = (" > ".join(path) if path
                 else "(EMPTY — this chunk merged sections with no common ancestor; "
                       "treat it as likely MIXED and attribute section by section)")
    headings = headings_in(chunk_text)
    headings_line = json.dumps(headings, ensure_ascii=False) if headings else "(none)"
    user = (
        f"candidates:\n{cand_json}\n\n"
        f"heading_path: {path_line}\n"
        f"headings_in_chunk: {headings_line}\n\n"
        f"Chunk text (page {page_str}):\n{chunk_text}"
    )
    try:
        content, _usage = _chat_ollama(SYSTEM_PROMPT, user)
        return _parse_assignments(content), None
    except ProviderError as exc:
        return [], str(exc)


def split_ownership_work(
    docs: list[dict],
    chunks_for,
    product_info: dict[str, dict],
    *,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], list[tuple], list[dict], list[dict]]:
    """Splits documents into single-owner / multi-owner; for single-owner
    documents immediately emits deterministic rows; for multi-owner
    documents BUILDS the LLM job list (`llm_jobs`) but does NOT make
    ANY CALL.

    This function is the foundation of O-06 (I-13) acceptance: it does
    NOT import or call `_chat_ollama`/`_tag_chunk_llm` -- for a single-
    owner corpus `llm_jobs` is GUARANTEED to be empty (verified by a
    counter test, see `facts/tests/test_chunk_full_run_load.py`).

    Returns: `(rows, docs_without_chunk_text, llm_jobs, single_docs,
    multi_docs)` -- `llm_jobs` are `(doc, (page_start, page_end,
    chunk_id, text), candidates)` triples, UNPROCESSED (the caller --
    `main()` or a test -- decides when/how to process them); `single_docs`/
    `multi_docs` are returned for the caller to build report/meta."""
    single_docs = [d for d in docs if len(d["product_codes"]) == 1]
    multi_docs = [d for d in docs if len(d["product_codes"]) >= 2]
    if verbose:
        print(f"docs in scope: {len(docs)} (single-owner {len(single_docs)}, multi-owner {len(multi_docs)})")

    rows: list[dict] = []
    docs_without_chunk_text: list[dict] = []

    # --- Single-owner: NO LLM, deterministic ---
    for d in single_docs:
        chunks = chunks_for(d["doc_id"])
        if not chunks:
            docs_without_chunk_text.append({"doc_id": d["doc_id"], "file_name": d["file_name"], "reason": "no chunk source coverage"})
            continue
        code = d["product_codes"][0]
        for ps, pe, cid, _text in chunks:
            rows.append({
                "doc_id": d["doc_id"], "file_name": d["file_name"], "chunk_id": cid,
                "page_start": ps, "page_end": pe, "product_code": code,
                "confidence": 1.0, "accepted": True, "evidence": None, "source": "deterministic",
            })

    # --- Multi-owner: per-chunk LLM labelling (job list -- NO CALL) ---
    llm_jobs: list[tuple] = []  # (doc, chunk_tuple, candidates)
    for d in multi_docs:
        chunks = chunks_for(d["doc_id"])
        if not chunks:
            docs_without_chunk_text.append({"doc_id": d["doc_id"], "file_name": d["file_name"], "reason": "no chunk source coverage"})
            continue
        candidates = [product_info[c] for c in d["product_codes"] if c in product_info]
        missing_codes = [c for c in d["product_codes"] if c not in product_info]
        if missing_codes and verbose:
            print(f"WARNING: product(s) not found in product table for {d['file_name']}: {missing_codes}")
        for ps, pe, cid, text in chunks:
            llm_jobs.append((d, (ps, pe, cid, text), candidates))

    return rows, docs_without_chunk_text, llm_jobs, single_docs, multi_docs


def run_job(job: tuple) -> tuple:
    """Actually PROCESSES one `llm_jobs` entry -- the SINGLE place that
    (indirectly via `_tag_chunk_llm`) CALLS `_chat_ollama`. Moved to
    module level (was inside `main()`) so tests can monkeypatch
    `_chat_ollama` and directly measure call count/behaviour (O-06)."""
    d, (ps, pe, cid, text), candidates = job
    if not text:
        return d, (ps, pe, cid, text), [], "chunk text empty"
    page_str = str(ps) if ps == pe else f"{ps}-{pe}"
    assignments, error = _tag_chunk_llm(
        candidates, text, page_str, HEADING_PATH_BY_CHUNK.get(cid))
    return d, (ps, pe, cid, text), assignments, error


def absorb_chunk_result(
    d: dict, ps, pe, cid: str, text: str | None, assignments: list, error: str | None,
) -> tuple[list[dict], dict[str, int]]:
    """Converts a `run_job` result into rows -- applies the
    confidence>=0.8 gate (`CONFIDENCE_THRESHOLD`) and KARAR-064 anchor
    validation HERE.

    Moved to module level (O-06) so the confidence gate can be tested
    directly with a page/value set, without running `main()`. Returns:
    `(new_rows, {"error": 0|1, "anchor": n, "bad_anchor": n})`."""
    new_rows: list[dict] = []
    stats = {"error": 0, "anchor": 0, "bad_anchor": 0}
    if error:
        stats["error"] = 1
    valid_codes = set(d["product_codes"])
    headings = headings_in(text or "")
    for a in assignments:
        if not isinstance(a, dict):
            continue
        code = a.get("product_code")
        if code not in valid_codes:
            continue
        conf = a.get("confidence")
        accepted = isinstance(conf, (int, float)) and conf >= CONFIDENCE_THRESHOLD
        anchor = normalize_anchor(a, headings)
        if a.get("heading_anchor") and anchor is None:
            # Model gave an anchor that is NOT in the closed set --
            # assignment covers the whole chunk. No silent drop: it is
            # counted and reported.
            stats["bad_anchor"] += 1
        new_rows.append({
            "doc_id": d["doc_id"], "file_name": d["file_name"], "chunk_id": cid,
            "page_start": ps, "page_end": pe, "product_code": code,
            "confidence": conf, "accepted": accepted, "evidence": a.get("evidence"),
            "heading_anchor": anchor, "source": "llm",
        })
        if anchor:
            stats["anchor"] += 1
    if error:
        new_rows.append({
            "doc_id": d["doc_id"], "file_name": d["file_name"], "chunk_id": cid,
            "page_start": ps, "page_end": pe, "product_code": None,
            "confidence": None, "accepted": False, "evidence": None,
            "heading_anchor": None, "source": "llm", "error": error,
        })
    return new_rows, stats


def main(argv: list[str] | None = None) -> dict | None:
    """N-22: `argv=None` (default) -- SAME pattern as
    `run_parse_pipeline.main(argv=[])` (see `run_nightly.py` module
    docstring): `run_nightly.stage_ownership` calls this function with
    `argv=[]` PROGRAMMATICALLY; it does NOT open a subprocess. On
    success it RETURNS the `out` dict (meta + rows) -- the single real
    source of `run_nightly.py`'s nightly report's deterministic/LLM-led/
    unresolved chunk counts. Killed by Ctrl+C (only seen on manual runs,
    not expected in nightly runs) it returns `{"interrupted": True,
    ...}` -- `chunk_ownership_all.json` is DELIBERATELY NOT OVERWRITTEN
    (the old full table is preserved)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart", action="store_true",
                    help="DELETE the checkpoint and start over (default: resume from where it stopped).")
    ap.add_argument("--issues-db", default=str(resolve_issues_db_path()),
                    help="Path to issues.db where documents-without-chunk-text are written (O-07, "
                         "default: ISSUES_DB_PATH env var, 2026-08-26 fix).")
    args = ap.parse_args(argv)
    results_dir = _results_dir()

    con = _connect_ro()
    config = load_config()
    exclude_doc_types = frozenset(config.scope.exclude_doc_types)
    docs = _load_documents(con, exclude_doc_types)
    primary = _load_chunks_from_all_chunks_json()
    fallback = _load_chunk_bridge()
    print(f"chunk source: all_chunks.json {len(primary)} documents, atoms.ndjson fallback {len(fallback)} documents")

    def _chunks_for(doc_id: str) -> list[tuple] | None:
        return primary.get(doc_id) or fallback.get(doc_id)

    all_codes: set[str] = set()
    for d in docs:
        all_codes.update(d["product_codes"])
    product_info = _load_product_info(con, all_codes)

    rows, docs_without_chunk_text, llm_jobs, single_docs, multi_docs = split_ownership_work(
        docs, _chunks_for, product_info)

    if docs_without_chunk_text:
        # O-07: documents with no chunk text in any source do not drop
        # silently -- they are written to issues.db as UNRESOLVED_DOC (I-14).
        issues_con = connect_issues(args.issues_db)
        try:
            n_issues = record_unresolved_docs(issues_con, docs_without_chunk_text)
        finally:
            issues_con.close()
        print(f"  {n_issues} UNRESOLVED_DOC issues written -> {args.issues_db}")

    print(f"LLM labelling jobs: {len(llm_jobs)} chunks, {MAX_WORKERS} parallel requests, model={MODEL}")

    # --- Checkpoint + resume (2026-08-06, user request: "start during
    # the day and pause when the GPU is needed for something else").
    # EVERY chunk's LLM response is appended (append-only) to
    # `chunk_owner_progress.jsonl` immediately. On restart the
    # (doc_id, chunk_id) pairs in that file are NOT asked again.
    # Deterministic rows are NOT checkpointed (they were produced
    # without an LLM).
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = results_dir / "chunk_owner_progress.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if not args.restart and progress_path.is_file():
        with progress_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[(rec["doc_id"], rec["chunk_id"])] = rec  # last record wins
    elif args.restart and progress_path.is_file():
        progress_path.unlink()

    pending = [j for j in llm_jobs if (j[0]["doc_id"], j[1][2]) not in done]
    if done:
        print(f"  resuming: {len(done)} chunks came from checkpoint, {len(pending)} chunks remaining")

    t0 = time.monotonic()
    n_done = 0
    n_err = 0
    n_anchor = 0
    n_bad_anchor = 0
    interrupted = False

    def _absorb(d, ps, pe, cid, text, assignments, error):
        """Calls `absorb_chunk_result` and folds the rows/counters into
        `rows` and the nonlocal counters of the outer scope (so that
        checkpoint-loaded and freshly-fetched results go through the
        SAME path; a thin wrapper stays inside `main()` so the actual
        logic can be tested at module level)."""
        nonlocal n_err, n_anchor, n_bad_anchor
        new_rows, stats = absorb_chunk_result(d, ps, pe, cid, text, assignments, error)
        rows.extend(new_rows)
        n_err += stats["error"]
        n_anchor += stats["anchor"]
        n_bad_anchor += stats["bad_anchor"]

    # Process checkpoint-loaded ones first (text re-read fresh from
    # all_chunks because anchor validation needs the body headings).
    text_by_chunk = {j[1][2]: j[1][3] for j in llm_jobs}
    doc_by_id = {d["doc_id"]: d for d in multi_docs}
    for (doc_id, cid), rec in done.items():
        d = doc_by_id.get(doc_id)
        if d is None:
            continue
        _absorb(d, rec.get("page_start"), rec.get("page_end"), cid,
                text_by_chunk.get(cid, ""), rec.get("assignments") or [], rec.get("error"))

    with progress_path.open("a", encoding="utf-8") as progress_fh, \
            ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run_job, job) for job in pending]
        try:
            for fut in as_completed(futures):
                d, (ps, pe, cid, text), assignments, error = fut.result()
                n_done += 1
                progress_fh.write(json.dumps({
                    "doc_id": d["doc_id"], "chunk_id": cid, "page_start": ps, "page_end": pe,
                    "assignments": assignments, "error": error,
                }, ensure_ascii=False) + "\n")
                progress_fh.flush()
                _absorb(d, ps, pe, cid, text, assignments, error)
                if n_done % 20 == 0 or n_done == len(pending):
                    print(f"  [{n_done}/{len(pending)}] processed ({n_err} errors), "
                          f"{round(time.monotonic() - t0, 1)}s")
        except KeyboardInterrupt:
            # Clean stop: every chunk that finished is ALREADY on disk.
            # We do NOT overwrite `chunk_ownership_all.json` with the
            # half-done table -- the old (full) table stays intact and the
            # resume will write the new one.
            interrupted = True
            for f in futures:
                f.cancel()
            print(f"\n  STOPPED -- {n_done}/{len(pending)} chunks finished in this session, "
                  f"total {len(done) + n_done}/{len(llm_jobs)} in the checkpoint.")
            print("  To resume, run the SAME command again (use --restart to start over).")

    if interrupted:
        return {"interrupted": True, "n_llm_jobs": len(llm_jobs),
                "n_done_this_session": n_done, "n_checkpointed": len(done) + n_done}

    # N-22: "missing env must not silently produce an empty index, fail
    # loudly" -- when ALL chunks returned errors (e.g. OWNERSHIP_LLM_BASE_URL
    # points to an address unreachable from the container), the result
    # `rows` would be FILLED with error/bad-assignment rows and silently
    # "complete"; instead FAIL LOUDLY, telling which env vars to check.
    # Single-owner (LLM-less, deterministic) documents are ALREADY
    # outside this condition -- if `llm_jobs` is empty (no multi-owner
    # documents), this check is NEVER TRIGGERED.
    if llm_jobs and n_err == len(llm_jobs):
        raise RuntimeError(
            f"ownership: ALL {len(llm_jobs)} LLM calls failed "
            f"(model={MODEL!r}, url={OLLAMA_URL!r}) -- check OWNERSHIP_LLM_MODEL/"
            "OWNERSHIP_LLM_BASE_URL env vars (is Ollama reachable from this container?)."
        )

    elapsed_s = round(time.monotonic() - t0, 1)
    out = {
        "meta": {
            "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
            "max_workers": MAX_WORKERS, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "n_docs_single_owner": len(single_docs), "n_docs_multi_owner": len(multi_docs),
            "n_llm_chunks": len(llm_jobs), "n_llm_errors": n_err,
            "n_rows": len(rows), "llm_elapsed_s": elapsed_s,
        },
        "docs_without_chunk_text": docs_without_chunk_text,
        "rows": rows,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "chunk_ownership_all.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {len(rows)} rows ({n_err} LLM errors, {len(docs_without_chunk_text)} docs-without-chunk-text) -> {out_path}")
    return out


if __name__ == "__main__":
    main()