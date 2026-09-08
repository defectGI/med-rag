"""Full run -- Strategy B (chunk + dictionary -> fact JSON) for the WHOLE
corpus, ONE call per product. Strategy A (atom) is NOT IN THIS SCRIPT --
user decision: atom strategy will not be used again; only the chunk
prompt from `atom_vs_chunk_bench` and the `run_bench.py` Anthropic client
were MOVED HERE, EXTENDED.

Differences from atom_vs_chunk_bench:
  - 4 fixed products vs EVERY product in specs.db (or those passed via --models).
  - Chunk source: `discover.py` (all_chunks.json if present, otherwise
    atoms.ndjson bridge + facts/snapshots) -- see that module's
    docstring; LIMITED SCOPE (today ~231/250 products may be partially/
    fully covered on this machine; run `discover.py` alone for the
    current count).
  - When a product has MANY chunks (BATCH_SIZE_CHUNKS exceeded) it is NOT
    crammed into a single call -- it is split across multiple calls and
    facts[] are merged (avoids max_tokens truncation; generalises the
    PN1071/61-chunk truncation risk called out in atom_vs_chunk_bench's
    README).
  - ONE JSON per product (`results/<PRODUCT_CODE>.json`) +
    `results/_run_summary.json` at run end (covered/skipped product
    counts, total usage).

This script STILL does NOT import facts.extract (deliberately kept
separate from the experiment/bridge code, same rationale as
`run_bench.py`'s production OpenAICompatExtractor -- prevents an LLM-call
format error from corrupting specs.db). specs.db is NEVER WRITTEN TO
(read-only connection, discover.py). The step that writes results TO
the DB is a SEPARATE script: `load_to_db.py` (PHASE C, deliberately
separate -- an LLM call / format error must not leave specs.db HALF-WRITTEN).

Usage (facts/.env must be filled -- FACTS_LLM_PROVIDER=anthropic, see
root CLAUDE.md: this machine DOES NOT run LLM inference; GPU/remote
inference, not local):
  cd facts/experiments/chunk_full_run
  python run_full.py                          # all products in specs.db
  python run_full.py --models PN1071,PN1204  # subset (trial/verification)
  python run_full.py --limit 20               # first N products (gradual run)
  python run_full.py --skip-existing          # skip products whose results/ file already exists (resume)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/ (code, O-03)
_REPO_ROOT = HERE.parents[3]  # medrag/ (facts/pipeline/urun/src/<repo>)
# `atom_vs_chunk_bench` is OUT of the K-59 scope and did NOT return --
# this variable now only exists to back `_FROZEN_SPEC_KEYS`'s (below,
# unused in practice) fallback path.
BENCH_DIR = _REPO_ROOT / "facts" / "experiments" / "atom_vs_chunk_bench"
# DATA (results/, .env) was NOT moved with the code -- O-03 scope was code
# only; same principle as `discover.py::_FACTS_DATA_ROOT`: lives in `facts/`
# at the repo root.
_FACTS_DATA_ROOT = _REPO_ROOT / "facts"
# Old location was facts/experiments/chunk_full_run/results/ -- data
# directory name (chunk_full_run/) is preserved, only the `experiments/`
# prefix on the code side dropped (see load_to_db.py's SAME RESULTS_DIR).
#
# N-22: `FACTS_RESULTS_DIR` -- SAME env var as `load_to_db.py::RESULTS_DIR`
# (default unchanged). In the container this path is NOT bundled with
# the code tree, so it can be moved to a host-persistent root (`/corpus`
# etc.) -- same pattern as `CHUNKER_OUTPUT_ROOT`/`NIGHTLY_BACKBACKUP`.
RESULTS_DIR = Path(os.environ.get(
    "FACTS_RESULTS_DIR", str(_FACTS_DATA_ROOT / "chunk_full_run" / "results")
))
ENV_PATH = _FACTS_DATA_ROOT / ".env"

from medrag.pipeline.facts.discover import (
    DEFAULT_ATOMS_BRIDGE,
    ProductManifest,
    _connect_ro,
    all_product_codes,
    build_product_manifest,
    load_indexes,
    owner_count_by_doc,
    products_for_doc_id,
)
from medrag.pipeline.facts.llm_schema import build_llm_schema_text

# 2026-08-04: prompt NO LONGER read from bench; read from this experiment's
# OWN file (`prompts/chunk_single_system.md`). Reason: the move to
# single-chunk mode changed the prompt's input contract (the
# `chunk_sha256` field was DROPPED, "ONE chunk" wording came in) --
# bench's prompt is a frozen part of ITS 4-product comparison; changing
# it would invalidate that benchmark's past results. The two files are
# DELIBERATELY SEPARATE; the shared text is a copy.
SYSTEM_PROMPT = (HERE / "prompts" / "chunk_single_system.md").read_text(encoding="utf-8")
# Dictionary is read from the CANONICAL source
# (tools/spec_schema/spec_keys.yaml) -- not bench's frozen copy, because
# this large run can take weeks and the dictionary may be updated in
# between. `llm_schema.build_llm_schema_text` drops fields the LLM
# does NOT actually use (evidence/facets/relations/...) -- nothing
# code-fillable is sent to the LLM (user decision).
_CANONICAL_SPEC_KEYS = _REPO_ROOT / "tools" / "spec_schema" / "spec_keys.yaml"
_FROZEN_SPEC_KEYS = BENCH_DIR / "schema_context" / "spec_keys.yaml"
SPEC_KEYS_PATH = _CANONICAL_SPEC_KEYS if _CANONICAL_SPEC_KEYS.is_file() else _FROZEN_SPEC_KEYS
SCHEMA_TEXT = build_llm_schema_text(SPEC_KEYS_PATH)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when not available
    tqdm = None


def _progress(iterable, *, total: int, desc: str):
    """Real progress bar when tqdm is available, otherwise plain
    "[i/N]" lines. Both flush AFTER each step, so the last completed
    step before a crash is always visible."""
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc, unit="product", dynamic_ncols=True)
        return
    for i, item in enumerate(iterable, start=1):
        print(f"[{i}/{total}] ", end="", flush=True)
        yield item

# Chunks per request -- reduced from 20 (ollama) to 1 (user decision on
# 2026-08-04). Old design required the model to MARK which chunk each
# fact belonged to by COPYING its 64-hex sha256; measurement showed
# gemma4:26b mangled this in 41.4% of facts and 39.6% of filled rows
# in the DB ended up SOURCE-LESS. In single-chunk mode attribution is
# NOT ASKED, code STAMPS it -- this class of error is structurally
# impossible. Side benefits: cross-chunk fact mixing stops, output is
# shorter (no num_predict multiplication risk), and the "batch JSON
# contract abandonment" recovery path is no longer needed. Cost: ~20x
# more requests. To compensate the dictionary (~8.6k tokens) was moved
# from user to SYSTEM message -- prefix is constant so Ollama's KV cache
# does not re-evaluate it (see build_system_message).
CHUNKS_PER_REQUEST = 1
MAX_WORKERS = 8  # Same value as `catalog_chunk_ownership/build_all.py` --
# same GPU runs 261 chunks in 449s without issue. Chunks-per-prompt is
# small, so parallelism (not context size) is what gives throughput.
# On the first full-corpus ollama run (2026-08-03) the 8192 output band
# caused truncation in 48/249 products ("cannot parse facts JSON" --
# the JSON was cut in half, multiplied by num_predict). Raised to 16384
# (comfortably inside num_ctx=65536). In single-chunk mode a call's
# output is much shorter, the band has plenty of room.
MAX_TOKENS_BY_PROVIDER = {"anthropic": 32000, "ollama": 16384}


class ProviderError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_llm_env(env: dict) -> tuple[str, str, str, str | None]:
    """Same as run_bench.py::_resolve_llm_env (deliberate copy; see that
    module's docstring for the rationale: every pass/experiment keeps
    its own copy so facts.extract does not have to be imported)."""
    provider = (env.get("FACTS_LLM_PROVIDER") or "").strip().lower()
    if not provider:
        raise ProviderError("FACTS_LLM_PROVIDER is not set -- fill facts/.env (FACTS_LLM_PROVIDER=anthropic).")
    model = (env.get("FACTS_LLM_MODEL") or "").strip()
    if not model:
        raise ProviderError("FACTS_LLM_MODEL is not set")
    default_urls = {
        "ollama": "http://localhost:11434/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
    }
    base_url = (env.get("FACTS_LLM_BASE_URL") or "").strip() or default_urls.get(provider, "")
    if not base_url:
        raise ProviderError(f"FACTS_LLM_BASE_URL is not set (no known default for provider {provider})")
    return provider, model, base_url, (env.get("FACTS_LLM_API_KEY") or None)


def _chat_anthropic(model: str, base_url: str, api_key: str | None, system: str, user: str) -> tuple[str, dict]:
    think_on = (os.environ.get("FACTS_LLM_THINKING_ON") or "").strip().lower() in ("1", "true", "yes", "on")
    payload = {
        "model": model, "max_tokens": MAX_TOKENS_BY_PROVIDER["anthropic"], "system": system,
        "thinking": {"type": "adaptive"} if think_on else {"type": "disabled"},
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(f"{base_url}/messages", data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("x-api-key", api_key or "")
    try:
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            yanit = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {base_url}/messages: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"cannot reach {base_url}/messages: {exc}") from exc
    try:
        content = "".join(b["text"] for b in yanit["content"] if b.get("type") == "text")
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"unexpected chat response shape: {str(yanit)[:500]}") from exc
    usage = dict(yanit.get("usage", {}))
    if yanit.get("stop_reason"):
        usage["stop_reason"] = yanit["stop_reason"]
    return content, usage


def _chat_ollama(model: str, base_url: str, system: str, user: str) -> tuple[str, dict]:
    """Same as run_bench.py::_chat's ollama branch -- native `/api/chat`
    (per-request `num_ctx`), `format=json` (grammar-constrained,
    reinforces the prompt's "ONLY JSON" directive), `think` tied to
    FACTS_LLM_THINKING_ON (DEFAULT OFF -- JSON-only response should not
    be mixed with thinking text)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS_BY_PROVIDER["ollama"]},
        "format": "json",
    }
    num_ctx = (os.environ.get("FACTS_LLM_NUM_CTX") or "").strip()
    payload["options"]["num_ctx"] = int(num_ctx) if num_ctx else 16384
    think_on = (os.environ.get("FACTS_LLM_THINKING_ON") or "").strip().lower() in ("1", "true", "yes", "on")
    payload["think"] = think_on
    native_base = base_url.removesuffix("/v1")
    req = urllib.request.Request(
        f"{native_base}/api/chat", data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            yanit = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {native_base}/api/chat: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"cannot reach {native_base}/api/chat: {exc}") from exc
    try:
        content = yanit["message"]["content"] or ""
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"unexpected chat response shape: {str(yanit)[:500]}") from exc
    usage = {k: yanit[k] for k in ("prompt_eval_count", "eval_count", "total_duration") if k in yanit}
    if yanit.get("done_reason"):
        usage["stop_reason"] = yanit["done_reason"]  # 'length' = truncated response (same meaning as anthropic's 'max_tokens')
    return content, usage


def _chat(provider: str, model: str, base_url: str, api_key: str | None, system: str, user: str) -> tuple[str, dict]:
    if provider == "ollama":
        return _chat_ollama(model, base_url, system, user)
    return _chat_anthropic(model, base_url, api_key, system, user)


def _parse_facts(content: str) -> list[dict]:
    for candidate in (content, content[content.find("{"): content.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("facts"), list):
            return data["facts"]
    raise ProviderError(f"cannot parse facts JSON from model reply: {content[:500]}")


def build_system_message() -> str:
    """The dictionary is now in the SYSTEM message (2026-08-04) -- not
    the user message. Reason: in single-chunk-per-request mode
    (~8.6k token dictionary) each request would re-send it; the system
    message is BYTE-IDENTICAL across the run so Ollama's KV prefix
    cache does not re-evaluate it. Content UNCHANGED, only the role --
    no behavioural difference."""
    return f"{SYSTEM_PROMPT}\n\n---\n\nAttribute dictionary (spec_keys.yaml):\n{SCHEMA_TEXT}"


def build_user_message(manifest: ProductManifest, chunk) -> str:
    """SINGLE chunk -- `chunk_sha256` is NOT IN THE PROMPT (user decision,
    2026-08-04). Old design sent 20 chunks per request, so the model had
    to MARK which chunk each fact belonged to by COPYING the 64-hex
    sha256 -- measurement showed gemma4:26b mangled this in 41.4% of
    facts (lost/extra characters, control-token leaks like `<|thought|>`),
    leaving 39.6% of filled rows in the DB SOURCE-LESS. Now there is
    one chunk per request and `source_chunk_id` is STAMPED by code (see
    `_call_chunk`)."""
    product_ctx = json.dumps(
        {"model": manifest.product_code, "family": manifest.family, "subfamily": manifest.subfamily,
         "title": manifest.display_name},
        ensure_ascii=False,
    )
    return f"Product context:\n{product_ctx}\n\nChunk text:\n{chunk.text}"


_PROMPT_SOURCE_FIELDS = ("source_chunk_id", "chunk_sha256", "source_doc_id", "chunk_id")


def _call_chunk(
    provider: str, model: str, base_url: str, api_key: str | None, manifest: ProductManifest,
    chunk, label: str, system: str,
) -> tuple[list[dict], list[dict]]:
    """ONE chunk = ONE request. Batch-split recovery is GONE -- that
    mechanism (see git history) was for splitting a 20-chunk batch when
    the model abandoned the JSON contract; with one chunk per request
    there is nothing to split. A failed chunk is left fact-less, but
    NOT silently: it appears in the report with `error` + `chunk_sha256`.

    `source_chunk_id` is STAMPED BY THIS FUNCTION: regardless of what
    the model returns, the correct sha is written, and every source
    field the model sent is DROPPED (`_PROMPT_SOURCE_FIELDS`). So the
    mangled-sha bug class is structurally impossible."""
    t0 = time.monotonic()
    user = build_user_message(manifest, chunk)
    try:
        content, usage = _chat(provider, model, base_url, api_key, system, user)
        raw_facts = _parse_facts(content)
    except ProviderError as exc:
        return [], [{
            "batch_label": label, "n_chunks": 1, "error": str(exc),
            "chunk_sha256": chunk.chunk_sha256, "elapsed_s": round(time.monotonic() - t0, 1),
        }]
    facts = []
    n_model_source = 0
    for fact in raw_facts:
        if not isinstance(fact, dict):
            facts.append(fact)  # load_to_db.py filters non-object entries in its own report
            continue
        for field_name in _PROMPT_SOURCE_FIELDS:
            if field_name in fact:
                n_model_source += 1
                fact.pop(field_name)
        fact["source_chunk_id"] = chunk.chunk_sha256
        facts.append(fact)
    report = {
        "batch_label": label, "n_chunks": 1, "n_facts": len(facts),
        "chunk_sha256": chunk.chunk_sha256, "usage": usage,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    if n_model_source:
        # Prompt no longer REQUESTS a source field; if the model still
        # sends one it is a prompt-compliance diagnostic (it is dropped,
        # so harmless) -- keep it visible.
        report["model_sent_source_fields"] = n_model_source
    return facts, [report]


def _is_partial(out_path: Path) -> bool:
    """`--skip-existing` must NOT skip a HALF-FINISHED product (2026-08-06).

    Checkpoint is written per chunk, and a product stays `partial=true`
    until done. Old code looked at FILE EXISTENCE only: if the run was
    killed at 3/11 chunks, the resume would say "already exists" and
    SKIP that product, leaving it silently incomplete. An unreadable
    or corrupted file is also treated as partial (re-processed) -- we
    don't trust a half-written JSON."""
    try:
        return bool(json.loads(out_path.read_text(encoding="utf-8")).get("partial"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return True


def run_one_product(
    provider: str, model: str, base_url: str, api_key: str | None, manifest: ProductManifest, out_path: Path,
    *, system: str | None = None,
) -> dict:
    """Writes to `out_path` AFTER EVERY CHUNK (checkpoint) -- if the run
    is killed mid-product (crash/Ctrl+C/network), the facts processed
    up to that point are NOT LOST, only marked with `partial=true` and
    `load_to_db.py` can see and warn. Checkpoint is now per chunk
    (formerly per batch) -- a natural consequence of single-chunk mode,
    granularity INCREASED."""
    base = {
        "strategy": "chunk", "llm_provider": provider, "llm_model": model,
        "model_code": manifest.product_code, "run_at": _now(),
        "family": manifest.family, "subfamily": manifest.subfamily,
        "doc_ids": manifest.doc_ids, "docs_missing_text": manifest.docs_missing_text,
        "docs_excluded_fanout": manifest.docs_excluded_fanout,
        # KARAR-064: which chunks were sliced to which anchor / which
        # anchors could not be found. No silent skipping -- if no
        # slicing happened, the chunk went in full.
        "anchors_not_found": manifest.anchors_not_found,
        "chunks_sliced": [
            {"chunk_sha256": c.chunk_sha256, "anchors": c.anchors_applied}
            for c in manifest.chunks if c.anchors_applied
        ],
    }
    if not manifest.chunks:
        reason = ("no chunk text available" if not manifest.docs_excluded_fanout
                  else "no chunk text available (multi-owner documents excluded because chunk_ownership_index is missing)")
        result = {**base, "facts": [], "batches": [], "skipped_reason": reason}
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    system = system if system is not None else build_system_message()
    all_facts: list[dict] = []
    batch_reports: list[dict] = []
    result = {**base, "n_chunks_sent": len(manifest.chunks), "facts": all_facts,
              "batches": batch_reports, "partial": True}

    # Chunks are sent IN PARALLEL (2026-08-06). They used to be serial,
    # measurement: after KARAR-064 the per-product chunk count grew
    # (facts 26 -> 44-65), per-product time ~93s -> ~210s, and a 100-
    # product run hit ~6 HOURS. `build_all.py` runs 8 parallel requests
    # on the same GPU without issue (261 chunks / 449s) -- same pattern
    # moved here. Results are written by the MAIN thread only
    # (`as_completed` loop), so the checkpoint file is still single-
    # writer and updated per chunk: Ctrl+C/crash behaviour UNCHANGED.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_call_chunk, provider, model, base_url, api_key, manifest, chunk, str(i), system): i
            for i, chunk in enumerate(manifest.chunks)
        }
        for fut in as_completed(futures):
            facts, sub_reports = fut.result()
            all_facts.extend(facts)
            batch_reports.extend(sub_reports)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["partial"] = False
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def resolve_codes(
    con, *, models: str | None = None, doc_id: str | None = None, limit: int | None = None,
) -> list[str]:
    """O-12: document/product selector layer for the entry point --
    `--models` (comma-separated code list), `doc_id` (all products that
    USE this document, `discover.products_for_doc_id`), or neither (ALL
    products in specs.db, old bulk-run behaviour, UNCHANGED). When both
    `models` and `doc_id` are given, the INTERSECTION is taken -- for the
    "this document changed BUT only try these products" scenario."""
    if doc_id:
        codes = products_for_doc_id(con, doc_id)
        if models:
            wanted = {c.strip() for c in models.split(",")}
            codes = [c for c in codes if c in wanted]
    elif models:
        codes = [c.strip() for c in models.split(",")]
    else:
        codes = all_product_codes(con)
    if limit:
        codes = codes[:limit]
    return codes


def run_products(
    codes: list[str],
    *,
    con,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None,
    all_chunks_index: dict,
    atoms_bridge_index: dict,
    exclude_doc_types: frozenset,
    owner_counts: dict,
    chunk_ownership_index: dict,
    chunk_anchor_index: dict,
    results_dir: Path = RESULTS_DIR,
    skip_existing: bool = False,
    system: str | None = None,
) -> dict:
    """O-12: the actual work loop -- `main()`'s argparse logic DECOUPLED.
    `codes` can also be a list with a single product code (equivalent to
    calling `run_one_product` directly, just with the
    checkpoint/summary/error-isolation discipline) -- this function is
    the programmatic entry point group N's nightly incremental run
    (re-process products affected by a doc_id change) can call directly;
    it does not depend on the CLI.

    Phase replace semantics: this function does NOT WRITE TO specs.db
    (run_full.py module docstring) -- "phase, when re-run, invalidates
    its own old derivatives" is applied on the
    `load_to_db.load_product(..., force=True)` side (see that module's
    `load_product_incremental`); here only FRESH fact JSON is produced,
    and `results/<CODE>.json`'s OLD content (if any) is ALWAYS OVERWRITTEN
    IN FULL (see `run_one_product`, unless `--skip-existing` is given)."""
    system = system if system is not None else build_system_message()
    # parents=True: in the container the INTERMEDIATE directory
    # (`facts/chunk_full_run/`) does NOT exist -- `.dockerignore` only
    # takes `db/` under `facts/`. Locally the directory already exists so
    # this error was never seen on this machine; the first real
    # container run on 2026-08-25 crashed the facts phase with
    # FileNotFoundError. `exist_ok` alone only says "no problem if the
    # last directory exists", not "create the missing intermediate".
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "_run_summary.json"
    summary = {"run_at": _now(), "model": model, "n_products_requested": len(codes),
               "n_processed": 0, "n_skipped_existing": 0, "n_no_chunks": 0, "n_errors": 0,
               "n_docs_missing_text_total": 0, "total_usage": {"input_tokens": 0, "output_tokens": 0},
               "errored_products": []}

    def _write_summary() -> None:
        # Re-written after each product (checkpoint, same discipline as
        # run_one_product's per-batch checkpoint) -- no matter when the
        # run is cut, _run_summary.json reflects the real state up to
        # that point.
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    bar = _progress(codes, total=len(codes), desc="products")
    interrupted = False
    try:
        for code in bar:
            out_path = results_dir / f"{code}.json"
            if skip_existing and out_path.is_file() and not _is_partial(out_path):
                summary["n_skipped_existing"] += 1
                continue

            try:
                manifest = build_product_manifest(
                    con, code, all_chunks_index=all_chunks_index, atoms_bridge_index=atoms_bridge_index,
                    exclude_doc_types=exclude_doc_types, owner_counts=owner_counts,
                    chunk_ownership_index=chunk_ownership_index,
                    chunk_anchor_index=chunk_anchor_index,
                )
                result = run_one_product(provider, model, base_url, api_key, manifest, out_path, system=system)
            except Exception as exc:  # noqa: BLE001 -- an UNEXPECTED error per product
                # (code error, disk full, etc.) must NOT DROP the entire
                # run -- the run can take hours; losing hundreds of
                # products for one bug is unacceptable. The error is
                # written to JSON, the next product continues; with
                # --skip-existing that product (out_path not written)
                # is AUTOMATICALLY retried.
                summary["n_errors"] += 1
                summary["errored_products"].append({"model_code": code, "error": f"{type(exc).__name__}: {exc}"})
                _write_summary()
                continue

            if result.get("skipped_reason"):
                summary["n_no_chunks"] += 1
            else:
                summary["n_processed"] += 1
                for b in result["batches"]:
                    u = b.get("usage", {})
                    # Anthropic: input_tokens/output_tokens. Ollama: prompt_eval_count/
                    # eval_count (different field names, same meaning) -- both accumulated.
                    summary["total_usage"]["input_tokens"] += (u.get("input_tokens") or u.get("prompt_eval_count") or 0)
                    summary["total_usage"]["output_tokens"] += (u.get("output_tokens") or u.get("eval_count") or 0)
            summary["n_docs_missing_text_total"] += len(result.get("docs_missing_text", []))
            _write_summary()
    except KeyboardInterrupt:
        # Ctrl+C: close the half-finished run with a clean summary
        # instead of a traceback -- every product processed up to that
        # point is already on disk (the checkpoints above); only the
        # single call that was in flight at the moment is lost.
        interrupted = True

    summary["interrupted"] = interrupted
    _write_summary()
    if interrupted:
        summary["_interrupted_hint"] = (
            f"User cancelled (Ctrl+C) -- re-run with --skip-existing to resume "
            f"from where it stopped ({summary_path})."
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="Comma-separated product_code list (default: all in specs.db).")
    ap.add_argument("--doc-id", default=None,
                    help="O-12: process only products that USE this document (nightly incremental run -- "
                         "'today's changed document' entry). Combined with --models, the INTERSECTION is taken.")
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N products (for gradual runs).")
    ap.add_argument("--skip-existing", action="store_true", help="If results/<CODE>.json already exists, skip that product.")
    ap.add_argument("--atoms-bridge", default=str(DEFAULT_ATOMS_BRIDGE), help="discover.py's temporary bridge source.")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    provider, model, base_url, api_key = _resolve_llm_env(os.environ)
    if provider not in ("anthropic", "ollama"):
        raise ProviderError(
            f"this script only supports FACTS_LLM_PROVIDER=anthropic|ollama (given: {provider!r}) -- "
            "run_bench.py's openai/openrouter branches were DELIBERATELY NOT carried over."
        )
    system = build_system_message()
    print(f"provider={provider} model={model} base_url={base_url} "
          f"({CHUNKS_PER_REQUEST} chunk/call, dictionary in system message: ~{len(system) // 4} token constant prefix)")

    con = _connect_ro()
    all_chunks_index, atoms_bridge_index, exclude_doc_types, chunk_ownership_index, chunk_anchor_index = load_indexes(Path(args.atoms_bridge))
    owner_counts = owner_count_by_doc(con)
    n_multi_owner = sum(1 for v in owner_counts.values() if v >= 2)
    print(f"all_chunks.json: {'PRESENT' if all_chunks_index else 'MISSING (bridge used)'}")
    print(f"multi-owner (>=2) documents: {n_multi_owner}/{len(owner_counts)} -- "
          + (f"narrowed via chunk_ownership_index ({len(chunk_ownership_index)} accepted (doc,chunk))"
             if chunk_ownership_index else "chunk_ownership_index MISSING, EXCLUDED entirely"))

    codes = resolve_codes(con, models=args.models, doc_id=args.doc_id, limit=args.limit)
    if args.doc_id:
        print(f"--doc-id={args.doc_id}: {len(codes)} products affected -> {codes}")

    summary = run_products(
        codes, con=con, provider=provider, model=model, base_url=base_url, api_key=api_key,
        all_chunks_index=all_chunks_index, atoms_bridge_index=atoms_bridge_index,
        exclude_doc_types=exclude_doc_types, owner_counts=owner_counts,
        chunk_ownership_index=chunk_ownership_index, chunk_anchor_index=chunk_anchor_index,
        skip_existing=args.skip_existing, system=system,
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    hint = summary.pop("_interrupted_hint", None)
    if hint:
        print(f"\n{hint}")


if __name__ == "__main__":
    main()