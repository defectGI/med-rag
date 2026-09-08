"""Re-tries one by one the chunks `build_all.py` left behind with `error`
(model responses that broke the JSON schema), WITHOUT re-running the
whole job.

Additionally uses a MORE TOLERANT parser than `_parse_assignments`: it
also rescues broken shapes `build_all.py` saw -- `{"products": [...]}`
(the candidate list echoed back in the SAME shape) and a flat
`{code: display_name}` map. If the second attempt also fails, the row
stays with `error` -- NOT silently skipped.

WARNING (caught in a live run on 2026-08-04): these two shapes can be
an error mode where the model echoes the candidate metadata back
WITHOUT having actually read the chunk text -- in PN5071_BROCHURE we
saw: 44 "assignments", ALL with the same fixed confidence, `evidence`
field was NOT a chunk quote but the candidate's OWN `display_name`
(`_lenient_parse` was using that as a fallback -- i.e. FABRICATING it
-- the model never gave that evidence). Those rows had leaked into the
DB as `accepted=true` and were manually rolled back (see git log).
`_lenient_parse` NO LONGER FABRICATES `evidence` -- if there is no real
evidence, that entry is DROPPED; if the entire list drops, a
ProviderError is raised (the chunk stays unresolved).

Usage:
  cd facts/experiments/catalog_chunk_ownership
  python retry_failed.py
"""
from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

from medrag.core.paths import resolve_specs_db_path
from medrag.pipeline.facts.catalog_chunk_ownership import build_all as _build_all

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/catalog_chunk_ownership/ (code, O-03)
REPO_ROOT = HERE.parents[4]  # medrag/ (catalog_chunk_ownership/facts/pipeline/urun/src/<repo>)
# DATA (db/) was NOT moved with the code -- O-03 scope was code only;
# same principle as `discover.py::_FACTS_DATA_ROOT`: lives in `facts/`
# at the repo root.
DB_PATH = resolve_specs_db_path()
RESULTS_PATH = HERE / "results" / "chunk_ownership_all.json"
CHUNKER_OUTPUT_ROOT = Path(REPO_ROOT / "chunker" / "storage")
ALL_CHUNKS_PATH = CHUNKER_OUTPUT_ROOT / "all_chunks.json"

# Model/limit settings are INHERITED from build_all.py (2026-08-06) --
# when the two were kept separately they silently drifted: build_all's
# TIMEOUT_S went 180->400 and NUM_PREDICT 4096->8192, this file stayed
# at the old values, so retry would re-hit the SAME failures build_all
# just fixed.
MODEL = _build_all.MODEL
NUM_CTX = _build_all.NUM_CTX
NUM_PREDICT = _build_all.NUM_PREDICT
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT_S = _build_all.TIMEOUT_S
CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (HERE / "prompts" / "chunk_owner_system.md").read_text(encoding="utf-8")


class ProviderError(RuntimeError):
    pass


def _chat_ollama(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "format": "json", "think": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        yanit = json.loads(resp.read().decode("utf-8"))
    return yanit["message"]["content"] or ""


def _lenient_parse(content: str, valid_codes: set[str]) -> list[dict]:
    """Rescues broken shapes outside `{"assignments":[...]}` seen in
    practice: `{"products": [{"product_code":..., "confidence":...,
    "evidence":...}, ...]}` (the candidate list echoed back in the SAME
    shape). ONLY entries where the model actually wrote its OWN
    `confidence`/`evidence` fields are accepted -- we do NOT fall
    back to `display_name` or a fixed value (2026-08-04 measurement:
    this fallback produced 44 fake assignments in PN5071_BROCHURE, all
    with the same fixed confidence + evidence=candidate's own
    display_name, meaning the model echoed the candidate back without
    reading the chunk). A flat `{code: name}` map is also NO LONGER
    ACCEPTED for the same reason."""
    for candidate in (content, content[content.find("{"): content.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("assignments"), list):
            return data["assignments"]
        if isinstance(data.get("products"), list):
            items = [
                p for p in data["products"]
                if isinstance(p, dict) and p.get("product_code") in valid_codes
                and isinstance(p.get("confidence"), (int, float))
                and isinstance(p.get("evidence"), str) and p["evidence"].strip()
                and p["evidence"].strip() != p.get("display_name")
            ]
            if items:
                return [{"product_code": p["product_code"], "confidence": p["confidence"],
                         "evidence": p["evidence"]} for p in items]
            continue
    raise ProviderError(f"lenient parse also failed (no real evidence): {content[:300]}")


def main() -> None:
    out = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    error_rows = [r for r in out["rows"] if r.get("error")]
    if not error_rows:
        print("no failed rows, nothing to do")
        return
    print(f"{len(error_rows)} failed chunks will be retried")

    all_chunks = json.loads(ALL_CHUNKS_PATH.read_text(encoding="utf-8"))["documents"]
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    doc_owner_rows = con.execute("SELECT doc_id, product_codes FROM document").fetchall()
    owners_by_doc = {r["doc_id"]: json.loads(r["product_codes"] or "[]") for r in doc_owner_rows}

    new_rows = [r for r in out["rows"] if not r.get("error")]
    still_failing = []
    for err_row in error_rows:
        doc_id, chunk_id = err_row["doc_id"], err_row["chunk_id"]
        codes = owners_by_doc.get(doc_id, [])
        valid_codes = set(codes)
        prod_rows = con.execute(
            f"SELECT product_code, display_name, family, subfamily, subfamily_2, acme_code FROM product "
            f"WHERE product_code IN ({','.join('?' * len(codes))})", codes,
        ).fetchall() if codes else []
        candidates = [dict(r) for r in prod_rows]

        doc = all_chunks.get(doc_id, {})
        node = next((n for n in doc.get("nodes", []) if n.get("node_id") == chunk_id), None)
        if node is None:
            still_failing.append(err_row)
            print(f"  {chunk_id}: node not found, skipping")
            continue
        text = node.get("text") or ""
        ps, pe = node.get("page_start"), node.get("page_end")
        page_str = str(ps) if ps == pe else f"{ps}-{pe}"
        # KARAR-064: SAME signals as build_all.py. Without them retry
        # chunks would land span-less -- and those are EXACTLY the ones
        # with the highest ownership where span is most needed.
        headings = _build_all.headings_in(text)
        hp = node.get("heading_path") or []
        path_line = (" > ".join(hp) if hp
                     else "(EMPTY - this chunk merged sections with no common ancestor; "
                          "treat it as likely MIXED and attribute section by section)")
        headings_line = json.dumps(headings, ensure_ascii=False) if headings else "(none)"
        user = (
            f"candidates:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"heading_path: {path_line}\n"
            f"headings_in_chunk: {headings_line}\n\n"
            f"Chunk text (page {page_str}):\n{text}"
        )

        try:
            content = _chat_ollama(SYSTEM_PROMPT, user)
            assignments = _lenient_parse(content, valid_codes)
        except (ProviderError, urllib.error.URLError, TimeoutError) as exc:
            still_failing.append(err_row)
            print(f"  {chunk_id}: still failing -- {exc}")
            continue

        n_added = 0
        for a in assignments:
            code = a.get("product_code")
            if code not in valid_codes:
                continue
            conf = a.get("confidence")
            accepted = isinstance(conf, (int, float)) and conf >= CONFIDENCE_THRESHOLD
            new_rows.append({
                "doc_id": doc_id, "file_name": err_row["file_name"], "chunk_id": chunk_id,
                "page_start": ps, "page_end": pe, "product_code": code,
                "confidence": conf, "accepted": accepted, "evidence": a.get("evidence"),
                "heading_anchor": _build_all.normalize_anchor(a, headings),
                "source": "llm_retry",
            })
            n_added += 1
        print(f"  {chunk_id}: rescued, {n_added} assignments")

    out["rows"] = new_rows + still_failing
    out["meta"]["n_llm_errors"] = len(still_failing)
    out["meta"]["n_rows"] = len(out["rows"])
    RESULTS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {len(error_rows) - len(still_failing)}/{len(error_rows)} rescued, "
          f"{len(still_failing)} still failing -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()