"""Carries over, from the PREVIOUS table, the assignments of chunks that the NEW
run could NOT solve but the PREVIOUS run did -- only for NON-MIXED chunks (2026-08-06).

WHY: in the 2026-08-06 ownership run (PROTOCOL KARAR-064) `PN5085_CATALOGUE`
`::c17` chunk first timed out, then returned half JSON on retry and fell in
the new table from 45 to **0** products. This is a REGRESSION: PN1260's
`storage_temperature -55..125` value comes from exactly that chunk (a value
rescued that same day with ROADMAP 20a). If extraction runs with this table,
the PN1253 family would lose that value.

WHY LEGITIMATE: the carry-over is done ONLY for chunks whose `heading_path` is
NON-EMPTY (i.e. they have a common ancestor, SINGLE-section). Such a chunk does
NOT need a span -- the only novelty KARAR-064 brought was splitting mixed chunks
into sections; in a single-section chunk the answer the old and new prompt MUST
produce is the same. A mixed chunk's assignment is NEVER carried over (there
the old answer may already be flawed -- that is what we are trying to fix).

Carried-over rows are MARKED with `source="carried_over"`: no silent drift, and
anyone looking at the table can see which row comes from this run and which
from the previous one. (Same principle as KARAR-062's `manual_family_wide` mark.)

Usage:
  cd facts/experiments/catalog_chunk_ownership
  python carry_over_unresolved.py --old <previous_table.json>          # DRY RUN
  python carry_over_unresolved.py --old <previous_table.json> --apply

O-07 (2026-08-20): with `--apply`, the `skipped` list (chunks still
unsolved) is also written to the `issues` table (`issues_bridge.py`,
`UNRESOLVED_CHUNK` code) -- no silent drop, each becomes individually visible
in the panel (M-09) (I-14).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from medrag.core.db.issues import connect_issues
from medrag.core.paths import resolve_issues_db_path
from medrag.pipeline.facts.issues_bridge import record_unresolved_chunks

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/catalog_chunk_ownership/ (code, O-03)
REPO_ROOT = HERE.parents[4]  # medrag/ (catalog_chunk_ownership/facts/pipeline/product/src/<repo>)
RESULTS_PATH = HERE / "results" / "chunk_ownership_all.json"
ALL_CHUNKS = REPO_ROOT / "chunker" / "storage" / "all_chunks.json"
HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def load_nodes() -> dict[str, dict]:
    data = json.loads(ALL_CHUNKS.read_text(encoding="utf-8"))["documents"]
    return {n["node_id"]: n
            for cs in data.values()
            for n in cs.get("nodes", []) if n.get("tree_level", 0) == 0}


def plan(new: dict, old: dict, nodes: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """`(rows to carry, errors not to carry)`."""
    old_by_chunk: dict[str, list[dict]] = {}
    for r in old.get("rows", []):
        if r.get("accepted") and r.get("product_code"):
            old_by_chunk.setdefault(r["chunk_id"], []).append(r)

    carried, skipped = [], []
    for err in [r for r in new["rows"] if r.get("error")]:
        cid = err["chunk_id"]
        node = nodes.get(cid)
        prior = old_by_chunk.get(cid, [])
        if node is None:
            skipped.append({**err, "_why": "chunk metni bulunamadi"})
            continue
        if not node.get("heading_path"):
            # No common ancestor -> likely mixed -> the old answer
            # may already be flawed, DON'T CARRY.
            skipped.append({**err, "_why": "KARMA chunk (heading_path bos) -- tasinmaz"})
            continue
        if not prior:
            skipped.append({**err, "_why": "onceki tabloda da cozulmemis"})
            continue
        for r in prior:
            carried.append({**r, "source": "carried_over"})
    return carried, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--apply", action="store_true", help="Gercekten yaz (varsayilan: dry run).")
    ap.add_argument("--issues-db", default=str(resolve_issues_db_path()),
                    help="Cozulememis chunk'larin yazilacagi issues.db yolu (O-07, "
                         "varsayilan: ISSUES_DB_PATH ortam degiskeni, 2026-08-26 fix).")
    args = ap.parse_args()

    new = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    carried, skipped = plan(new, old, load_nodes())

    by_chunk: dict[str, int] = {}
    for r in carried:
        by_chunk[r["chunk_id"]] = by_chunk.get(r["chunk_id"], 0) + 1
    for cid, n in by_chunk.items():
        print(f"  TASINDI  {cid[-8:]}: {n} atama (onceki kosudan)")
    for s in skipped:
        print(f"  BIRAKILDI {s['chunk_id'][-8:]}: {s['_why']}")

    if args.apply and carried:
        # DROP the `error` rows of the carried chunks (now resolved).
        resolved = set(by_chunk)
        new["rows"] = [r for r in new["rows"] if not (r.get("error") and r["chunk_id"] in resolved)]
        new["rows"].extend(carried)
        new["meta"]["n_rows"] = len(new["rows"])
        new["meta"]["n_llm_errors"] = sum(1 for r in new["rows"] if r.get("error"))
        new["meta"]["n_carried_over"] = len(carried)
        RESULTS_PATH.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    n_issues = 0
    if args.apply and skipped:
        # O-07: chunks still unsolved do not drop silently -- written to
        # issues.db as UNRESOLVED_CHUNK (each visible individually in the panel, M-09).
        con = connect_issues(args.issues_db)
        try:
            n_issues = record_unresolved_chunks(con, skipped)
        finally:
            con.close()

    mode = "UYGULANDI" if args.apply else "DRY RUN"
    print(f"[{mode}] {len(carried)} satir / {len(by_chunk)} chunk tasindi, "
          f"{len(skipped)} chunk cozulmemis kaldi"
          + (f", {n_issues} issue yazildi -> {args.issues_db}" if args.apply else ""))


if __name__ == "__main__":
    main()
