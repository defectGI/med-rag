"""`specs.db` staleness AUDIT -- read-only, writes NOTHING. Designed to be
run hard-on right after a full nightly run: at that point all three
criteria are expected to come back ~0 -- if not, `load_to_db.py::_reset_product`
(chunk_full_run_v1's force reset path) missed something.

Scope: only rows with `extractor = 'chunk_full_run_v1'` (`load_to_db.EXTRACTOR`)
are scanned. Bootstrap rows (`extractor IS NULL`, `evidence` always `'[]'`)
are EXCLUDED UP FRONT -- they were NEVER extracted ("empty"); that is a
DIFFERENT state from a row whose source later disappeared ("stale");
mixing them would drown the `empty_evidence` criterion in thousands of
bootstrap rows and render the "~0 expected" acceptance clause meaningless.

Three criteria (a row can match MORE THAN ONE -- each criterion is
COUNTED INDEPENDENTLY, intersections are NOT reported):

  a) dangling_evidence       -- a `chunk_sha256` in `evidence[]` is
     MISSING from the current chunk set (`all_chunks.json`).
  b) empty_evidence          -- extractor set but `evidence[]` EMPTY. The
     trace of a bug class; this read-only audit DETECTS it but does NOT
     FIX it.

     2026-08-27 FIX (K-102): the previous text here said "no normal
     write path produces this" -- that was WRONG, and this audit's own
     output refuted it (386 rows / 94 products). `load_to_db.load_product`
     was rejecting an unresolvable `source_chunk_id` ONLY for
     `status='absent'`; `present` facts were getting through,
     `_merge_evidence` returned an empty list for them, and the row was
     being written as `evidence='[]'` + `extractor` SET. The write side
     was closed by K-102 (now rejected for every status + a second gate
     at the write point), so from here on this criterion should
     legitimately be ~0 on newly-written rows. Rows written BEFORE the
     closure remain in the DB: `forget_source` only deletes rows whose
     evidence FIELD empties, so they do not self-clean -- a separate
     cleanup/migration decision is required.
  c) stale_extractor_version -- `extractor_version` is BEHIND the current
     `[pipeline.versions] facts_version` (the extraction METHOD --
     prompt/schema/threshold -- changed but this row was written by the
     old method).

The chunk set is built SOLELY from `all_chunks.json` (K-96 instruction)
-- discover.py's TEMPORARY atoms_bridge source is NOT COUNTED: evidence
from the bridge therefore shows as dangling under (a), which is
DELIBERATE (the bridge is a limited-scope scaffold until chunker's real
output replaces it; see discover.py docstring "Scope (critical)").

K-97 (2026-08-26 fix, replaces N-20/K-96 step 3): `stale_product_codes`
is NO LONGER `stage_facts`'s product SELECTOR -- it is just an
AUDIT/ops convenience returning the UNION of the three criteria above
(see that function's docstring). Previously there was a fourth
"selector" layer here: "products whose current manifest contains a
chunk_sha256 that appears in NO `evidence[]` for that product". That
rule was REMOVED -- a chunk that PRODUCES NO FACTS (e.g. one that
contains only headings/boilerplate) never appears in any `evidence[]`,
so it would appear "without evidence" FOREVER and practically flag
the ENTIRE corpus (~218 products) as stale every night -- the
incremental run was effectively NONE (see
`run_nightly.py::_resolve_incremental_codes` K-97 rationale).
`run_nightly.py` now derives the product selection DIRECTLY from
NEW+MODIFIED documents (`_incremental_doc_ids` +
`run_full.resolve_codes`); it does NOT CALL this module. K-96's real
purpose was AUDIT (a verification tool), not selection; this module
now serves that purpose alone."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from medrag.pipeline.facts.discover import ALL_CHUNKS_PATH, DB_PATH, _connect_ro
from medrag.pipeline.facts.load_to_db import EXTRACTOR
from medrag.pipeline.nightly_report import _recorded_version_is_behind


@dataclass
class StalenessExample:
    value_id: int
    product_code: str
    key: str
    reason: str

    def to_dict(self) -> dict:
        return {"value_id": self.value_id, "product_code": self.product_code,
                "key": self.key, "reason": self.reason}


@dataclass
class StalenessCriterion:
    """The result of one criterion: row count + product count + first 10
    examples (the output format K-96 asked for)."""

    row_count: int = 0
    products: set[str] = field(default_factory=set)
    examples: list[StalenessExample] = field(default_factory=list)

    @property
    def product_count(self) -> int:
        return len(self.products)

    def register(self, *, value_id: int, product_code: str, key: str, reason: str) -> None:
        self.row_count += 1
        self.products.add(product_code)
        if len(self.examples) < 10:
            self.examples.append(StalenessExample(value_id, product_code, key, reason))

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "product_count": self.product_count,
            "examples": [e.to_dict() for e in self.examples],
        }


@dataclass
class StalenessReport:
    dangling_evidence: StalenessCriterion
    empty_evidence: StalenessCriterion
    stale_extractor_version: StalenessCriterion

    def to_dict(self) -> dict:
        return {
            "dangling_evidence": self.dangling_evidence.to_dict(),
            "empty_evidence": self.empty_evidence.to_dict(),
            "stale_extractor_version": self.stale_extractor_version.to_dict(),
        }


def current_chunk_sha_set(all_chunks_path: Path = ALL_CHUNKS_PATH) -> frozenset[str]:
    """Same FALLBACK as `discover.py::build_product_manifest`: if
    `content_sha256` is present use it, else `node_id`
    (`sha = n.get("content_sha256") or n["node_id"]`) -- `evidence[].
    chunk_sha256` may have been written as either one, both must count as
    "current"."""
    all_chunks_path = Path(all_chunks_path)
    if not all_chunks_path.is_file():
        return frozenset()
    data = json.loads(all_chunks_path.read_text(encoding="utf-8"))
    shas: set[str] = set()
    for chunk_set in data.get("documents", {}).values():
        for n in chunk_set.get("nodes", []):
            shas.add(n.get("content_sha256") or n["node_id"])
    return frozenset(shas)


def compute_staleness(
    con: sqlite3.Connection,
    *,
    current_facts_version: str,
    all_chunks_path: Path = ALL_CHUNKS_PATH,
) -> StalenessReport:
    """Computes all three criteria in one pass (K-96: "no LLM, no network,
    seconds"). WRITES NOTHING -- reads `spec_value` only."""
    current_shas = current_chunk_sha_set(all_chunks_path)

    dangling = StalenessCriterion()
    empty = StalenessCriterion()
    stale_version = StalenessCriterion()

    rows = con.execute(
        "SELECT value_id, product_code, key, evidence, extractor_version "
        "FROM spec_value WHERE extractor = ?",
        (EXTRACTOR,),
    ).fetchall()

    for value_id, product_code, key, evidence_json, extractor_version in rows:
        evidence = json.loads(evidence_json) if evidence_json else []
        if not evidence:
            empty.register(value_id=value_id, product_code=product_code, key=key,
                            reason="evidence[] empty")
        else:
            missing_shas = [e.get("chunk_sha256") for e in evidence
                            if e.get("chunk_sha256") not in current_shas]
            if missing_shas:
                dangling.register(
                    value_id=value_id, product_code=product_code, key=key,
                    reason=f"not in current chunk set: {missing_shas[:3]}",
                )
        if _recorded_version_is_behind(extractor_version, current_facts_version):
            stale_version.register(
                value_id=value_id, product_code=product_code, key=key,
                reason=f"extractor_version={extractor_version!r} < facts_version={current_facts_version!r}",
            )

    return StalenessReport(dangling_evidence=dangling, empty_evidence=empty,
                            stale_extractor_version=stale_version)


def stale_product_codes(
    con: sqlite3.Connection,
    *,
    current_facts_version: str,
    all_chunks_path: Path = ALL_CHUNKS_PATH,
) -> list[str]:
    """Union of the products affected by the THREE criteria
    (dangling_evidence / empty_evidence / stale_extractor_version) of
    `compute_staleness` -- writes NOTHING (read-only like
    `compute_staleness`); just a thin AUDIT/ops convenience wrapper.

    K-97 (2026-08-26 fix): NO LONGER `stage_facts`'s product SELECTOR --
    `run_nightly.py::_resolve_incremental_codes` does NOT call this
    function; it derives the selection directly from NEW+MODIFIED
    documents. Previously a fourth "no evidence in any chunk of the
    current manifest" criterion lived here -- REMOVED (the root cause
    was chunks that produce no fact being marked stale forever, which
    practically flagged the ENTIRE corpus -- ~218 products -- every
    night, making the incremental run effectively none). K-96's real
    purpose was AUDIT (a verification tool), not selection; this
    function now returns to that audit purpose."""
    report = compute_staleness(con, current_facts_version=current_facts_version,
                                all_chunks_path=all_chunks_path)
    stale_codes: set[str] = set()
    for criterion in (report.dangling_evidence, report.empty_evidence,
                      report.stale_extractor_version):
        stale_codes |= criterion.products
    return sorted(stale_codes)


def main() -> None:
    from medrag.pipeline.nightly_report import current_pipeline_versions

    con = _connect_ro(DB_PATH)
    try:
        facts_version = current_pipeline_versions()["facts"]
        report = compute_staleness(con, current_facts_version=facts_version)
    finally:
        con.close()

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()