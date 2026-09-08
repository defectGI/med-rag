"""N-08/N-09 (I-19, I-21): nightly run's SINGLE report data model.

Rule (I-21, N-08): the run produces one report JSON; the human-readable
page is rendered FROM that JSON. **Two separate producers are NEVER
WRITTEN** -- otherwise numbers diverge. So here:

  1. `NightlyReport` (and sub-section models) -- pydantic-defined SINGLE
     schema. The run (N-01) fills this model.
  2. `NightlyReport.to_json_str()` / `to_json_dict()` -- structured view
     (the panel M-01..M-14's source, per the M group note: "panel does
     NOT compute, it reads the report").
  3. `render_markdown(report)` -- human-readable view. Renders from the
     SAME `report` object as `to_json_*`; no recomputation between them,
     render only formats.

N-09 (I-19): all nine sections below are SEPARATE in the model as
NON-OPTIONAL fields (every section must be present):
  Source Diff · Skipped · Parse · Chunk · Vectorize · Ownership ·
  Product Features · Failures · Integrity.

The Integrity section (`IntegritySection`) is filled by N-05 (backup-first
step) and O-08 (DB integrity gate). N-05 may not have been finished when
this module was written -- so `backup_location`/`backup_completed` fields
are kept as `str | None` / `bool` as an INTERFACE (the caller fills them
once N-05 is wired in; schema does NOT change).

Layer rule: this module may depend on `medrag.core` (if any) but does NOT
IMPORT `medrag.api` (pipeline -> core, no path pipeline -> api).

N-10 (I-20): the run's three-valued outcome (`full_success`/`partial`/
`failed`) is DERIVED via `NightlyReport.outcome()` (NOT a separately
settable field on the report -- same reasoning as N-08's "no two separate
producers" rule). Rule: a single failure makes "full success" impossible;
if the nightly window hits its ceiling (`NIGHTLY_WINDOW_MAX_HOURS`,
K-62) the outcome is ALWAYS `partial` (controlled stop, not "cut mid-way").

N-11 (I-22): `save_nightly_report`/`load_nightly_report_dict` -- each
night's report is stored UNBOUNDED as `reports/nightly_YYYY-MM-DD.json`
(K-62: <G> = unbounded, no auto-delete). The reader side returns the raw
dict (consistent with N-08's "panel reads JSON directly" rule) -- since it
carries derived fields like `outcome` that aren't STORED in the model, it
is NOT rebuilt as `NightlyReport` (`extra="forbid"` would reject it).

N-12 (I-24/I-25, K-64): no separate `lineage` table is CREATED. Status is
derived from three sources: scanner's `scan_status`, existence of the
derivative path, and comparison of the record's `*_version` with the
current config version (N-13). `derive_derivative_status()` is the PURE
function that performs this derivation; `NightlyReport.derivative_status`
(doc_id -> stage -> status) is the result written to the report.

N-13 (the "stage changed" leg of I-24): current versions live in ONE
place -- `core/config/default.toml` [pipeline.versions] -- and are read
via `current_pipeline_versions()`.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# K-62: nightly window starts at 23:00, max 8 hours (07:00 cutoff). N-10
# always considers a run that hits this ceiling as `partial` (controlled
# stop, not "cut mid-way").
NIGHTLY_WINDOW_MAX_HOURS = 8.0

Outcome = Literal["full_success", "partial", "failed"]


class FailureEntry(BaseModel):
    """One row in the Failures section (N-09)."""

    model_config = ConfigDict(extra="forbid")

    file: str
    stage: Literal["products", "scan", "parse", "chunk", "ownership", "facts", "load", "vectorize",
                   "db_write", "backup", "integrity"]
    reason: str
    error_text: str


class SourceDiffSection(BaseModel):
    """Source diff: from the scanner's `scan_status` (N-09 row 1)."""

    model_config = ConfigDict(extra="forbid")

    changed: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    moved: list[str] = Field(default_factory=list)


class SkippedSection(BaseModel):
    """Skipped: count + paths of unchanged documents."""

    model_config = ConfigDict(extra="forbid")

    unchanged_count: int = Field(ge=0)
    unchanged_paths: list[str] = Field(default_factory=list)


class ParseSection(BaseModel):
    """Parse: documents processed, duration, model call count.

    N-21: `model_call_count` is Optional -- the current `run_parse_pipeline.py`
    does NOT count LLM/VLM calls anywhere. Writing `0` would say "no calls
    made" which is misleading -- `None` = "not measured", which
    `render_markdown` prints explicitly."""

    model_config = ConfigDict(extra="forbid")

    processed_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    model_call_count: int | None = Field(default=None, ge=0)


class ChunkSection(BaseModel):
    """Chunk: produced count, per-doc breakdown, removed-old count."""

    model_config = ConfigDict(extra="forbid")

    produced_count: int = Field(ge=0)
    per_document_counts: dict[str, int] = Field(default_factory=dict)
    removed_count: int = Field(ge=0)


class VectorizeSection(BaseModel):
    """Vectorize: points written/deleted, total after, model used.

    N-21: `points_deleted`/`total_points_after` are Optional --
    `replace_scope` (delete scope, then re-upsert) never returns the
    number of points actually deleted from Qdrant (`client.delete()`
    doesn't carry that), and "total after" requires a real Qdrant
    `count()` network call -- no fake store supports this. The fake
    stores used in tests don't either. `points_written`/`embedding_model`
    come from LOCAL counters, real values."""

    model_config = ConfigDict(extra="forbid")

    points_written: int = Field(ge=0)
    points_deleted: int | None = Field(default=None, ge=0)
    total_points_after: int | None = Field(default=None, ge=0)
    embedding_model: str


class OwnershipSection(BaseModel):
    """Ownership: deterministic / model-routed / unresolved chunk counts."""

    model_config = ConfigDict(extra="forbid")

    deterministic_count: int | None = Field(default=None, ge=0)
    model_routed_count: int | None = Field(default=None, ge=0)
    unresolved_count: int | None = Field(default=None, ge=0)


class ProductFeaturesSection(BaseModel):
    """Product features: added/removed feature and evidence counts."""

    model_config = ConfigDict(extra="forbid")

    features_added: int | None = Field(default=None, ge=0)
    evidence_added: int | None = Field(default=None, ge=0)
    evidence_removed: int | None = Field(default=None, ge=0)
    features_removed_for_no_evidence: int | None = Field(default=None, ge=0)


class IntegritySection(BaseModel):
    """Integrity: O-08 verification result + N-05 backup location.

    N-05 may not have been finished when this file was written -- so
    these fields are kept as an INTERFACE (the caller fills them with
    real values once N-05 is wired in; the schema does NOT change).
    `backup_completed=False` + `backup_location=None` is used while
    N-05 isn't integrated yet -- a non-empty section that just means
    "not done yet", NOT violating N-09's "every section must be present".
    """

    model_config = ConfigDict(extra="forbid")

    pre_write_check_passed: bool
    post_write_check_passed: bool
    backup_completed: bool
    backup_location: str | None = None
    notes: str = ""


class NightlyReport(BaseModel):
    """N-08/N-09: nightly run's SINGLE report schema (nine sections, all required)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: datetime
    finished_at: datetime

    source_diff: SourceDiffSection
    skipped: SkippedSection
    parse: ParseSection
    chunk: ChunkSection
    vectorize: VectorizeSection
    ownership: OwnershipSection
    product_features: ProductFeaturesSection
    failures: list[FailureEntry] = Field(default_factory=list)
    integrity: IntegritySection

    # N-12: no separate lineage table -- derived status lives here.
    # doc_id -> stage name -> derive_derivative_status() output. Empty dict
    # (default) is backward-compatible with runs before N-12 is integrated.
    derivative_status: dict[str, dict[str, str]] = Field(default_factory=dict)

    def outcome(self, *, window_max_hours: float = NIGHTLY_WINDOW_MAX_HOURS) -> Outcome:
        """N-10: three-valued outcome, derived from THIS `report` object.

        Rule: if the nightly window hits its ceiling the outcome is ALWAYS
        `partial` (controlled stop, not "cut mid-way"). Otherwise a single
        failure makes `full_success` impossible; if none of the progress
        indicators (parse/chunk/vectorize/product_features) increased
        (nothing was produced) the outcome is `failed`, else `partial`.
        """
        duration_hours = (self.finished_at - self.started_at).total_seconds() / 3600.0
        if duration_hours >= window_max_hours:
            return "partial"
        if not self.failures:
            return "full_success"
        # `or 0`: N-21 made some fields Optional ("not measured") but the
        # check here stays numeric -- `features_added` is None every night,
        # so this branch would TypeError exactly when failures is non-empty,
        # i.e. when the report is needed most. "Not measured" is not progress
        # proof; treat as 0.
        made_progress = (
            (self.parse.processed_count or 0) > 0
            or (self.chunk.produced_count or 0) > 0
            or (self.vectorize.points_written or 0) > 0
            or (self.product_features.features_added or 0) > 0
        )
        return "partial" if made_progress else "failed"

    def to_json_dict(self) -> dict:
        """Structured view -- the panel's source.

        `outcome` is NOT a STORED field on the model (N-10) -- it is
        computed here and added, so there is one source of "outcome"."""
        data = json.loads(self.model_dump_json())
        data["outcome"] = self.outcome()
        return data

    def to_json_str(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False, sort_keys=False)


def _fmt_dt(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _fmt_measured(value: int | None) -> str:
    """N-21: `None` = the current code path doesn't measure this count
    (see the relevant section docstring) -- writing `0` would say
    "measured and zero", which is a MISLEADING impression. Output
    "not measured" explicitly."""
    return "not measured" if value is None else str(value)


def render_markdown(report: NightlyReport) -> str:
    """Human-readable view -- rendered FROM `report` (same object, no
    second computation).

    N-08 acceptance criterion: the numbers in this function's output MUST
    match those of `to_json_dict()` -- both read from a single source
    (this `report` object).
    """
    lines: list[str] = []
    lines.append(f"# Nightly Run Report -- {report.run_id}")
    lines.append("")
    lines.append(f"- Start: {_fmt_dt(report.started_at)}")
    lines.append(f"- End: {_fmt_dt(report.finished_at)}")
    lines.append(f"- Outcome: {report.outcome()}")
    lines.append("")

    sd = report.source_diff
    lines.append("## Source Diff")
    lines.append(f"- Changed: {len(sd.changed)}")
    for p in sd.changed:
        lines.append(f"  - {p}")
    lines.append(f"- Added: {len(sd.added)}")
    for p in sd.added:
        lines.append(f"  - {p}")
    lines.append(f"- Removed: {len(sd.removed)}")
    for p in sd.removed:
        lines.append(f"  - {p}")
    lines.append(f"- Moved: {len(sd.moved)}")
    for p in sd.moved:
        lines.append(f"  - {p}")
    lines.append("")

    sk = report.skipped
    lines.append("## Skipped")
    lines.append(f"- Skipped (unchanged): {sk.unchanged_count}")
    for p in sk.unchanged_paths:
        lines.append(f"  - {p}")
    lines.append("")

    pa = report.parse
    lines.append("## Parse")
    lines.append(f"- Documents processed: {pa.processed_count}")
    lines.append(f"- Duration (s): {pa.duration_seconds}")
    lines.append(f"- Model call count: {_fmt_measured(pa.model_call_count)}")
    lines.append("")

    ch = report.chunk
    lines.append("## Chunk")
    lines.append(f"- Chunks produced: {ch.produced_count}")
    lines.append(f"- Old chunks removed: {ch.removed_count}")
    if ch.per_document_counts:
        lines.append("- Per-document breakdown:")
        for doc_id, count in ch.per_document_counts.items():
            lines.append(f"  - {doc_id}: {count}")
    lines.append("")

    ve = report.vectorize
    lines.append("## Vectorize")
    lines.append(f"- Points written: {ve.points_written}")
    lines.append(f"- Points deleted: {_fmt_measured(ve.points_deleted)}")
    lines.append(f"- Total after: {_fmt_measured(ve.total_points_after)}")
    lines.append(f"- Model used: {ve.embedding_model}")
    lines.append("")

    ow = report.ownership
    lines.append("## Ownership")
    lines.append(f"- Resolved deterministically: {_fmt_measured(ow.deterministic_count)}")
    lines.append(f"- Routed to model: {_fmt_measured(ow.model_routed_count)}")
    lines.append(f"- Unresolved: {_fmt_measured(ow.unresolved_count)}")
    lines.append("")

    pf = report.product_features
    lines.append("## Product Features")
    lines.append(f"- Features added: {_fmt_measured(pf.features_added)}")
    lines.append(f"- Evidence added: {_fmt_measured(pf.evidence_added)}")
    lines.append(f"- Evidence removed: {_fmt_measured(pf.evidence_removed)}")
    lines.append(f"- Features removed for no evidence: {_fmt_measured(pf.features_removed_for_no_evidence)}")
    lines.append("")

    lines.append("## Failures")
    if not report.failures:
        lines.append("- (none)")
    else:
        for f in report.failures:
            lines.append(f"- [{f.stage}] {f.file}: {f.reason} -- {f.error_text}")
    lines.append("")

    it = report.integrity
    lines.append("## Integrity")
    lines.append(f"- Pre-write check: {'passed' if it.pre_write_check_passed else 'FAILED'}")
    lines.append(f"- Post-write check: {'passed' if it.post_write_check_passed else 'FAILED'}")
    lines.append(f"- Backup taken: {'yes' if it.backup_completed else 'no'}")
    lines.append(f"- Backup location: {it.backup_location or '(none)'}")
    if it.notes:
        lines.append(f"- Notes: {it.notes}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# N-11: report history -- dated file, unbounded, no auto-delete.
# ---------------------------------------------------------------------------


def report_history_filename(started_at: datetime) -> str:
    """The file-name part of the `reports/nightly_YYYY-MM-DD.json` pattern.

    The date is taken from `started_at` (run start, converted to UTC) --
    `finished_at` may cross midnight into the next day, but "what was
    yesterday's count?" always means the night the run STARTED."""
    return f"nightly_{started_at.astimezone(UTC).date().isoformat()}.json"


def save_nightly_report(report: NightlyReport, *, reports_dir: Path) -> Path:
    """Writes the report to `reports_dir/nightly_YYYY-MM-DD.json` (N-11).

    No rotation -- old files are NEVER deleted (K-62). If a second run
    happens on the same day (not expected -- N-14's single scheduled
    task), the file is OVERWRITTEN; storing multiple runs on the same
    day separately is out of scope for this task."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / report_history_filename(report.started_at)
    path.write_text(report.to_json_str(), encoding="utf-8")
    return path


def load_nightly_report_dict(date: str, *, reports_dir: Path) -> dict | None:
    """Reads a previous night's report as a RAW dict ("what was yesterday's
    count?").

    NOT rebuilt as `NightlyReport`: the `outcome` field that
    `to_json_dict()` adds is not stored on the model; `extra="forbid"`
    would reject it on reload. The panel reads raw JSON anyway."""
    path = reports_dir / f"nightly_{date}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# N-12: status derivation.
# ---------------------------------------------------------------------------

ScanStatus = Literal["UNCHANGED", "MODIFIED", "NEW", "DELETED"]
DerivativeStatus = Literal["source_missing", "source_changed", "not_generated", "stage_changed", "fresh"]


def _semver_part(version: str) -> str | None:
    """Comparable semver part -- `"1.0.0+chunk_v1:a1b2c3"` -> `"1.0.0"`. A
    string with NO numeric part (`"chunk_v1:a1b2c3"`) is NOT semver,
    returns None.

    The old `_version_tuple` rule ("non-numeric parts count as 0")
    ALONE caused a silent bug: a non-semver version would map to `(0,)`,
    which sorts BEHIND every real version (see the K-101 note on
    `load_to_db::extractor_version` -- in production a value like
    `"chunk_v1:<sha12>"` was treated as "older than everything", marking
    every row as stale every night). That rule is kept INSIDE
    `_version_tuple` (a half-broken version like `"1.x.0"` should not
    stop a nightly run), but "no part is numeric" is now a separate
    case handled explicitly."""
    head = version.split("+", 1)[0].strip()
    if not head:
        return None
    if not any(piece.strip().isdigit() for piece in head.split(".")):
        return None
    return head


def _version_tuple(version: str) -> tuple[int, ...]:
    """`"1.5.0"` -> `(1, 5, 0)`; non-numeric parts count as 0 (so it never
    crashes -- a version comparison must not halt a nightly run). `+`
    suffix (build-metadata) is DROPPED (`"1.0.0+chunk_v1:a1b2c3"` ->
    `(1, 0, 0)` -- semver rule: build-metadata does NOT participate in
    ordering)."""
    parts: list[int] = []
    for piece in version.split("+", 1)[0].split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _recorded_version_is_behind(recorded: str | None, current: str) -> bool:
    """`recorded is None` -> True (derivative was NEVER generated/written).

    K-101: if `recorded` cannot be read AS semver (e.g. pre-K-101 rows like
    `"chunk_v1:<sha12>"`) return False -- "behind" is wrong, treat as
    "INCOMPARABLE". Treating old-format rows as stale would re-trigger the
    same "every row every night" behavior this fix is closing. To refresh
    such rows the right path ALREADY EXISTS and is invoked manually:
    `FACTS_PROCESS_ALL=1` (see `run_nightly.py::stage_facts`'s escape
    hatches)."""
    if recorded is None:
        return True
    if _semver_part(recorded) is None:
        return False
    return _version_tuple(recorded) < _version_tuple(current)


def derive_derivative_status(
    *,
    scan_status: ScanStatus,
    derivative_exists: bool,
    recorded_version: str | None,
    current_version: str,
) -> DerivativeStatus:
    """Derives K-64's four states (PURE function, no I/O):

    - `source_missing`  <- `scan_status == DELETED`
    - `source_changed`  <- `scan_status in {MODIFIED, NEW}`
    - `not_generated`   <- derivative path missing
    - `stage_changed`   <- recorded version < current version (N-13)
    - `fresh`           <- none of the above

    Order matters: if the source is gone/changed the version comparison is
    MEANINGLESS (a new source will be regenerated), so the first two
    checks come first.
    """
    if scan_status == "DELETED":
        return "source_missing"
    if scan_status in ("MODIFIED", "NEW"):
        return "source_changed"
    if not derivative_exists:
        return "not_generated"
    if _recorded_version_is_behind(recorded_version, current_version):
        return "stage_changed"
    return "fresh"


def current_pipeline_versions() -> dict[str, str]:
    """N-13: reads current stage versions from ONE place
    (`core/config/default.toml` [pipeline.versions]). Layer rule OK --
    `pipeline` -> `core` is allowed (see module docstring)."""
    from medrag.core.config.schema import load_core_config

    versions = load_core_config().pipeline.versions
    return {
        "parser": versions.parser_version,
        "chunker": versions.chunker_version,
        "facts": versions.facts_version,
        "vectorize": versions.vectorize_version,
    }