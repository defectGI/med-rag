"""Shared frozen-artifact paths, anchored to the repo root.

`facts/db/specs.db` (9 files referenced it) and
`chunker/storage/all_chunks.json` (6 files) were each rebuilt the same way in
every consumer -- `Path(__file__).resolve().parents[N] / "facts" / "db" /
"specs.db"` with `N` picked to land on the repo root from wherever that file
happened to sit. Consolidated here so the repo-root computation exists once.

**Scope note (same "at least two already-installed consumers" entry discipline
the other core/ consolidations used):** only the chatbot both is installed as
an `medrag` dependent AND actually constructed these paths itself (in
`factory.py::_BUNDLED_DB_PATH` and `chunk_store.py::_BUNDLED_CORPUS_PATH`) --
that's one component, not two, but it's two independent construction sites
within it, which is why moving them here still de-duplicates real code (not
speculative); both now import from here. Retrieval's db_query module never
built a `specs.db` path itself (the path is threaded in as a parameter from
`factory.py`), and vectorize's discovery code never hardcodes
`all_chunks.json` by name -- so neither added a second real consumer.

NOT wired in yet: `src/medrag/api/panel/ingest.py::SPECS_DB_PATH`/
`ALL_CHUNKS_PATH`/`DOCUMENT_NODES_PATH` still build their own copies even
though the panel moved under `src/medrag/` (that move only wired its own
`lineage.db`/`LINEAGE_DB` connections, not these -- debt noted there, not
re-closed here), and `src/medrag/pipeline/cli/run_chunk_pipeline.py`'s
`all_chunks_file` construction -- neither `pipeline` nor `chatbot-corpus` is
an installed `medrag` dependent. These should point here once their component is
moved under `src/medrag/`.

`FACTS_DB_DIR` was added alongside `SPECS_DB_PATH` because
`src/medrag/api/factory.py::_FACTS_DB_DIR` (used for `schema.yaml` and
`schema_model.yaml`, not just `specs.db`) computed the same repo-root-relative
`facts/db/` directory independently via its own `Path(__file__).resolve()`.
Now it is derived from here instead, so there is exactly one repo-root
computation for the `facts/db/` directory.

`chunker` itself moved to `src/medrag/pipeline/chunker/` and is now an installed
`medrag` dependent, but its own `layout.py::ALL_CHUNKS_FILE` is just a filename
constant under a caller-supplied root (`CHUNKER_OUTPUT_DIR`), not a second
hardcoded copy of this frozen path -- nothing to wire on chunker's own side.
`ALL_CHUNKS_PATH` below is unchanged by the move: `chunker/storage/` (the
frozen `all_chunks.json` itself) stayed at the repo-root `chunker/` directory,
the same "data doesn't move with code" precedent already set for
`vectorize/storage/`.

`document_nodes.json`'s path is deliberately NOT added here: it is a frozen
carrier-identity artifact (random `doc_id`, editorial `is_active`,
expensive-to-recompute parse/chunk state) and none of its three path-building
consumers are `medrag` dependents yet either -- same "not wired in yet" bucket,
noted separately so a future reader doesn't conflate "not centralized" with
"safe to regenerate".
"""

from __future__ import annotations

import os
from pathlib import Path

# src/medrag/core/paths.py -> core/ -> urun/ -> src/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

FACTS_DB_DIR = _REPO_ROOT / "facts" / "db"
SPECS_DB_PATH = FACTS_DB_DIR / "specs.db"
ALL_CHUNKS_PATH = _REPO_ROOT / "chunker" / "storage" / "all_chunks.json"

#: A fix: the WRITING side of specs.db (discover.py/load_to_db.py/build_all.py/
#: retry_failed.py/build_facts_db.py) carried NO env override at all -- each one
#: computed the SAME repo-root-relative copy of SPECS_DB_PATH independently
#: (inside the container `/app/facts/db/specs.db`, attached to no volume).
#: Result: the specs.db the `pipeline` container's nightly run wrote to was a
#: COMPLETELY DIFFERENT in-container copy of the file the serving side (the
#: `api`/`worker`/`web`, already env-overridable through
#: `CHATBOT_DB_QUERY_DB_PATH`) read -- the nightly run never REACHED the
#: chatbot (found by inspecting the server). SAME pattern as
#: `resolve_reports_dir()`/`resolve_issues_db_path()`: fixed default
#: (UNCHANGED) + override via a single environment variable.
SPECS_DB_PATH_ENV = "SPECS_DB_PATH"


def resolve_specs_db_path() -> Path:
    """Returns the `SPECS_DB_PATH` environment variable if it is defined,
    otherwise the `SPECS_DB_PATH` constant (the fixed default,
    `facts/db/specs.db`). EVERY writer/reader side of specs.db (`discover.py`,
    `load_to_db.py`, `build_all.py`, `retry_failed.py`, `build_facts_db.py`)
    must call THIS -- if two separate resolutions are written, one sees the env
    var and the other doesn't, so the nightly run writes to one place while
    someone else reads another (see the note above)."""
    raw = os.environ.get(SPECS_DB_PATH_ENV)
    return Path(raw) if raw else SPECS_DB_PATH

#: The directory the nightly run reports (`nightly_YYYY-MM-DD.json`, kept
#: without any retention limit) are written to.
#: `medrag.pipeline.nightly_report.save_nightly_report`/
#: `load_nightly_report_dict` take this directory as a `reports_dir` parameter
#: chosen by the CALLER (that module keeps no constant tied to the repo root);
#: the panel (`medrag.api.panel`) cannot IMPORT `medrag.pipeline` (layer rule), so
#: it reads the same directory from HERE -- the shared `core` layer. Both sides
#: align on this SINGLE constant, the path is not defined twice.
#:
#: This constant means `/app/reports` inside the container -- it EVAPORATES on
#: every redeploy (logs/backups/corpus are already attached to a persistent
#: root on the host, and the reports should live there too).
#: `NIGHTLY_REPORTS_DIR_ENV` overrides it -- SAME pattern as
#: `nightly_backup.py::NIGHTLY_BACKUP_ROOT_ENV`/`nightly_lock.py::
#: NIGHTLY_LOCK_PATH` (fixed default + environment-variable override, NOT added
#: to TOML -- for the same reason as those two: deployment-environment/path
#: information, not a value). `REPORTS_DIR` (below) is ONLY the default VALUE --
#: the real resolution must ALWAYS go through `resolve_reports_dir()` (the
#: caller reads the env at CALL time, not at import time -- that is what makes
#: `monkeypatch.setenv` work in tests).
NIGHTLY_REPORTS_DIR_ENV = "NIGHTLY_REPORTS_DIR"
REPORTS_DIR = _REPO_ROOT / "reports"


def resolve_reports_dir() -> Path:
    """Returns `NIGHTLY_REPORTS_DIR` if defined, otherwise the `REPORTS_DIR`
    default. `run_nightly.py` (the side WRITING the report) and
    `panel/blueprint.py` (the side READING it) call THIS SAME function -- two
    separate resolutions are not written, because then the environment variable
    would only affect one of them and the report would be written but invisible
    to the panel."""
    raw = os.environ.get(NIGHTLY_REPORTS_DIR_ENV)
    return Path(raw) if raw else REPORTS_DIR


#: `catalog_chunk_ownership/build_all.py` (the WRITING side, multi-owner
#: document -> chunk ownership table) and `facts/discover.py` (the READING
#: side, the multi-owner fan-out narrowing in `build_product_manifest`) MUST
#: see the SAME directory -- if they diverge, build_all writes and discover
#: finds NOTHING, so multi-owner documents (catalogs/brochures) silently drop
#: into `docs_excluded_fanout` (a safe drop, but hard to notice). SAME pattern
#: as `resolve_reports_dir()`: fixed default (today's behavior, under `src/`
#: inside the container -- NOT persistent) + a single env override (for the
#: deployment environment).
CHUNK_OWNERSHIP_DIR_ENV = "CHUNK_OWNERSHIP_DIR"
CHUNK_OWNERSHIP_DIR = (
    Path(__file__).resolve().parents[1] / "pipeline" / "facts" / "catalog_chunk_ownership" / "results"
)


def resolve_chunk_ownership_dir() -> Path:
    """Returns `CHUNK_OWNERSHIP_DIR` if the environment variable is defined,
    otherwise the `CHUNK_OWNERSHIP_DIR` constant (the fixed default).
    `build_all.py::RESULTS_DIR` and `discover.py::
    load_chunk_ownership_index`/`load_chunk_anchor_index` call THIS SAME
    function at CALL time (not a value FROZEN at import time) -- that is what
    makes `monkeypatch.setenv` work in tests; see `resolve_reports_dir()`'s own
    docstring for the same reasoning."""
    raw = os.environ.get(CHUNK_OWNERSHIP_DIR_ENV)
    return Path(raw) if raw else CHUNK_OWNERSHIP_DIR


#: `issues.db` -- a SIBLING file of `specs.db` (`facts/db/issues.db`). Same
#: class as `FACTS_RESULTS_DIR`/`CHUNK_OWNERSHIP_DIR`/`NIGHTLY_REPORTS_DIR`/
#: `STORAGE_IMAGES_DIR`/`STORAGE_LABELS_DIR`/`VECTORIZE_OUTPUT_DIR`: the
#: default path means `/app/facts/db/issues.db` inside the container, it lives
#: inside the image and EVAPORATES ON EVERY REDEPLOY (nightly-run log: "4
#: UNRESOLVED_DOC issue yazildi -> /app/facts/db/issues.db"). SAME pattern as
#: `resolve_reports_dir()`/`resolve_chunk_ownership_dir()`: fixed default
#: (today's behavior, UNCHANGED) + a single env override (for the deployment
#: environment). The writing side (`medrag.pipeline.facts.
#: issues_bridge.DEFAULT_ISSUES_DB_PATH`, `load_to_db.py`/`build_all.py`/
#: `carry_over_unresolved.py`) AND the reading side
#: (`medrag.api.panel.blueprint`) call THIS SAME function -- two separate
#: resolutions are not written (otherwise the same risk `resolve_reports_dir()`
#: already argues against: one side sees it, the other doesn't).
ISSUES_DB_PATH_ENV = "ISSUES_DB_PATH"
ISSUES_DB_PATH = FACTS_DB_DIR / "issues.db"


def resolve_issues_db_path() -> Path:
    """Returns `ISSUES_DB_PATH` if the environment variable is defined,
    otherwise the `ISSUES_DB_PATH` constant (the fixed default,
    `facts/db/issues.db`). SAME pattern as
    `resolve_reports_dir()`/`resolve_chunk_ownership_dir()`: the caller reads it
    at CALL time (not at import time) -- that is what makes
    `monkeypatch.setenv` work in tests."""
    raw = os.environ.get(ISSUES_DB_PATH_ENV)
    return Path(raw) if raw else ISSUES_DB_PATH


#: The PERMANENT queue of MODIFIED doc_ids that were "forgotten" (whose
#: derivatives `run_nightly.py::_forget_deleted_sources` DELETED) but whose
#: `facts`/`load` run has NOT YET COMPLETED SUCCESSFULLY.
#: `classify_documents.py::scan_status_for` answers MODIFIED/UNCHANGED against
#: the hash stored by the PREVIOUS scan, and the scan saves the NEW hash
#: IMMEDIATELY -- so the assumption "the same document still looks MODIFIED the
#: next night and fixes itself" is WRONG: if the run BREAKS between `forget`
#: (start of parse) and `facts` (the end of the chain), the document looks
#: UNCHANGED THE NEXT NIGHT and is NEVER PICKED AGAIN -- silent, PERMANENT data
#: loss. This queue keeps that interrupted stretch somewhere that SURVIVES the
#: run. SAME discipline as `chunk_owner_progress.jsonl`: append-only JSONL,
#: every line `{"doc_id":..., "status": "pending"|"done"|"orphaned"}`
#: (`orphaned`: an ownerless doc -- `resolve_codes` returns empty, no product
#: to process; found by a nightly-run review), LAST record wins (the reading
#: side `run_nightly.py::_pending_rework_doc_ids` reduces them). SAME pattern as
#: `FACTS_RESULTS_DIR`/`ISSUES_DB_PATH`: fixed default
#: (`facts/pending_rework.jsonl`, lives inside the image in the container,
#: evaporates on every redeploy) + a single env override.
PENDING_REWORK_PATH_ENV = "FACTS_PENDING_REWORK_PATH"
PENDING_REWORK_PATH = _REPO_ROOT / "facts" / "pending_rework.jsonl"


def resolve_pending_rework_path() -> Path:
    """Returns `FACTS_PENDING_REWORK_PATH` if defined, otherwise
    `PENDING_REWORK_PATH` (the fixed default). SAME pattern as
    `resolve_reports_dir()`: the caller reads it at CALL time."""
    raw = os.environ.get(PENDING_REWORK_PATH_ENV)
    return Path(raw) if raw else PENDING_REWORK_PATH
