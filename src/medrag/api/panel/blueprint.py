"""Deploys the lineage panel as a Flask **blueprint** -- same image, same
deploy, no extra service.

**Layering note:** this module uses `medrag.core` (`core.db.issues`, `core.paths`)
and the panel's own `medrag.api.panel.*` modules; it does NOT import
`medrag.pipeline` -- the panel has no need to call the pipeline (it only reads and
writes `resolved_*`). `src/medrag/tests/test_import_contracts.py` already enforces
this contract (`medrag.api` <-> `medrag.pipeline` independence) for the whole
package.

The query functions (`api_stages`/`api_conflicts`/`api_graph`/`api_cascade`) are
not copied from `serve.py`, they are IMPORTED from it -- the two servers
(stdlib `http.server`, local development; Flask blueprint, deploy) SHARE the
same query logic rather than duplicating it.

**Landing page:** `GET /` returns the panel's static `index.html` (a page hidden
in a submenu informs no one -- so the dashboard is the FIRST section of
`index.html` and needs no extra click). `GET /api/dashboard` produces the data
for that page -- it calls `medrag.api.panel.dashboard.build_dashboard`; the panel
does NO computation itself, it only reads the latest nightly report and adds the
open-issue count via `count_open`.

**Issue list / filtering:** `GET /api/issues` narrows by the
`stage`/`severity`/`source_doc`/`reason` query parameters
(`core.db.issues.fetch_filtered`) -- the accumulated flag queue was unusable as
a flat list.

**Access / authorization:**
- Read (`GET /api/*`): everyone, no auth required.
- Write (`POST /api/issues/<id>/resolve`): updates only the `resolved_*` fields
  (`core.db.issues.resolve_issue` -- the single write surface), guarded by
  `auth.check_write_token`. The panel has no other write/trigger surface.
- Isolating the panel from the NETWORK (closed to outside, open to the internal
  network) is NOT this blueprint's responsibility -- see the `app.py` module
  docstring.
"""

from __future__ import annotations

import os
import sqlite3

from flask import Blueprint, jsonify, request, send_from_directory

from medrag.api.panel import auth
from medrag.api.panel import dashboard as dashboard_mod
from medrag.api.panel.serve import (
    DB_PATH,
    STAGE_ORDER,
    STATIC_DIR,
    _connect,
    api_cascade,
    api_conflicts,
    api_graph,
    api_stages,
)
from medrag.core.db.issues import (
    connect_issues,
    count_open,
    fetch_filtered,
    resolve_issue,
)
from medrag.core.paths import resolve_issues_db_path, resolve_reports_dir

#: `issues.db` -- a SIBLING file of `facts/db/specs.db`. The panel calls
#: `resolve_issues_db_path()` (the SAME resolver as the writers --
#: `medrag.pipeline.facts.issues_bridge`/`load_to_db.py`/`build_all.py`/
#: `carry_over_unresolved.py`) on EVERY request, not a constant frozen at module
#: import: this lets the `ISSUES_DB_PATH` environment variable be pointed at a
#: persistent path in the container (`/corpus/facts/issues.db`), whereas the old
#: frozen default lived INSIDE the image and was LOST on every redeploy. It does
#: NOT import `medrag.pipeline` (layering rule) -- it only uses the shared
#: `core.paths` layer.

panel_bp = Blueprint(
    "panel",
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="",
)


def _lineage_connect() -> sqlite3.Connection:
    return _connect()


def _lineage_db_missing_response():
    if not os.path.exists(DB_PATH):
        return jsonify({"error": "lineage.db yok -- once `python ingest.py` calistirin"}), 503
    return None


@panel_bp.get("/")
def _index():
    """The panel's landing page -- `index.html`, where the dashboard lives.
    Static file serving via `static_url_path=""` covers `/index.html` but NOT
    the root `/` -- this route closes that gap, so anyone entering the panel
    sees the dashboard without an extra click."""
    return send_from_directory(STATIC_DIR, "index.html")


@panel_bp.get("/api/dashboard")
def _dashboard():
    """Data for the landing page -- last night + four counters. The panel does
    NO computation; `dashboard.build_dashboard` only reads the latest nightly
    report, and the open-issue count is added here via `count_open` (it is not
    part of the report, it is `issues.db`'s own counter).

    The directory is resolved on EVERY request via `resolve_reports_dir()` (the
    SINGLE resolver in `core.paths`, the same one `run_nightly.py` uses when
    WRITING the report) -- not a constant frozen at import time; otherwise a
    change to `NIGHTLY_REPORTS_DIR` after the panel process starts (or a test
    doing `monkeypatch.setenv`) would never be seen here."""
    open_issues_count = 0
    issues_db_path = resolve_issues_db_path()
    if issues_db_path.exists():
        con = connect_issues(issues_db_path)
        try:
            open_issues_count = count_open(con)
        finally:
            con.close()
    return jsonify(
        dashboard_mod.build_dashboard(reports_dir=resolve_reports_dir(),
                                       open_issues_count=open_issues_count)
    )


@panel_bp.get("/api/stages")
def _stages():
    missing = _lineage_db_missing_response()
    if missing is not None:
        return missing
    conn = _lineage_connect()
    try:
        return jsonify(api_stages(conn))
    finally:
        conn.close()


@panel_bp.get("/api/conflicts")
def _conflicts():
    missing = _lineage_db_missing_response()
    if missing is not None:
        return missing
    conn = _lineage_connect()
    try:
        return jsonify(api_conflicts(conn, request.args.to_dict(flat=False)))
    finally:
        conn.close()


@panel_bp.get("/api/graph")
def _graph():
    missing = _lineage_db_missing_response()
    if missing is not None:
        return missing
    conn = _lineage_connect()
    try:
        return jsonify(api_graph(conn, request.args.to_dict(flat=False)))
    finally:
        conn.close()


@panel_bp.get("/api/cascade")
def _cascade():
    missing = _lineage_db_missing_response()
    if missing is not None:
        return missing
    conn = _lineage_connect()
    try:
        return jsonify(api_cascade(conn, request.args.to_dict(flat=False)))
    finally:
        conn.close()


@panel_bp.get("/api/issues")
def _issues_list():
    """Read -- everyone. Narrows by the `stage`/`severity`/`source_doc`/`reason`
    query parameters -- the accumulated flag queue was unusable as a flat list
    (nobody looked at a multi-megabyte dump). With no filter given (same as the
    old behavior) all rows are returned, so every unresolved chunk can be seen
    individually."""
    issues_db_path = resolve_issues_db_path()
    if not issues_db_path.exists():
        return jsonify({"error": "issues.db yok", "issues": []}), 200
    con = connect_issues(issues_db_path)
    try:
        issues = fetch_filtered(
            con,
            stage=request.args.get("stage") or None,
            severity=request.args.get("severity") or None,
            source_doc=request.args.get("source_doc") or None,
            reason=request.args.get("reason") or None,
        )
        return jsonify({"issues": issues})
    finally:
        con.close()


@panel_bp.post("/api/issues/<int:issue_id>/resolve")
def _issues_resolve(issue_id: int):
    """Write -- authorized. The panel's single write surface: it updates only
    the `resolved_at`/`resolved_by` fields (`core.db.issues.resolve_issue`),
    touches no other column, and triggers no pipeline stage."""
    provided = request.headers.get(auth.WRITE_TOKEN_HEADER)
    if not auth.check_write_token(provided):
        return jsonify({"error": "yetkisiz -- gecerli bir yazma token'i gerekli"}), 403

    body = request.get_json(silent=True) or {}
    resolved_by = str(body.get("resolved_by") or "").strip()
    if not resolved_by:
        return jsonify({"error": "resolved_by gerekli"}), 400

    issues_db_path = resolve_issues_db_path()
    if not issues_db_path.exists():
        return jsonify({"error": "issues.db yok"}), 404

    con = connect_issues(issues_db_path)
    try:
        resolve_issue(con, issue_id, resolved_by=resolved_by)
    finally:
        con.close()
    return jsonify({"status": "resolved", "id": issue_id, "resolved_by": resolved_by})


__all__ = ["STAGE_ORDER", "panel_bp"]
