"""Panel Flask blueprint -- deployability + read/write authorization split.

We do not re-test here that `medrag.pipeline` is never imported --
`src/medrag/tests/test_import_contracts.py` already enforces that for the whole
package; this file locks the blueprint's OWN behavior.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from medrag.api.panel import auth, blueprint
from medrag.api.panel.app import create_panel_app
from medrag.core.db.issues import record_issue


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Neither DB touches this component's external artifacts such as the real
    # `all_chunks.json`/`specs.db` -- the module-level constants in serve.py/
    # blueprint.py are redirected to tmp_path.
    monkeypatch.setattr(blueprint, "DB_PATH", str(tmp_path / "lineage.db"))
    # `blueprint.py` now calls `resolve_reports_dir()`/`resolve_issues_db_path()`
    # on EVERY request (not an import-time constant) -- tests route it the SAME
    # way as the real mechanism (via env vars).
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ISSUES_DB_PATH", str(tmp_path / "issues.db"))
    import medrag.api.panel.serve as serve_module

    monkeypatch.setattr(serve_module, "DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.delenv(auth.ENV_WRITE_TOKEN, raising=False)
    app = create_panel_app()
    app.testing = True
    with app.test_client() as c:
        yield c


def test_read_endpoints_are_open_no_auth_required(client):
    # lineage.db absent -- returns 503 but NOT 401/403: read requires no auth.
    resp = client.get("/api/stages")
    assert resp.status_code == 503
    assert "lineage.db" in resp.get_json()["error"]


def test_issues_list_is_open_and_empty_without_db(client):
    resp = client.get("/api/issues")
    assert resp.status_code == 200
    assert resp.get_json() == {"error": "issues.db yok", "issues": []}


def test_resolve_fails_closed_when_token_unset(client):
    resp = client.post("/api/issues/1/resolve", json={"resolved_by": "ops"})
    assert resp.status_code == 403


def test_resolve_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "dogru-token")
    resp = client.post(
        "/api/issues/1/resolve",
        json={"resolved_by": "ops"},
        headers={auth.WRITE_TOKEN_HEADER: "yanlis-token"},
    )
    assert resp.status_code == 403


def test_resolve_requires_resolved_by(client, monkeypatch):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "dogru-token")
    resp = client.post(
        "/api/issues/1/resolve",
        json={},
        headers={auth.WRITE_TOKEN_HEADER: "dogru-token"},
    )
    assert resp.status_code == 400


def test_resolve_updates_only_resolved_fields_with_valid_token(client, monkeypatch, tmp_path):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "dogru-token")
    issues_db = tmp_path / "issues.db"
    from medrag.core.db.issues import connect_issues

    con = connect_issues(issues_db)
    issue_id = record_issue(
        con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="doc-1"
    )
    con.close()

    resp = client.post(
        f"/api/issues/{issue_id}/resolve",
        json={"resolved_by": "yetkili-kullanici"},
        headers={auth.WRITE_TOKEN_HEADER: "dogru-token"},
    )
    assert resp.status_code == 200
    assert resp.get_json() == {
        "status": "resolved",
        "id": issue_id,
        "resolved_by": "yetkili-kullanici",
    }

    con2 = sqlite3.connect(issues_db)
    con2.row_factory = sqlite3.Row
    row = con2.execute("SELECT * FROM issues WHERE id=?", (issue_id,)).fetchone()
    con2.close()
    assert row["resolved_by"] == "yetkili-kullanici"
    assert row["resolved_at"] is not None
    # Only resolved_* changed -- the remaining fields are untouched.
    assert row["stage"] == "chunk"
    assert row["source_doc"] == "doc-1"


def test_issues_list_reflects_db_state(client, tmp_path):
    from medrag.core.db.issues import connect_issues

    issues_db = tmp_path / "issues.db"
    con = connect_issues(issues_db)
    record_issue(con, stage="facts", severity="warning", reason="NEW_KEY_PROPOSAL")
    con.close()

    resp = client.get("/api/issues")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["issues"]) == 1
    assert body["issues"][0]["reason"] == "NEW_KEY_PROPOSAL"


def test_index_route_serves_dashboard_before_the_rest_of_the_page(client):
    """The panel's landing page `/` -- the dashboard section is placed BEFORE
    the other sections such as `stages`/`conflicts` on the page; no extra
    click/menu navigation is required."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'id="dashboard"' in html
    assert html.index('id="dashboard"') < html.index('id="stages"')


def test_dashboard_endpoint_without_any_report_is_all_zero(client):
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["last_night"] is None
    assert body["counters"] == {
        "stale_source_changed": 0,
        "stale_stage_changed": 0,
        "ownerless": 0,
        "missing": 0,
        "open_issues": 0,
    }


def test_dashboard_endpoint_surfaces_intentionally_failed_night(client, tmp_path, monkeypatch):
    """Make the night intentionally failed (fixture report JSON), open the
    panel -- does `/api/dashboard` (the source for the landing screen, see the
    `/` test above) show it on the first read?"""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = {
        "run_id": "2026-08-19T23-00-00Z",
        "started_at": "2026-08-19T23:00:00+00:00",
        "finished_at": "2026-08-20T02:00:00+00:00",
        "outcome": "failed",
        "ownership": {"deterministic_count": 0, "model_routed_count": 0, "unresolved_count": 5},
        "derivative_status": {
            "doc-a": {"parse": "kaynak_degisti"},
            "doc-b": {"chunk": "asama_degisti"},
            "doc-c": {"facts": "uretilmemis"},
        },
    }
    (reports_dir / "nightly_2026-08-19.json").write_text(json.dumps(report), encoding="utf-8")
    # The `client` fixture already pointed NIGHTLY_REPORTS_DIR at this reports_dir --
    # we don't write it again here; the fixture is the single source.

    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["last_night"]["outcome"] == "failed"
    assert body["counters"] == {
        "stale_source_changed": 1,
        "stale_stage_changed": 1,
        "ownerless": 5,
        "missing": 1,
        "open_issues": 0,
    }


def test_dashboard_endpoint_includes_open_issue_count(client, tmp_path):
    issues_db = tmp_path / "issues.db"
    from medrag.core.db.issues import connect_issues

    con = connect_issues(issues_db)
    record_issue(con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="doc-1")
    record_issue(con, stage="facts", severity="warning", reason="SKIPPED_FACT", source_doc="doc-2")
    con.close()

    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    assert resp.get_json()["counters"]["open_issues"] == 2


def test_issues_list_filters_by_stage_severity_source_doc_and_reason(client, tmp_path):
    from medrag.core.db.issues import connect_issues

    issues_db = tmp_path / "issues.db"
    con = connect_issues(issues_db)
    record_issue(con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="doc-1")
    record_issue(con, stage="facts", severity="warning", reason="SKIPPED_FACT", source_doc="doc-2")
    record_issue(con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="doc-3")
    con.close()

    resp_all = client.get("/api/issues")
    assert len(resp_all.get_json()["issues"]) == 3

    resp_stage = client.get("/api/issues?stage=facts")
    assert [i["source_doc"] for i in resp_stage.get_json()["issues"]] == ["doc-2"]

    resp_reason = client.get("/api/issues?reason=UNRESOLVED_CHUNK")
    assert {i["source_doc"] for i in resp_reason.get_json()["issues"]} == {"doc-1", "doc-3"}

    resp_combo = client.get("/api/issues?stage=chunk&source_doc=doc-3")
    assert [i["source_doc"] for i in resp_combo.get_json()["issues"]] == ["doc-3"]

    resp_severity = client.get("/api/issues?severity=warning")
    assert [i["source_doc"] for i in resp_severity.get_json()["issues"]] == ["doc-2"]

    resp_none = client.get("/api/issues?stage=vectorize")
    assert resp_none.get_json()["issues"] == []


def test_issues_list_lets_every_unresolved_chunk_be_seen_individually(client, tmp_path):
    """Every unresolved chunk is visible individually in the panel -- multiple
    `UNRESOLVED_CHUNK` records under the same `source_doc` come back as separate
    rows (not aggregated)."""
    from medrag.core.db.issues import connect_issues

    issues_db = tmp_path / "issues.db"
    con = connect_issues(issues_db)
    for i in range(5):
        record_issue(
            con,
            stage="chunk",
            severity="review",
            reason="UNRESOLVED_CHUNK",
            source_doc="doc-1",
            entity_ref=f"doc-1::c{i}",
        )
    con.close()

    resp = client.get("/api/issues?stage=chunk&reason=UNRESOLVED_CHUNK")
    issues = resp.get_json()["issues"]
    assert len(issues) == 5
    assert {i["entity_ref"] for i in issues} == {f"doc-1::c{i}" for i in range(5)}


def test_pipeline_not_imported_by_blueprint_module():
    """Layering note: the panel module does NOT import medrag.pipeline
    (test_import_contracts.py checks the same for the whole package; here it is
    also checked against this module's own IMPORT statements -- a docstring could
    contain the string 'medrag.pipeline', so real import statements are parsed
    rather than doing a raw text search)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(blueprint))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(m == "medrag.pipeline" or m.startswith("medrag.pipeline.") for m in imported_modules)
