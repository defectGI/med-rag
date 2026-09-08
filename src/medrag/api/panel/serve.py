"""
Observation panel server. A stdlib `http.server` on localhost only, no new
dependency. **Read-only** -- no endpoint triggers or stops the pipeline (this is
an observation-only layer). Even `/api/cascade` only RETURNS the information
"if you re-run this, these go stale"; it RUNS nothing.

Run: first `python ingest.py`, then `python serve.py`.
Browser: http://localhost:8010/
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from medrag.core.db.sqlite import BUSY_TIMEOUT_MS

PANEL_DIR = os.path.dirname(os.path.abspath(__file__))
# lineage.db is stored outside the repo -- from src/medrag/api/panel/ the repo
# root is reached via parents[4].
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "lineage_panel_data"
DB_PATH = str(DATA_DIR / "lineage.db")
STATIC_DIR = os.path.join(PANEL_DIR, "static")
PORT = 8010

# Fixed stage order -- the cascade warning is computed against it (per document).
STAGE_ORDER = ["parse", "chunk", "facts", "vectorize"]


def _connect():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    # The WAL exemption on specs.db does not apply here -- a read-only open does
    # not touch journal_mode, only busy_timeout is set.
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.row_factory = sqlite3.Row
    return conn


def _latest_ingested_at(conn):
    row = conn.execute("SELECT MAX(ingested_at) AS t FROM pipeline_runs").fetchone()
    return row["t"] if row else None


def api_stages(conn):
    """Per stage: status summary from the latest ingest pass."""
    latest = _latest_ingested_at(conn)
    out = []
    for stage in STAGE_ORDER:
        rows = conn.execute(
            "SELECT status, is_stale, ran_at FROM pipeline_runs WHERE stage=? AND ingested_at=?",
            (stage, latest),
        ).fetchall()
        if not rows:
            out.append({"stage": stage, "total": 0, "stale": 0, "last_ran_at": None})
            continue
        stale = sum(1 for r in rows if r["is_stale"])
        ran_ats = [r["ran_at"] for r in rows if r["ran_at"]]
        out.append({
            "stage": stage,
            "total": len(rows),
            "stale": stale,
            "not_run": sum(1 for r in rows if not r["ran_at"]),
            "last_ran_at": max(ran_ats) if ran_ats else None,
        })
    return {"generated_from_ingest_at": latest, "stages": out}


def api_conflicts(conn, params):
    latest = _latest_ingested_at(conn)
    q = "SELECT * FROM conflicts WHERE ingested_at=?"
    args = [latest]
    product_code = params.get("product_code", [None])[0]
    if product_code:
        q += " AND product_code=?"
        args.append(product_code)
    rows = conn.execute(q + " ORDER BY product_code, key", args).fetchall()
    return {"generated_from_ingest_at": latest, "conflicts": [dict(r) for r in rows]}


def api_graph(conn, params):
    node_type = params.get("node_type", ["document"])[0]
    node_id = params.get("node_id", [None])[0]
    if not node_id:
        return {"error": "node_id gerekli"}
    latest = _latest_ingested_at(conn)
    up = conn.execute(
        "SELECT * FROM lineage_edges WHERE dst_type=? AND dst_id=? AND ingested_at=?",
        (node_type, node_id, latest),
    ).fetchall()
    down = conn.execute(
        "SELECT * FROM lineage_edges WHERE src_type=? AND src_id=? AND ingested_at=?",
        (node_type, node_id, latest),
    ).fetchall()
    return {
        "node": {"type": node_type, "id": node_id},
        "upstream": [dict(r) for r in up],
        "downstream": [dict(r) for r in down],
    }


def api_cascade(conn, params):
    """'If you re-run this stage, the ones below go stale' -- information only,
    triggers nothing."""
    stage = params.get("stage", [None])[0]
    node_id = params.get("node_id", [None])[0]
    if stage not in STAGE_ORDER or not node_id:
        return {"error": "stage ve node_id gerekli"}
    idx = STAGE_ORDER.index(stage)
    downstream_stages = STAGE_ORDER[idx + 1:]
    latest = _latest_ingested_at(conn)
    affected = []
    for s in downstream_stages:
        row = conn.execute(
            "SELECT status, ran_at FROM pipeline_runs WHERE stage=? AND node_id=? AND ingested_at=?",
            (s, node_id, latest),
        ).fetchone()
        if row and row["ran_at"]:
            affected.append({"stage": s, "current_status": row["status"], "current_ran_at": row["ran_at"]})
    return {
        "stage": stage, "node_id": node_id,
        "would_go_stale": affected,
        "note": "Bu sadece bilgi amaçlıdır; hiçbir aşama gerçekten tetiklenmedi.",
    }


ROUTES = {
    "/api/stages": lambda conn, params: api_stages(conn),
    "/api/conflicts": api_conflicts,
    "/api/graph": api_graph,
    "/api/cascade": api_cascade,
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        handler = ROUTES.get(parsed.path)
        if handler is None:
            super().do_GET()
            return
        params = urllib.parse.parse_qs(parsed.query)
        if not os.path.exists(DB_PATH):
            self._json_response(503, {"error": "lineage.db yok -- once `python ingest.py` calistirin"})
            return
        conn = _connect()
        try:
            result = handler(conn, params)
        finally:
            conn.close()
        self._json_response(200, result)

    def _json_response(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Panel: http://localhost:{PORT}/")
        httpd.serve_forever()
