"""Interactive HTML visualization for the chunk tree (parser-agnostic).

Takes a `ChunkSet` and produces a single, fully self-contained HTML string:
no CDN/script/font/network dependencies (project principle: tests and
production are offline). Data is embedded into the page via
`<script type="application/json">`; the tree is built in JS.

Why a top-level module (not under `enrichment`): visualization is a
PRESENTATION of the output; it doesn't contribute to chunk production. It
imports nothing but `ChunkSet`, knows no parser types.

Two entry points:
* `render_document_html(chunk_set)` — one document's tree (collapsible
  tree on the left + detail panel on the right; leaf vs ↔ summary node
  distinguished by color).
* `render_index_html(entries)` — index linking all documents in the run.

Security/robustness: document text is NOT embedded as raw HTML — only the
JSON payload is embedded, and JS renders it via `textContent` (so any
`<`, `&` etc. inside content is never interpreted as markup). The `<`/`>`
inside JSON are also escaped to `\\uXXXX` so a `</script>` substring in
embedded text can't break out of the page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medrag.pipeline.chunker.core.chunk import ChunkSet

DOC_SUFFIX = ".tree.html"
INDEX_NAME = "index.html"


@dataclass(frozen=True)
class IndexEntry:
    """One row on the index page: which document, which HTML file."""

    doc_id: str
    html_name: str  # filename relative to index.html
    leaf_count: int
    node_count: int


def _embed_json(data: object) -> str:
    """Make `data` safe to embed inside `<script type="application/json">`:
    `<`/`>`/`&` are escaped to `\\uXXXX` so a `</script>` substring in the
    text can't close the page early (the classic escape rule for embedded
    JSON)."""
    text = json.dumps(data, ensure_ascii=False)
    return (text.replace("<", "\\u003c").replace(">", "\\u003e")
                .replace("&", "\\u0026"))


def render_document_html(chunk_set: ChunkSet) -> str:
    """Self-contained HTML for one document's chunk tree."""
    payload = {
        "doc_id": chunk_set.doc_id,
        "schema_version": chunk_set.schema_version,
        # exclude_none: same simplicity as the JSON output; the detail panel
        # only shows fields that are actually populated.
        "nodes": [n.model_dump(mode="json", exclude_none=True)
                  for n in chunk_set.nodes],
    }
    return _DOC_TEMPLATE.replace("__DATA__", _embed_json(payload))


def render_index_html(entries: list[IndexEntry]) -> str:
    """Self-contained index page linking all documents in the run."""
    payload = [
        {"doc_id": e.doc_id, "html_name": e.html_name,
         "leaf_count": e.leaf_count, "node_count": e.node_count}
        for e in entries
    ]
    return _INDEX_TEMPLATE.replace("__DATA__", _embed_json(payload))


def write_document_html(chunk_set: ChunkSet, out_dir: str | Path) -> Path:
    """Write `{out_dir}/{doc_id}.tree.html` and return its path."""
    path = Path(out_dir) / f"{chunk_set.doc_id}{DOC_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_document_html(chunk_set), encoding="utf-8")
    return path


def write_index_html(entries: list[IndexEntry], out_dir: str | Path) -> Path:
    """Write `{out_dir}/index.html` and return its path."""
    path = Path(out_dir) / INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_index_html(entries), encoding="utf-8")
    return path


# -- Templates -------------------------------------------------------------------
# Single file: embedded CSS + JS, no external resources. The `__DATA__`
# placeholder is replaced with the embedded JSON. Document content is
# ALWAYS written via textContent in JS.

_DOC_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chunker — chunk tree</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1c2024; --muted: #6b7480; --line: #e3e6ea;
    --panel: #f6f7f9; --accent: #2f6feb; --leaf: #1f9d55; --node: #8a5cf6;
    --chip: #eceef1; --code: #f0f2f4;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171a; --fg: #e6e8eb; --muted: #97a0aa; --line: #2a2f35;
      --panel: #1b1f24; --accent: #6ea0ff; --leaf: #4ad07f; --node: #b18cff;
      --chip: #262b31; --code: #1f242a;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--fg); display: flex; flex-direction: column;
  }
  header {
    padding: 10px 16px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  header .title { font-weight: 600; }
  header .stats { color: var(--muted); font-size: 13px; }
  header a { color: var(--accent); text-decoration: none; }
  #search {
    margin-left: auto; padding: 6px 10px; border: 1px solid var(--line);
    border-radius: 6px; background: var(--bg); color: var(--fg); min-width: 220px;
  }
  .main { flex: 1; display: flex; min-height: 0; }
  #tree {
    width: 42%; min-width: 280px; overflow: auto; padding: 8px 4px;
    border-right: 1px solid var(--line);
  }
  #detail { flex: 1; overflow: auto; padding: 16px 20px; }
  .row {
    display: flex; align-items: flex-start; gap: 6px; padding: 3px 6px;
    border-radius: 6px; cursor: pointer; white-space: nowrap;
  }
  .row:hover { background: var(--panel); }
  .row.sel { background: color-mix(in srgb, var(--accent) 18%, transparent); }
  .toggle {
    width: 14px; flex: none; text-align: center; color: var(--muted);
    user-select: none;
  }
  .toggle.leaf { visibility: hidden; }
  .badge {
    flex: none; font-size: 11px; font-weight: 600; padding: 1px 6px;
    border-radius: 10px; color: #fff;
  }
  .badge.leaf { background: var(--leaf); }
  .badge.node { background: var(--node); }
  .label { overflow: hidden; text-overflow: ellipsis; }
  .children { margin-left: 16px; }
  .children.collapsed { display: none; }
  .hidden { display: none; }
  #detail h2 { margin: 0 0 4px; font-size: 16px; }
  #detail .sub { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
  .section-title {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .04em; color: var(--muted); margin: 18px 0 6px;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    background: var(--chip); border-radius: 12px; padding: 2px 10px; font-size: 12px;
  }
  pre.text {
    background: var(--code); border: 1px solid var(--line); border-radius: 8px;
    padding: 12px; white-space: pre-wrap; word-break: break-word; margin: 0;
  }
  table.meta { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.meta td { border-top: 1px solid var(--line); padding: 5px 8px; vertical-align: top; }
  table.meta td.k { color: var(--muted); width: 190px; white-space: nowrap; }
  table.meta td.v { word-break: break-word; }
  .placeholder { color: var(--muted); margin-top: 40px; text-align: center; }
</style>
</head>
<body>
<header>
  <span class="title">Chunk tree: <span id="doc-id"></span></span>
  <span class="stats" id="stats"></span>
  <a href="index.html" id="index-link">← index</a>
  <input id="search" type="search" placeholder="search the tree…" autocomplete="off">
</header>
<div class="main">
  <div id="tree"></div>
  <div id="detail"><div class="placeholder">Select a node.</div></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const nodes = data.nodes;
  const byId = new Map(nodes.map(n => [n.node_id, n]));
  document.getElementById("doc-id").textContent = data.doc_id;
  const leafCount = nodes.filter(n => (n.tree_level || 0) === 0).length;
  document.getElementById("stats").textContent =
    nodes.length + " nodes · " + leafCount + " leaves · schema v" + data.schema_version;

  // Roots: nodes without parent_id. Highest level (deepest "root") first.
  const roots = nodes.filter(n => !n.parent_id)
    .sort((a, b) => (b.tree_level || 0) - (a.tree_level || 0)
      || a.node_id.localeCompare(b.node_id));

  function labelOf(n) {
    const hp = n.heading_path;
    if (hp && hp.length) return hp[hp.length - 1];
    const line = (n.text || "").split("\\n").find(s => s.trim());
    const t = (line || "(empty)").trim();
    return t.length > 90 ? t.slice(0, 90) + "…" : t;
  }

  const treeEl = document.getElementById("tree");
  let selectedRow = null;

  function makeRow(n, depth) {
    const isLeaf = (n.tree_level || 0) === 0;
    const kids = (n.child_ids || []).map(id => byId.get(id)).filter(Boolean);

    const row = document.createElement("div");
    row.className = "row";
    row.style.paddingLeft = (depth * 16 + 6) + "px";
    row.dataset.nodeId = n.node_id;

    const toggle = document.createElement("span");
    toggle.className = "toggle" + (kids.length ? "" : " leaf");
    toggle.textContent = kids.length ? "▸" : "•";
    row.appendChild(toggle);

    const badge = document.createElement("span");
    badge.className = "badge " + (isLeaf ? "leaf" : "node");
    badge.textContent = "L" + (n.tree_level || 0);
    row.appendChild(badge);

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = labelOf(n);
    row.appendChild(label);

    const childBox = document.createElement("div");
    childBox.className = "children collapsed";
    let expanded = false;
    if (kids.length) {
      toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        expanded = !expanded;
        childBox.classList.toggle("collapsed", !expanded);
        toggle.textContent = expanded ? "▾" : "▸";
      });
    }
    row.addEventListener("click", () => selectNode(n, row));

    const wrap = document.createElement("div");
    wrap.appendChild(row);
    for (const k of kids) childBox.appendChild(makeRow(k, depth + 1));
    wrap.appendChild(childBox);
    return wrap;
  }

  for (const r of roots) treeEl.appendChild(makeRow(r, 0));

  const detailEl = document.getElementById("detail");
  // Fields not shown in the detail panel (already in their own sections / noise).
  const SKIP = new Set(["text", "summary", "keywords", "node_id", "doc_id"]);

  function addSection(title) {
    const h = document.createElement("div");
    h.className = "section-title";
    h.textContent = title;
    detailEl.appendChild(h);
  }

  function selectNode(n, row) {
    if (selectedRow) selectedRow.classList.remove("sel");
    if (row) { row.classList.add("sel"); selectedRow = row; }
    detailEl.innerHTML = "";

    const h = document.createElement("h2");
    h.textContent = labelOf(n);
    detailEl.appendChild(h);
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = n.node_id + "  ·  tree_level " + (n.tree_level || 0)
      + ((n.tree_level || 0) === 0 ? " (leaf)" : " (summary)");
    detailEl.appendChild(sub);

    if (n.summary) {
      addSection("Summary");
      const p = document.createElement("pre");
      p.className = "text";
      p.textContent = n.summary;
      detailEl.appendChild(p);
    }
    if (n.keywords && n.keywords.length) {
      addSection("Keywords");
      const box = document.createElement("div");
      box.className = "chips";
      for (const k of n.keywords) {
        const c = document.createElement("span");
        c.className = "chip";
        c.textContent = k;
        box.appendChild(c);
      }
      detailEl.appendChild(box);
    }

    addSection("Text");
    const pre = document.createElement("pre");
    pre.className = "text";
    pre.textContent = n.text || "(empty)";
    detailEl.appendChild(pre);

    addSection("Metadata");
    const table = document.createElement("table");
    table.className = "meta";
    for (const key of Object.keys(n)) {
      if (SKIP.has(key)) continue;
      const tr = document.createElement("tr");
      const k = document.createElement("td");
      k.className = "k"; k.textContent = key;
      const v = document.createElement("td");
      v.className = "v";
      const val = n[key];
      v.textContent = (typeof val === "object")
        ? JSON.stringify(val, null, 2) : String(val);
      tr.appendChild(k); tr.appendChild(v);
      table.appendChild(tr);
    }
    detailEl.appendChild(table);
  }

  // Search: keep matching nodes AND their ancestors visible; hide the rest.
  const search = document.getElementById("search");
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    const rows = treeEl.querySelectorAll(".row");
    if (!q) {
      rows.forEach(r => r.parentElement.classList.remove("hidden"));
      treeEl.querySelectorAll(".children").forEach(c => {
        c.classList.remove("hidden");
      });
      return;
    }
    // Set of matching node_ids + all their ancestors.
    const keep = new Set();
    for (const n of nodes) {
      const hay = ((n.text || "") + " " + (n.summary || "") + " "
        + (n.keywords || []).join(" ") + " "
        + (n.heading_path || []).join(" ")).toLowerCase();
      if (hay.includes(q)) {
        let cur = n;
        while (cur) { keep.add(cur.node_id); cur = byId.get(cur.parent_id); }
      }
    }
    rows.forEach(r => {
      const on = keep.has(r.dataset.nodeId);
      r.parentElement.classList.toggle("hidden", !on);
    });
  });
})();
</script>
</body>
</html>
"""

_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chunker — run index</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1c2024; --muted: #6b7480; --line: #e3e6ea;
    --panel: #f6f7f9; --accent: #2f6feb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171a; --fg: #e6e8eb; --muted: #97a0aa; --line: #2a2f35;
      --panel: #1b1f24; --accent: #6ea0ff;
    }
  }
  * { box-sizing: border-box; }
  body {
    font: 15px/1.6 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--fg); margin: 0; padding: 32px;
    max-width: 820px; margin-inline: auto;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; }
  ul { list-style: none; padding: 0; margin: 0; }
  li {
    border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px;
    padding: 14px 18px; display: flex; align-items: baseline; gap: 12px;
  }
  li:hover { background: var(--panel); }
  a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; margin-left: auto; }
  .empty { color: var(--muted); }
</style>
</head>
<body>
<h1>chunker — visualization index</h1>
<div class="sub" id="sub"></div>
<ul id="list"></ul>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const entries = JSON.parse(document.getElementById("data").textContent);
  document.getElementById("sub").textContent = entries.length + " documents";
  const list = document.getElementById("list");
  if (!entries.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No documents were produced in this run.";
    list.appendChild(li);
    return;
  }
  for (const e of entries) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = e.html_name;
    a.textContent = e.doc_id;
    li.appendChild(a);
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = e.node_count + " nodes · " + e.leaf_count + " leaves";
    li.appendChild(meta);
    list.appendChild(li);
  }
})();
</script>
</body>
</html>
"""