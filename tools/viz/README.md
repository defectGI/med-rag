# ACME Product Family Tree — visualizer

A front-end-only visualizer that reads `product_nodes.json` and
`document_nodes.json` and renders them as an expandable box/line tree.
Clicking a node expands/collapses it; clicking a file/image opens it in
the OS's default application.

## Overview

This tool is an *output consumer* of the pipeline, not part of it:
`chatbot-corpus/product_info/export_products.py` and `chatbot-corpus/
document_info/classify_documents.py` produce `product_nodes.json` and
`document_nodes.json`, respectively; `tools/viz/` reads those two files
and presents them as a navigable tree in the browser. It runs after the
parse/chunk/vectorize chain as a separate, optional inspection layer —
no component imports `tools/viz/`.

`tools/` is an installed package (`tools.benchmark`, `tools.viz`, …), so
the module entry point is `python -m tools.viz.serve`.

## Known issue: data paths

`serve.py` (and `app.js`) locate the data files relative to the parent of
`tools/viz/` — i.e. it expects them under `tools/{product_info,
document_info}/`. The producer scripts, however, still write them to
their original location under `chatbot-corpus/`. Until this is fixed, the
two options are: (a) copy or symlink the JSON files into
`tools/{product_info, document_info}/` before running the server, or
(b) update the paths in `serve.py` and `app.js`. Without one of those,
the server crashes with `FileNotFoundError`.

## How it works

- From the root "ACME" box, families, categories, and products branch
  out; click any node to expand or collapse it.
- Files directly attached to a node (`owner_ids` in
  `document_nodes.json`) appear as dashed boxes under that node's own
  branch — this also applies to non-product nodes (family/category).
- Image files (`doc_type: PRODUCT_IMAGE`) render as small thumbnails —
  resized JPEGs cached on disk under `.thumb_cache/` (originals can run
  into tens of MB).
- Clicking a file/image opens it in the OS's default app via the
  `/open` endpoint (Windows-specific `os.startfile`); both `/open` and
  `/thumb` only accept paths registered in `document_nodes.json`
  (whitelist) and only bind to `127.0.0.1`.

## Requirements

Thumbnail generation needs `Pillow` (declared in the root
`pyproject.toml`; install via `uv sync` / `pip install -e .`). If
`Pillow` is missing, `/thumb` silently returns 404; the rest of the tree
view is unaffected.

## Running

After the data files mentioned in the "Known issue" section are in
place, from the repo root:

```
python -m tools.viz.serve
```

If port 8000 is busy, the server falls forward to the first free port
in 8000-8009 (the console prints the actual port). Open in browser:

```
http://localhost:8000/viz/
```

You can also open the static page with `python -m http.server`, but
click-to-open (`/open`) and thumbnails (`/thumb`) won't work that way —
you must run it through `serve.py` for those endpoints.

## Configuration

There are no secret/sensitive settings; constants like `PORT` and
`THUMB_MAX_SIZE` live as in-code values in `serve.py`.