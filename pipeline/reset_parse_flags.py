"""Force a full re-parse: null every active record's parse.parsed_from_hash
in document_nodes.json so run_parse_pipeline.py's incremental check
(_needs_parse) treats the whole corpus as unparsed.

This is an ESCAPE HATCH, not the normal way to re-parse after a code change:
`_needs_parse` already re-parses anything whose `parse.parser_version` is
older than the current PARSER_VERSION (parser/parsers/base.py), so bumping
that constant -- which every behavior-changing parser fix must do -- is the
supported path. Reach for this script when the version gate cannot see the
change: an edit that (deliberately) did not bump PARSER_VERSION, a corrupted
or partially-written output dir, or a storage/ cache wipe you want reflected
everywhere.

Chunk outputs need no equivalent reset: chunker re-derives any set whose
provenance no longer matches its IR, so a re-parse that actually changes
the IR pulls the chunks along by itself.

Atomic write (tmp + os.replace), same as the pipeline itself.

Run from pipeline/:  python reset_parse_flags.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# moved run_parse_pipeline.py and its .env to src/medrag/pipeline/cli/ ;
# this script stayed behind in pipeline/ (it's a standalone escape hatch,
# not part of that package) but DOCUMENT_NODES_PATH has one home now, not
# a duplicated copy -- read it from the same .env run_parse_pipeline.py
# uses, and resolve the path relative to THAT file's own directory (paths
# inside it are relative to cli/, not to pipeline/).
_ENV_DIR = BASE_DIR.parent / "src" / "urun" / "pipeline" / "cli"
load_dotenv(_ENV_DIR / ".env")

path = (_ENV_DIR / os.environ["DOCUMENT_NODES_PATH"]).resolve()
data = json.loads(path.read_text(encoding="utf-8"))

n = 0
for record in data["documents"]:
    if record.get("is_active", True):
        record["parse"]["parsed_from_hash"] = None
        n += 1

fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
os.replace(tmp, path)

print(f"{n} / {len(data['documents'])} records reset -> {path}")
