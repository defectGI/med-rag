#!/usr/bin/env bash
# Baseline capture script. Idempotent: each run opens a fresh
# tools/baseline/runs/<timestamp>/ directory, never overwriting.
#
# This script writes ZERO bytes to specs.db (all connections mode=ro) and
# does not touch production files. The pipeline step (below) reads and
# writes document_nodes.json via a SCRATCH COPY rather than the live
# file -- the constant DOCUMENT_NODES_PATH points at the scratch copy in
# every branch, on purpose.
#
# Usage: bash tools/baseline/capture_baseline.sh
#   Optional environment variables:
#     WEBAPP_URL   default http://127.0.0.1:8507 -- the address at which
#                  the chatbot webapp is ALREADY running. The script
#                  does NOT start the webapp itself (preflight + Ollama
#                  round-trips can take minutes, and environment-specific
#                  issues like port conflicts can arise). If it is not
#                  up, the script skips that step and records the reason
#                  in runs/<ts>/README.txt.
#     RUN_PIPELINE=1  also run the single-product parse->chunk->vectorize
#                  scratch step in 2d) (default: skip -- slow, makes real
#                  Ollama/embedding calls).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$REPO_ROOT/tools/baseline/runs/$TS"
mkdir -p "$RUN_DIR"

# Windows-style paths: native python.exe does not understand git-bash's
# /c/... paths (the leading '/' + 'c' drive letter is NOT translated,
# it is taken as a literal directory). Any path passed as an env variable
# to Python subprocesses uses the _WIN variant; bash's own file
# operations (mkdir/cp) use the original (git-bash) path.
_winpath() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
REPO_ROOT_WIN="$(_winpath "$REPO_ROOT")"
RUN_DIR_WIN="$(_winpath "$RUN_DIR")"
echo "== baseline capture -> $RUN_DIR =="

WEBAPP_URL="${WEBAPP_URL:-http://127.0.0.1:8507}"
RUN_PIPELINE="${RUN_PIPELINE:-0}"

README="$RUN_DIR/README.txt"
: > "$README"

# ---------------------------------------------------------------------------
# 1) Environment fingerprint (no secrets -- only version/name/counter)
# ---------------------------------------------------------------------------
{
  echo "timestamp: $TS"
  echo "git_sha: $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>&1)"
  echo "git_branch: $(cd "$REPO_ROOT" && git branch --show-current 2>&1)"
  echo "python: $(python --version 2>&1)"
  echo
  echo "-- ollama models (name + digest, localhost:11434) --"
  curl -s -m 5 http://localhost:11434/api/tags 2>/dev/null \
    | python -c "import json,sys; d=json.load(sys.stdin); [print(m['name'], m['digest'][:12]) for m in d.get('models',[])]" 2>&1 \
    || echo "(ollama unreachable)"
  echo
  echo "-- qdrant collections (localhost:6333) --"
  cols=$(curl -s -m 5 http://localhost:6333/collections 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); [print(c['name']) for c in d['result']['collections']]" 2>&1)
  if [ -z "$cols" ]; then
    echo "(no collections)"
  else
    for c in $cols; do
      info=$(curl -s -m 5 "http://localhost:6333/collections/$c" 2>/dev/null)
      pc=$(echo "$info" | python -c "import json,sys; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null)
      dim=$(echo "$info" | python -c "import json,sys; d=json.load(sys.stdin); v=d['result']['config']['params']['vectors']; print(v.get('size', v))" 2>/dev/null)
      echo "$c: points_count=$pc dim=$dim"
    done
  fi
  echo
  echo "qdrant_version: $(curl -s -m 5 http://localhost:6333/ 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin)['version'])" 2>/dev/null)"
} > "$RUN_DIR/env_fingerprint.txt" 2>&1

pip freeze > "$RUN_DIR/pip_freeze.txt" 2>&1

# ---------------------------------------------------------------------------
# 2) Env key names + set/unset (NO VALUES)
# ---------------------------------------------------------------------------
{
  for f in chatbot/.env src/medrag/pipeline/vectorize/.env facts/.env retrieval/.env pipeline/.env parser/.env chunker/.env; do
    path="$REPO_ROOT/$f"
    echo "=== $f ==="
    if [ ! -f "$path" ]; then
      echo "(file missing)"
      continue
    fi
    python -c "
import re
with open(r'$path', encoding='utf-8') as fh:
    for line in fh:
        m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)\$', line.rstrip('\n'))
        if m:
            key, val = m.group(1), m.group(2).strip()
            print(f'{key}: {\"set\" if val else \"unset(empty)\"}')"
  done
} > "$RUN_DIR/env_keys.txt" 2>&1

# ---------------------------------------------------------------------------
# 3) Three representative questions, each 3x in a clean session
#    (no cookie -> each call generates its own uuid4 session_id)
# ---------------------------------------------------------------------------
Q1="PN1162 ürününün liste fiyatı nedir?"
Q2="PN1169 PXIe Jetson AGX Orin modülünün GMSL2 arayüzü hakkında bilgi ver"
Q3="Fiyat teklifi için kiminle iletişime geçmeliyim?"

if curl -s -m 3 -o /dev/null "$WEBAPP_URL/"; then
  : > "$RUN_DIR/timings.txt"
  qi=0
  for q in "$Q1" "$Q2" "$Q3"; do
    qi=$((qi+1))
    for run in 1 2 3; do
      payload=$(python -c "import json,sys; print(json.dumps({'message': sys.argv[1]}))" "$q")
      t0=$(date +%s%3N)
      curl -s -m 90 -X POST "$WEBAPP_URL/api/chat" \
        -H "Content-Type: application/json" -d "$payload" \
        > "$RUN_DIR/q${qi}_run${run}.json"
      t1=$(date +%s%3N)
      echo "q${qi} run${run} elapsed_ms=$((t1-t0)) question=\"$q\"" >> "$RUN_DIR/timings.txt"
    done
  done
  echo "chatbot questions: complete (3 questions x 3 runs), see q*_run*.json + timings.txt" >> "$README"
else
  echo "chatbot webapp is not running at $WEBAPP_URL -- question step SKIPPED." >> "$README"
  echo "To start it: cd chatbot && python -m chatbot.webapp  (includes preflight, may take minutes; on a port conflict try CHATBOT_WEB_PORT for a different port)" >> "$README"
fi

# ---------------------------------------------------------------------------
# 4) Test count
# ---------------------------------------------------------------------------
( cd "$REPO_ROOT" && python -m pytest --collect-only -q ) > "$RUN_DIR/pytest_collect.txt" 2>&1
tail -3 "$RUN_DIR/pytest_collect.txt" >> "$README"

# ---------------------------------------------------------------------------
# 5) Pipeline baseline (optional, RUN_PIPELINE=1 -- slow + makes real
#    embedding calls). document_nodes.json is read/written via a SCRATCH
#    COPY; the live file is never written.
# ---------------------------------------------------------------------------
if [ "$RUN_PIPELINE" = "1" ]; then
  PSCRATCH="$RUN_DIR/pipeline_scratch"
  mkdir -p "$PSCRATCH/parsed_output" "$PSCRATCH/chunks_output"
  cp "$REPO_ROOT/chatbot-corpus/document_info/document_nodes.json" "$PSCRATCH/document_nodes.json"
  PSCRATCH_WIN="$(_winpath "$PSCRATCH")"
  cat > "$PSCRATCH/vectorize_override.toml" << EOF
[qdrant]
collection_name = "baseline_scratch_$TS"
EOF
  (
    cd "$REPO_ROOT/pipeline" && \
    PARSED_OUTPUT_DIR="$PSCRATCH_WIN\\parsed_output" \
    DOCUMENT_NODES_PATH="$PSCRATCH_WIN\\document_nodes.json" \
    python run_parse_pipeline.py --limit 1
  ) > "$RUN_DIR/pipeline_01_parse.log" 2>&1
  (
    cd "$REPO_ROOT/pipeline" && \
    CHUNKER_INPUT_DIR="$PSCRATCH_WIN\\parsed_output" \
    CHUNKS_OUTPUT_DIR="$PSCRATCH_WIN\\chunks_output" \
    DOCUMENT_NODES_PATH="$PSCRATCH_WIN\\document_nodes.json" \
    python run_chunk_pipeline.py
  ) > "$RUN_DIR/pipeline_02_chunk.log" 2>&1
  (
    cd "$REPO_ROOT/src/medrag/pipeline/vectorize" && \
    VECTORIZE_INPUT_DIR="$PSCRATCH_WIN\\chunks_output" \
    VECTORIZE_CONFIG="$PSCRATCH_WIN\\vectorize_override.toml" \
    VECTORIZE_OUTPUT_DIR="$PSCRATCH_WIN\\vectorize_state" \
    python -m medrag.pipeline.vectorize
  ) > "$RUN_DIR/pipeline_03_vectorize.log" 2>&1
  # drop the scratch collection -- it doesn't need to persist, was only for counter measurement
  curl -s -X DELETE "http://localhost:6333/collections/baseline_scratch_$TS" > /dev/null 2>&1
  {
    echo "-- pipeline scratch summary --"
    tail -2 "$RUN_DIR/pipeline_01_parse.log"
    tail -2 "$RUN_DIR/pipeline_02_chunk.log"
    tail -3 "$RUN_DIR/pipeline_03_vectorize.log"
  } >> "$README"
  # verify the live document_nodes.json was not written (paranoia check)
  if ! git -C "$REPO_ROOT" diff --quiet -- chatbot-corpus/document_info/document_nodes.json; then
    echo "!!! WARNING: live document_nodes.json appears to have changed, check: git diff chatbot-corpus/document_info/document_nodes.json" >> "$README"
  fi
else
  echo "pipeline scratch run skipped (enable with RUN_PIPELINE=1)." >> "$README"
fi

# ---------------------------------------------------------------------------
# 6) Did specs.db change? (paranoia check, read-only connection)
# ---------------------------------------------------------------------------
( cd "$REPO_ROOT" && python -c "
import sqlite3, hashlib
with open('facts/db/specs.db', 'rb') as f:
    print('specs_db_sha256:', hashlib.sha256(f.read()).hexdigest())
con = sqlite3.connect('file:facts/db/specs.db?mode=ro', uri=True)
cur = con.cursor()
cur.execute('PRAGMA integrity_check;')
print('specs_db_integrity:', cur.fetchone()[0])
con.close()
" ) >> "$README" 2>&1

echo "== done: $RUN_DIR =="
cat "$README"