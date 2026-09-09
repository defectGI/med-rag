# med-rag — Deploy & Operations Guide

> Goal: a running installation **tonight** on a home machine (via Coolify or plain
> `docker compose`). Setup → backup → restore → security are all in this file. The full
> environment-variable catalogue is in [`CONFIG.md`](CONFIG.md); the user manual is
> [`USAGE.md`](USAGE.md).

> **Naming note:** some runtime directory and variable names are Turkish (a legacy of the
> original project). They are preserved so an existing deployment keeps working:
> `korpus/` = corpus, `loglar/` = logs, `BELGELER/` = source documents, `durum/` = status,
> `isler/` = job queue, `yedekler/` = backups, `MEDRAG_SIFRE` = password,
> `MEDRAG_GIZLI_ANAHTAR` = secret signing key, `MEDRAG_MAX_YUKLEME_MB` = max upload size MB.

## 1. Quick setup (docker compose)

```bash
cp .env.example .env          # fill in: MEDRAG_SIFRE (password) + LLM/embedding values
docker compose up -d --build  # web + pipeline-worker + pipeline + qdrant
```

Four services:

| Service | Role |
|---|---|
| `web` | SPA + API (gunicorn, port 8507). Upload, chat, library, notes. |
| `pipeline-worker` | Job queue (`/corpus/isler`): processes an uploaded file through parse→chunk→vectorize, and handles delete/cleanup. Single replica, sequential run. |
| `pipeline` | Idle (`sleep infinity`) `docker exec` target — the nightly chain and backups are run by exec-ing into this container. |
| `qdrant` | Vector store (pinned `v1.19.0`). |

Health check:

```bash
docker compose ps                 # web should be healthy
curl -s http://localhost:8507/api/auth/session
```

Browser: `http://<machine-ip>:8507` → password screen → library.

### First run

1. Always set `MEDRAG_SIFRE` in `.env` (if empty, auth is **disabled**).
2. If LLM/embedding endpoints run on the host, use `http://host.docker.internal:...`
   (`extra_hosts` is already declared in compose for Linux).
3. Upload the first document from the library; the status badge follows
   `queued → processing → ready`. First processing downloads the embedding model
   (can be slow).
4. **Critical alignment:** `EMBEDDING_MODEL` + the Qdrant collection name must be the
   same on all three sides (vectorize, retrieval, chatbot). To change the model, delete
   and recreate the collection (`curl -X DELETE
   http://localhost:6333/collections/<name>`), then re-upload all documents.

## 2. Layout (volume scheme)

A single persistent root: `CORPUS_HOST_DIR` (e.g. `/home/user/med-rag/korpus`):

```
korpus/
  BELGELER/            source documents + notes/ (SINGLE SOURCE OF TRUTH)
  document_nodes.json  registry (event-driven; written by upload/delete)
  parsed/              IR + markdown derivations
  chunks/              all_chunks.json
  vectorize/           embedding state
  durum/               per-file processing status (feeds the badge)
  isler/               job queue (file-based)
  storage/images       parser blob store (DATA LOSS point — must be persistent)
  yedekler/            nightly backups (below)
loglar/
  chatbot/             application logs
  conversations/       chat transcripts (accountability trace)
```

`web` and `pipeline-worker` both read-write the corpus (the API is a writer:
upload → BELGELER + registry + isler/). The nightly chain is run only from the `pipeline`
container.

## 3. Nightly backup (K17/E3)

The backup has two layers:

1. **Infrastructure** (`nightly_backup.run_backup`): registry, chunks, parse output,
   specs.db + **Qdrant snapshot** (server-side; `qdrant/snapshot_name.txt`).
2. **med-rag extras**: source corpus `BELGELER/` (documents + notes) and `durum/`.

Run it (cron or a Coolify scheduled task — target: the `pipeline` container):

```bash
docker exec med-rag-pipeline-1 python -m medrag.pipeline.lifecycle.backup_cli
```

Backup root: `NIGHTLY_BACKUP_ROOT` (mounted to `/corpus/nightly_backups` in compose
instead of `/corpus/yedekler`; on the host `$CORPUS_HOST_DIR/nightly_backups`). Cron
example (every night at 03:00):

```cron
0 3 * * * docker exec med-rag-pipeline-1 python -m medrag.pipeline.lifecycle.backup_cli >> /home/user/med-rag/loglar/backup.log 2>&1
```

### 3.1. Restore procedure (drill steps)

> Principle: restore returns to the state at backup time. Even if derivations (parsed/chunks)
> come back from a bad backup, the source `BELGELER/` is always in your hands.

```bash
# 0) Stop the worker (so it takes no new jobs); keep web up so we can test afterwards
docker compose stop pipeline-worker

# 1) Determine the latest backup
YEDEK=$(ls -1d $CORPUS_HOST_DIR/nightly_backups/*/ | sort | tail -1); echo $YEDEK
cat "$YEDEK/manifest.json"   # check for missing/lost items

# 2) Restore the infrastructure derivations (registry, chunks, parsed, specs.db)
docker exec med-rag-pipeline-1 python -c "
from medrag.pipeline.cli import nightly_backup
nightly_backup.restore_backup('$YEDEK')
"
# Note: the qdrant item is SKIPPED here (snapshot is server-side) — that is step 3.

# 3) Restore the Qdrant snapshot
SNAP=$(cat "$YEDEK/qdrant/snapshot_name.txt")
# 3a) If the snapshot file is not on the qdrant volume, copy it first:
#     docker cp "$YEDEK/qdrant/$SNAP" med-rag-qdrant-1:/qdrant/storage/snapshots/
docker exec med-rag-qdrant-1 curl -s -X PUT \
  http://localhost:6333/collections/medrag_chunks/snapshots/recover \
  -H 'Content-Type: application/json' \
  -d "{\"location\": \"snapshots/$SNAP\"}"

# 4) Restore the source corpus + notes + status (med-rag extras) by hand
rsync -a --delete "$YEDEK/BELGELER/" "$CORPUS_HOST_DIR/BELGELER/"
rsync -a "$YEDEK/durum/"      "$CORPUS_HOST_DIR/durum/"

# 5) Restart the worker; make sure the queue is clean
rm -f "$CORPUS_HOST_DIR"/isler/*.json "$CORPUS_HOST_DIR"/isler/*.claim 2>/dev/null
docker compose start pipeline-worker

# 6) Drill verification
curl -s http://localhost:8507/api/library/documents | head -c 400
# → if the document list comes back, restore is complete. Open a document and ask a
#   question in chat; if the citation appears, the Qdrant restore is sound too.
```

`restore_backup` rewrites sqlite with a WAL-safe copy (`_restore_sqlite`) and returns all
items in manifest order except qdrant.

## 4. Observability (E4)

| What | Where |
|---|---|
| Application logs (api) | `$LOGS_HOST_DIR/chatbot/` |
| Chat transcripts | `$LOGS_HOST_DIR/conversations/` (the single accountability trace) |
| Processing status | `$CORPUS_HOST_DIR/durum/<doc_id>.json` (+ UI badges, SSE) |
| Failed-document diagnosis | worker logs: `docker logs med-rag-pipeline-worker-1` |
| Nightly reports | `$CORPUS_HOST_DIR/reports/nightly_YYYY-MM-DD.json` |

The old lineage panel (`src/medrag/api/panel/`) is **out of scope**: the code remains but
is not wired into compose (decision: Open-4). Diagnosis is done via the logs + `durum/`
files + nightly reports above.

## 5. Security hardening (E5)

- **Password**: if `MEDRAG_SIFRE` is empty, auth is off — never leave it empty. The
  session is a signed httpOnly cookie (signed with `MEDRAG_GIZLI_ANAHTAR`; if empty,
  derived from the password).
- **Upload limits**: files outside the extension whitelist
  (`.pdf .docx .pptx .xlsx .html .md`) get 415; per-file limit is `MEDRAG_MAX_YUKLEME_MB`
  (default 200 MB), over that returns 413.
- **Network**: this setup is **for a home network**. Do NOT port-forward from the router;
  if outside access is needed, expose it behind Coolify/Traefik with HTTPS + an extra auth
  layer. `web` listens on 8507 on all interfaces — make sure only the LAN can see it
  (if needed, `ports: "127.0.0.1:8507:8507"` in compose).
- **Secrets**: only in `.env` (gitignored, never committed).

## 6. End-to-end acceptance (F2)

Automated smoke test (against a running stack):

```bash
.venv/bin/python tools/e2e_smoke.py --base-url http://localhost:8507 \
    --sample tests_sample.pdf           # e.g. a small PDF
# if a password is set: --password ...
# full turn including the LLM (slow, spends tokens): --chat
```

Scenario: upload → watch status → ready → view content → add note → search citing the
note → delete document → verify it disappears immediately. With `--chat`: ask a question →
cited answer → click the source (manual step).

Manual acceptance checklist (for the record):

1. [ ] Unauthenticated requests get 401 on `/api/*`
2. [ ] Multi-file drag-and-drop upload works
3. [ ] Status badges advance live (queued → processing → ready)
4. [ ] The answer arrives with inline `[n]` badges; the badge leads to the source
5. [ ] A question with no source shows "I couldn't find that" + clinician note
6. [ ] Create a note → processed online → chat cites the note as a source
7. [ ] Deleting a document removes its library row, chunk and vector traces immediately
8. [ ] Re-uploading a changed version of the same file no longer finds the old content
9. [ ] The backup runs and the §3.1 drill completes successfully
