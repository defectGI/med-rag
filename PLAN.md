# med-rag — Development Plan

> Status: IMPLEMENTATION COMPLETE (groups A, B, C, D + E; F in acceptance).
> Full suite at baseline: 2105 passed / 61 pre-existing failures (fork baseline) / zero
> regressions; frontend `npm run build` green; ruff clean on new code. B5 (chat history +
> new chat) complete. Group E: compose (web + pipeline-worker + pipeline + qdrant),
> read-write volume scheme, nightly backup + restore procedure (`DEPLOY.md` §3.1),
> observability and auth/upload hardening complete. F: the end-to-end smoke test
> `tools/e2e_smoke.py` is ready; the live run happens after first deploy. User manual:
> `USAGE.md`.
>
> Note (specs.db): by repo policy `facts/db/specs.db` is deliberately not tracked (facts is
> dormant, DATA ballast); some dormant-path tests (doc_download/comparison/factory) expect
> this file to exist. On a clean checkout those tests fail unless the file is produced by the
> facts pipeline (LLM inference — needs a network/key); the remaining active paths
> (default_topn, library, status) are unaffected.

---

## 1. Vision

A self-growing document assistant that answers from the clinical documents a physician
uploads — digital PDF, scanned/photo PDF and other office formats — **showing evidence**.
The system grows as files are added, cleans up derivations when a file is deleted, and drops
old information when a file changes. The UI is a modern SPA built on shadcn/ui with the Wada
Sanzo ivory + sun palette (light/dark themes).

### Non-goals

- WhatsApp / external messaging integration (already removed in the fork)
- Multi-user support, role management
- Activating the facts/specs.db + text2sql path (the code is dormant, unused)
- Native mobile app (a responsive web is enough)

## 2. Decisions taken (answers to the open questions)

| # | Topic | Decision |
|---|---|---|
| K1 | Install | Deploy on the user's home machine with **Coolify** |
| K2 | File input | **Upload from the UI** (drag-and-drop / multi-select); NO watched folder |
| K3 | User | Single user + **simple password** (login screen, session cookie) |
| K4 | Language | **Fully mixed** TR/EN documents + questions; answers in the question's language |
| K5 | Scale | **Large: 500+ documents** — incremental processing, cost control and status tracking are critical |
| K6 | File types | **All that the parser supports** (PDF, DOCX, PPTX, XLSX, HTML, MD) |
| K7 | LLM | Keep the **provider-agnostic** architecture; **cloud models** in use |
| K8 | Frontend | **Vite + React + TypeScript + Tailwind + shadcn/ui**; backend stays Flask API |
| K9 | Theme | **Light + dark** toggle; CSS-variable based |
| K10 | Palette | Wada Sanzo: **ivory ground + sun tones** (yellow/orange/red accents) |
| K11 | UI scope | **Library + chat + notes + document navigation** (detailed below) |
| K12 | Notes | User-created/edited **text notes enter the corpus too**; the chatbot cites them as sources |
| K13 | Document review | **Rendered, user-friendly Markdown** (not raw; Claude-artifacts feel) + "view original" for the PDF |
| K14 | Evidence | **Both inline badges and a collected source list**; citation mandatory |
| K15 | Conflict/safety | Conflicting sources **both shown + warning**; if not found **"I couldn't find that"**; short **"consult a clinician"** note under clinical answers |
| K16 | Processing status | Per-file **status badge** in the library (queued → processing → ready/error), page-based progress |
| K17 | Backup | **Automatic nightly**: corpus + notes + Qdrant snapshot (original backup logic adapted) |

## 3. Architecture (target)

```
┌───────────────────────────── Home machine / Coolify ─────────────────────────────┐
│                                                                                 │
│  ┌───────────────┐   REST + SSE    ┌──────────────┐   job queue     ┌─────────┐ │
│  │ web (SPA)     │ ◄─────────────► │ api (Flask)  │ ──────────────► │ pipeline│ │
│  │ Vite+React    │                 │ auth, upload │   (simple file- │ worker  │ │
│  │ shadcn/ui     │                 │ status, chat │   based queue)  │ parse→  │ │
│  └───────────────┘                 └──────┬───────┘                 │ chunk→  │ │
│                                           │                         │ vector  │ │
│                                    ┌──────▼──────┐                  └────┬────┘ │
│                                    │   qdrant    │◄─────────────────────┘      │
│                                    └─────────────┘                             │
│  /corpus (single unit): BELGELER/ notes/ parsed/ chunks/ vectorize/ yedekler/    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Principles:

1. **Event-driven registry**: `document_nodes.json` is no longer produced by a folder
   scan but written by **UI events** (upload/delete/edit-note). The existing
   `classify_documents.py` (scan) stays as a one-off reconciliation tool; it is NOT
   required in the nightly chain.
2. **File lifecycle = single source of truth**: upload → (parse → chunk → vectorize);
   change → delete old derivations + reprocess; delete → remove all derivations. These
   three are gathered into a single "document processor" abstraction.
3. **Frontend/API separation**: Flask serves only JSON + SSE; the template UI (the HTML
   side in webapp.py) is phased out gradually.
4. **Cost awareness**: at 500+ document scale each upload is processed individually; no
   code path falls back to reprocessing "the whole corpus" by default.

## 4. Task groups

### Group A — Document lifecycle and pipeline triggering (core)

| ID | Task | Acceptance criterion |
|----|------|---------------|
| A1 | **Upload API**: `POST /api/documents` (multi-file, type/size check, name-collision resolution). The file is written to `/corpus/BELGELER/`, a `NEW` record is added to the registry, and it enters the job queue. | Multi-upload; invalid type rejected; same-name `-1` derivation; registry consistent |
| A2 | **Single-file pipeline runner**: run the existing parse→chunk→vectorize stages over one `doc_id` (using the existing `--doc`/incremental gates). Page-based progress is written to state for VLM-heavy scanned PDFs. | A single-file upload does not process the others; progress readable per page % |
| A3 | **Delete**: `DELETE /api/documents/{id}` → adapt `forget_deleted_source`: clean the parsed folder, chunks, Qdrant points and the registry record. | No trace of the deleted file (chunk/vector/library row) remains |
| A4 | **Change**: re-upload the same rel_path (different content_hash) → old derivations deleted (A3 path), new processing (A2) triggered. Semantics: "old version deleted + new version added". | Chat cannot answer from old chunks of a changed file; new content is found |
| A5 | **Status store + API**: per-file `queued/parsing(%)/chunking/vectorizing/ready/error(+reason)`; `GET /api/documents` and `GET /api/documents/{id}/status` + SSE stream. | Badge data readable from the API; human-readable reason on error |
| A6 | **Job queue**: a redis-less simple mechanism — a **bounded thread pool inside the api** (suggestion: max 2 concurrent, serializing around the single VLM). Cancel support on long scanned PDFs. | Two concurrent uploads are serialized; on api restart "queued" jobs are re-discovered (interruption recovery) |
| A7 | **Notes → corpus**: the `notlar/` directory hooks into the same lifecycle for text notes (doc_type=NOTE). On note save, old chunks are deleted and new ones written to vectors. | After editing a note, chat cites the note's new version |

### Group B — Frontend (Vite + React + shadcn/ui)

| ID | Task | Acceptance criterion |
|----|------|---------------|
| B1 | **Scaffold**: Vite + React + TS + Tailwind + shadcn/ui; folder layout `web/`; Flask serves it statically under `/` (single origin, no CORS) | SPA opens with `docker compose up` |
| B2 | **Theme system**: Wada Sanzo palette via CSS variables (group E), light/dark toggle (localStorage + `prefers-color-scheme`) | WCAG AA contrast in both themes |
| B3 | **Login screen** (K3): single password, httpOnly session cookie, end-to-end auth middleware | Every password-less API returns 401 |
| B4 | **Library view**: document list (name, type, date, size, status badge), drag-and-drop multi-upload, delete (confirmed), error detail, search/filter | K16 badges update live (SSE) |
| B5 | **Chat view**: message flow, streaming answer (SSE), chat-history persistence, new chat | History survives a page refresh; the stream is live up to the evidence |
| B6 | **Evidence components** (K14/K15): inline `[n]` badges, numbered source list after the answer (document + page + section), badge click scrolls to the source, conflict warning box, "consult a clinician" footnote | No citation-free clinical answer; warning visible in a conflict example |
| B7 | **Document viewer** (K13): `react-markdown` + typography/pretty-render (tables, code, lists, heading hierarchy), page-reference anchors, "view original" (pdf.js / browser-embedded viewer) | A scanned book section renders readably; evidence click goes to the relevant section |
| B8 | **Notes module**: note list, create/edit/delete (simple text editor), save → triggers A7, "included in chat" status visible | A note edit is vectorized within a roughly online process |
| B9 | **API client + types**: type-safe fetch layer, SSE helpers, error/toast standards | Single place for API versioning |

### Group C — Evidence & prompt layer (clinical adaptation)

| ID | Task | Acceptance criterion |
|----|------|---------------|
| C1 | **doc_question strategy rewrite**: citation is no longer optional — every factual claim carries a mandatory `doc_id + page + section` attribution | Strategy prompt and the relevant templates updated; sample answers obey the rule |
| C2 | **Conflict policy**: for chunks that give different values of the same fact, show both + a "sources report different information" warning; the model cannot invent a single value | Tested on a synthetic conflict corpus |
| C3 | **Not found + disclaimer**: when there is no source, invention is forbidden ("I couldn't find that" + suggestion); a short clinician-consult note at the end of every clinical answer (K15) | A fake answer without a disclaimer cannot be produced on empty retrieval |
| C4 | **Language policy** (K4): answer in the question's language; on mixed TR/EN chunks keep term fidelity (English medical term + Turkish explanation) | TR question → TR answer; EN source quotation preserved |
| C5 | **Router simplification**: nearly all intents go to `default_topn`; unused strategy prompts are archived (not deleted) | Router table reviewed in config |

### Group D — Design system (Wada Sanzo: ivory + sun)

| ID | Task | Acceptance criterion |
|----|------|---------------|
| D1 | **Palette tokens**: ivory-ground family from Wada Sanzo (e.g. the Gofun/Torinoko cream line) + sun accents (Yamabuki yellow, Kaki orange, dark red); shadcn CSS variables (primary/secondary/accent/destructive/muted/ring) | Palette table listed with hex values in PLAN; applied to the shadcn theme |
| D2 | **Dark theme mapping**: dark-ground counterparts of the same hues (Sumi ink ground + warm accents) | The accents carry the same identity in dark mode |
| D3 | **Component polish**: status badges, evidence boxes, toast, empty-state screens aligned with the palette | All pages share one design language |

> D1 palette sketch (hexes calibrated during implementation):
> `--background` ivory `#FAF6EE` · `--foreground` ink `#2B2926` ·
> `--primary` Yamabuki `#E8A020` · `--accent` Kaki `#D96C2C` ·
> `--secondary` neutral tea-green equivalent `#8C8473` · `--destructive` red `#B3352C`.
> Dark: ground `#201D1A`, card `#2A2622`, same accents in lighter tones.

### Group E — Deploy & operations

| ID | Task | Acceptance criterion |
|----|------|---------------|
| E1 | **Compose update**: `web` (SPA static + nginx or Flask static), `api`, `pipeline-worker` (if A6 threads are in the api it merges with api — decision point), `qdrant`. Coolify-compatible env/volume definitions | `docker compose up -d` works from scratch |
| E2 | **Volume scheme**: `/corpus` single-writer layout updated for the upload API (api is now a WRITER — review the `:ro` rule); permissions compatible with uid 1000 | After upload, pipeline and api see the same file |
| E3 | **Nightly backup** (K17): corpus + notes + Qdrant snapshot → `/corpus/yedekler/`; adapt the existing `nightly_backup` logic; write the restore procedure | Restore-from-backup drill succeeds |
| E4 | **Observability**: upload/processing events logged; the panel (lineage) is adapted to the med-rag flow or dropped from scope (decision point) | A failing document is diagnosable from logs |
| E5 | **Security hardening**: single password + session; upload size limit; extension whitelist; notes on not exposing outside the home network (Coolify/reverse proxy) | Unauthenticated access and large-file DoS are closed |

### Group F — Quality & acceptance

| ID | Task | Acceptance criterion |
|----|------|---------------|
| F1 | **Existing offline suite kept**: no NEW failures beyond the baseline 61 pre-existing ones; import-linter contracts up to date | `pytest src/ tools/ -q` stays at the fork baseline |
| F2 | **End-to-end acceptance scenario**: upload → watch status → ready → ask → cited answer → click source → document viewer opens → add note → note-sourced answer → delete file → answer disappears immediately | The whole scenario is run manually and recorded |
| F3 | **Change/delete tests**: automated tests of A3/A4 on a synthetic corpus | Automated tests green |
| F4 | **Parser quality measurement**: `tools/benchmark` scores a scanned-PDF sample (scope file focused on medical tables/dose lines) | Table integrity verified on dose-table examples |
| F5 | **Documentation**: README + CONFIG.md updated to the new architecture; a 1-page user manual | New users self-serve from install to use |

## 5. Suggested implementation order

1. **A1→A6** (document lifecycle + status) — carries the core value
2. **C1→C5** (evidence layer) — early win since the chatbot already exists
3. **B1+B2+D1→D3** (SPA skeleton + theme) → **B4** (library) → **B5+B6** (chat+evidence)
4. **B7** (viewer) → **B8** (notes, together with A7)
5. **E1→E5** (deploy/backup hardening) → **F2** (end-to-end acceptance)

## 6. Open decision points (clarified during implementation)

| # | Topic | DECISION |
|---|------|-------|
| Open-1 | Job-queue location | **Separate `pipeline-worker` service**: file-based queue (`/corpus/isler`), single replica, sequential run; the api only WRITES the job file (no import). Interruption recovery via worker restart. |
| Open-2 | Scan/nightly role | Lifecycle is **event-driven** (UI events write); `classify_documents.py` stays as a reconciliation tool, NOT required in the chain. |
| Open-3 | Embedding model | The model in the current config is **kept**. If a change is wanted: delete and recreate the collection (three sides aligned) — procedure in `DEPLOY.md` §1. |
| Open-4 | Panel (lineage) fate | **Out of scope**: the code remains (dormant like facts), not wired to compose; diagnosis uses logs + `durum/` + nightly reports (`DEPLOY.md` §4). |
| Open-5 | Upload upper limits | **200 MB** per file (overridable via `MEDRAG_MAX_YUKLEME_MB`), extension whitelist; file count per request unlimited (single user). |
| Open-6 | Chat-history storage | The `conversation_log` is **persistent** (`/logs/conversations` volume); read endpoint `/api/chat/history`, tied to the session cookie; the user leaves with "new chat", old records stay on disk. |

## 7. Traceability: expectation → task

| User expectation | Task |
|----------------------|------|
| "corpus grows as files are added, pipeline triggers" | A1, A2, A6, B4 |
| "deleting a file removes its derivations" | A3 |
| "changed file = delete old + add new" | A4 |
| "wada sanzo colors, shadcn, colorful combination" | D1–D3, B2 |
| "library + chat + notes + document navigation" | B4, B5, B7, B8, A7 |
| "user-friendly markdown render" | B7 |
| "evidence critical in every answer" | C1–C3, B6 |
| "simple password" | B3, E5 |
| "deploy on a home machine with coolify" | E1, E2 |
| "nightly backup" | E3 |
