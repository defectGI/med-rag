# pipeline (`src/medrag/pipeline/cli/`)

The single file that connects `parser` and `chatbot-corpus`. Neither
imports the other; only this script knows about both.

This folder moved here from repo-root `pipeline/` (`git mv`, now a real
installed package, `medrag.pipeline.cli`). Three scripts did **not** move and
still live at repo-root `pipeline/reset_parse_flags.py`,
`pipeline/reset_visual_cache.py` and `pipeline/mark_docs_inactive.py` -- none
imports anything that moved here, all are standalone escape-hatch CLIs.
Everywhere below that says `python reset_visual_cache.py` or
`python reset_parse_flags.py`, run it from repo-root `pipeline/` instead.
`mark_docs_inactive.py` retires a document editorially (`is_active=false`)
-- e.g. a file that fails parse every single night and would otherwise keep
the nightly report at `partial` forever; scan preserves the flag
(`classify_documents.py::merge_record`), `_needs_parse` skips it, so
`last_parsed` goes stale and the tombstone filter does its job.

## The whole chain in one command (`run_full_corpus.py`)

From a folder of documents to chunks + RAPTOR summaries + fact ledger. It runs
the five stages in dependency order, each as a subprocess in its own directory
(every component resolves its own `.env` relative to itself):

| # | Stage | Script | Does |
|---|-------|--------|------|
| 1 | `products` | `chatbot-corpus/product_info/export_products.py` | Excel -> `product_nodes.json` |
| 2 | `scan` | `chatbot-corpus/document_info/classify_documents.py` | `BELGELER/` -> `document_nodes.json` (registry) |
| 3 | `parse` | `run_parse_pipeline.py` | registry -> IR + each record's `parse` block (VLM/LLM) |
| 4 | `chunk` | `run_chunk_pipeline.py` | IR -> `chunker/storage/{all_chunks,all_raptor,all_combined}.json` + `viz/` + each record's `chunk` block |
| 5 | `facts` | `run_facts_pipeline.py` | chunks -> fact ledger (pass 1-3) + `facts/db/specs.db` + each record's `facts` block |

### Output goes to one timestamped run directory

To avoid having to wipe `parsed/`, `storage/images` and the chunk output by
hand before every full run (they used to accumulate), `run_full_corpus.py`
gathers **all** of a run's output under a single timestamped directory
(`runs/<YYYY-MM-DD_HH-MM-SS>/`):

```
runs/<YYYY-MM-DD_HH-MM-SS>/
  parsed/              IR + <doc_id>.md + raw copy   (PARSED_OUTPUT_DIR)
  images/              crop/blob store                (STORAGE_IMAGES_DIR + PDF_CROP_DIR)
  chunks/              all_chunks.json, all_raptor.json (RAPTOR on), all_combined.json, viz/
  document_nodes.json  registry snapshot of THIS run  (copy)
  product_nodes.json   product tree snapshot          (copy)
  run_manifest.json    ts, git commit, PARSER_VERSION, RAPTOR mode, stage timings, shared labels dir
```

It does this by injecting timestamped **absolute** paths into each stage's
subprocess `env`; `python-dotenv` never overrides an already-set env var
(see `parser/storage_paths.py`), so these win over each stage's own `.env`
without touching any stage's code. `runs/` is gitignored; clean up old runs
by deleting the folders you no longer need (one folder per run).

**Two things deliberately stay shared** (not copied into the run dir):

- **The VLM cache** (`STORAGE_LABELS_DIR`, default `parser/storage/labels/` --
  both the classify and describe caches) is *not* redirected. It's keyed by
  crop-sha + prompt version and survives IR/parser version bumps, so a full
  run never re-pays the expensive vision calls. When you improve visual
  quality **without** bumping a prompt version, clear it on purpose:
  `python reset_visual_cache.py` (dry-run by default; `--yes` to delete).
- **The canonical `document_nodes.json`** is updated in place (doc_id
  persistence); only a *copy* of its post-run state lands in the run dir.

The chunk stage writes exactly three files, not per-document ones -- so
stage 4 has **no staleness gate** anymore; every run reprocesses the whole
corpus (and re-runs RAPTOR LLM calls when `real`), trading incremental cost
for output simplicity. Stages 1-3 keep their own staleness gates unaffected.

```
python run_full_corpus.py                # the whole corpus
python run_full_corpus.py --limit 3      # trial run: parse only 3 documents
python run_full_corpus.py --from parse   # registry is ready, continue from parse
python run_full_corpus.py --dry-run      # print the plan + preflight, run nothing
```

`--limit N` reaches **only the parse stage** -- the expensive one. The scan
stages always see the whole corpus and must: the registry is the source of
truth for what the corpus contains, and a truncated scan would push
previously-seen files to `DELETED`. N documents get parsed, the rest stay
`PENDING` for a later run; stage 4 then chunks whatever IR is present (the
whole corpus, every run -- chunking has no staleness gate).

The run stops at the first failing stage (a later stage would only run on
missing input), and the preflight refuses to start when a stage's input isn't
there yet -- so a long parse never ends in "the chunk stage had RAPTOR off
anyway". Stages 3-4 need real models; on this machine no LLM runs, they
belong on the GPU box.

For any LLM/VLM/VLM_CLASSIFY role that resolves to a local Ollama target (in
`parser/.env` for stage 3, `chunker/.env` for stage 4 when `CHUNKER_RAPTOR=real`),
the preflight also does the same live "thinking is really off" smoke call
`run_e2e_test.py` does (`ensure_ollama_models.check_thinking_disabled` --
shared, not duplicated): a thinking-capable model (e.g. a Nemotron/Qwen3
family model) that ignores the toggle leaks a `<think>` block or returns
empty content, which would otherwise silently ruin every OCR/table-description
call for the whole run before anyone noticed. Cloud/openrouter roles are
skipped -- the check is Ollama-specific.

## What it does

For every active document in `document_nodes.json` that hasn't yet been
parsed for its current `content_hash`, runs the corpus through THREE PHASES
by model affinity (so a single-GPU Ollama swaps models once per run instead
of interleaving VLM_CLASSIFY/VLM/LLM calls throughout -- see
`run_parse_pipeline.py`'s own module docstring for the full reasoning):

1. **Phase 0/3 classify** -- `VLM_CLASSIFY`-only pass over every document's
   visual regions (`phase0_classify_one`), warming `visual_classify.py`'s
   on-disk cache. Skipped with a note when `VLM_CLASSIFY` isn't configured
   distinctly from `VLM`.
2. **Phase 1/3 parse+ocr** -- `parser_for(path).parse()` + image OCR +
   `describe_blocks(stage="vlm")` (charts/diagrams/drawings), writing the IR
   JSON to `PARSED_OUTPUT_DIR`.
3. **Phase 2/3 check+tables** -- deferred LLM OCR checks + `describe_blocks
   (stage="llm")` for tables.

The result (`status`, `stages`, `parsed_json_path`, ...) is written into the
document's `parse` block and `document_nodes.json` is updated.

An extension the parser doesn't support (`.stp`, `.png`, ...) is not an
error, it's counted as `SKIPPED`.

## Setup

```
pip install -r requirements.txt
pip install -r ../parser/requirements.txt
```

Copy `.env.example` to `.env` and fill in the paths. LLM/VLM model settings
don't live here -- they stay in `parser/.env`; the parser knows from its own
settings which model server (local Ollama/vLLM, hosted API, ...) to talk to,
this script just loads it.

## Preparing Ollama models (optional)

If any of the `LLM_*`/`VLM_*`/`VLM2_*` roles in `parser/.env` point to a
local Ollama server (`PROVIDER=ollama`, or `BASE_URL`'s port is `11434`),
run this before running the pipeline so the models get pulled and loaded
into memory:

```
python ensure_ollama_models.py
```

It reads the model name from `.env` as-is (not hardcoded) and talks to
Ollama only through its HTTP API, so it works regardless of the installed
Ollama version. If the server is already running it leaves it alone; it
only starts `ollama serve` itself if nothing is listening on localhost. All
roles currently point to the cloud (openrouter), so this step is not
needed right now.

## Running

```
python run_parse_pipeline.py              # every pending document
python run_parse_pipeline.py --limit 3    # a 3-document trial run
```

`document_nodes.json` is written to disk (atomic replace) after every
document -- if a long batch is interrupted midway, no work is lost and it
resumes where it left off (records where `parsed_from_hash == content_hash`
are not reprocessed, unless the parser version moved on -- see below).

`--limit N` parses the first N documents that actually need work and leaves
the rest `PENDING`; the registry keeps all its records either way. It's a run
control, not a setting, which is why it's a flag rather than an env var
(CONFIG.md's taxonomy). Use it to sanity-check a config change on a few
documents before handing the whole corpus to the GPU.

A record is re-parsed when its bytes changed OR when its `parse.parser_version`
is older than the current `PARSER_VERSION` (`parser/parsers/base.py`) -- so a
parser fix reaches the corpus on its own, without `reset_parse_flags.py`.

## End-to-end test (`run_e2e_test.py`)

A fixed, repeatable end-to-end run over **real** corpus documents with fully
**local** Ollama models -- for checking the whole `parser` + `chatbot-corpus`
chain works, without touching `document_nodes.json`. Unlike
`run_parse_pipeline.py` (which processes the whole corpus and writes results
back), this takes a small fixed sample and lays the output out for eyeballing;
it runs a single fixed config (no config-matrix comparison mode exists).

**Sample:** 10 datasheets + 10 "mixed" documents (brochure / catalogue /
CE declaration / technical drawing / user manual / quick-start, round-robined so
the mix spans as many types as the corpus has). Picked **randomly** each run (a
fresh seed by default, so repeated runs cover more of the corpus instead of
always testing the same files) -- still reproducible: the run prints its seed,
and `--seed <n>` (or `E2E_SAMPLE_SEED`) replays the exact same sample.
`run_extra_corpus.py`'s `--limit N` sampling follows the identical
random+seed pattern. `.png/.jpg/.stp` are excluded (the parser has no reader
for them -- they'd just be guaranteed SKIPs).

**Models** live in a config **profile** under `config/` -- default
`cfg_e2e_local.env` (the single source of truth: the script reads the model set
and thinking toggles straight from it). Shipped default: `LLM=qwen2.5:32b`,
`VLM=qwen2.5vl:7b`, no VLM2. Both are non-reasoning Qwen2.5 models chosen on
purpose -- see that file's header for why `qwen3-vl:8b` was rejected (its
thinking can't be turned off on Ollama 0.31.1, which empties OCR/table output).
To try different models or flip thinking on, edit that one file -- nothing else.
Select a different profile with `--profile <name>` (or `PIPELINE_PROFILE` env),
e.g. `--profile cfg_default`; see `config/README.md`.

**Pipeline's own tuning** (`DOC_CONCURRENCY`, `HEALTH_CHECK`,
`HEALTH_FAIL_STREAK`) has its defaults + docs in `pipeline_config.toml`
(committed); override any per-run via the matching env var (env wins). Model
connection stays in `parser/.env`. See root `CONFIG.md` for the taxonomy.

**Preflight (aborts the run if any fails):**
1. Python dependencies present.
2. Ollama reachable + both models pulled and warmed (pulls them if missing).
3. **Thinking really is off** -- a live smoke call through each model, not a
   trust-the-flag check. If a model ignores the toggle and returns empty/`<think>`
   content, the run aborts here instead of producing 20 broken outputs.
4. The sample can actually be assembled from files on disk.

**Steps to run** (on the machine that has Ollama + the models):

```
# from the src/medrag/pipeline/cli/ folder
pip install -r requirements.txt
pip install -r ../parser/requirements.txt

# make sure this folder's .env points at the parser repo and the corpus
# (PARSER_DIR / BELGELER_DIR / DOCUMENT_NODES_PATH) -- copy .env.example if needed

python run_e2e_test.py
```

That's it -- it pulls/warms the models itself if they aren't ready. Useful flags:

```
python run_e2e_test.py --n-datasheets 3 --n-mixed 3   # smaller sample
python run_e2e_test.py --skip-checks                  # re-run, skip preflight 1-3
python run_e2e_test.py --workers 2                    # parallel docs (default 1)
python run_e2e_test.py --out-dir my_run               # custom output folder
```

**Output** -- raw input, rendered Markdown, and IR JSON kept **together** per
document (`e2e_test_output/<timestamp>/` by default):

```
01_DATASHEET_<stem>/
    <original file name>     <- raw copy of the source
    <doc_id>.json            <- parsed IR
    <doc_id>.md              <- rendered Markdown
    run.log                  <- that document's stdout/stderr
...
summary.json / summary.md    <- per-document status + totals (PASS/FAIL, block/
                                table/image counts, time)
```

Exit code is non-zero if any document failed, so it doubles as a CI check.

> **To run arbitrary files (not the corpus) with your own models and thinking
> on/off**, use `parser/scripts/to_markdown.py` directly -- see the "Quick start"
> section in `../parser/README.md`.

## Chunking (`run_chunk_pipeline.py`)

The chunking counterpart of `run_parse_pipeline.py`: feeds the IR JSONs in
`PARSED_OUTPUT_DIR` (or `CHUNKER_INPUT_DIR`) to the `chunker` package, which
writes `all_chunks.json` (every document, doc_id -> ChunkSet) under
`CHUNKS_OUTPUT_DIR`. chunker's interface is env variables only, so this
script just resolves paths from `.env` and runs `python -m medrag.pipeline.chunker`
(the chunker package is now installed via `pip install -e .`; `CHUNKER_DIR`
is only its DATA home, no longer where the code lives); discovery, error
handling and exit codes (0/1/2) are chunker's own.

RAPTOR (embedding-based summary tree / corpus-profile clustering) was removed
entirely -- no LLM/embedding config exists for chunking any more; `chunker`
is pure offline leaf chunking. `all_combined.json` carries the same
`documents` dict as `all_chunks.json`, wrapped.

After chunker exits, this script writes each record's `chunk` block back into
`document_nodes.json`: status, `chunked_at`, `chunked_from_hash`,
`chunks_path`, `chunker_version` -- copied from that document's own entry
inside `all_chunks.json`'s `provenance`, so the registry and the chunk data
can never disagree about which bytes a chunk set came from. A record with no
entry is reset to `PENDING` rather than left claiming an old SUCCESS. No
`DOCUMENT_NODES_PATH` (or no file there) simply skips this step, so chunking
a loose IR folder still works.

There is no staleness gate here anymore -- `all_chunks.json` is a single file
covering the whole corpus, so every run reprocesses everything from scratch.
This trades the per-document incremental cost for a simpler, fixed two-file
output.

```
pip install -r ../chunker/requirements.txt
python run_chunk_pipeline.py
```

## Facts (`run_facts_pipeline.py`)

The chunk-to-ledger bridge: runs the four passes over `chunker`'s output
(pass 1 chunk->atom, pass 2 atom->key, pass 2.5 vocabulary expansion,
pass 2 again on newly-resolved queue records), reduces the linked pairs into
the fact ledger, sweeps pass 3 (`absent`/`not_specified`), and rebuilds
`facts/db/specs.db` from the ledger + node corpus. `facts` is an in-repo
package (unlike `chunker`, which is subprocessed for its own env-only interface)
-- this script imports it directly.

`FACTS_LLM_MODE=fake` is the default -- no LLM runs on this machine; `real`
switches every pass to its `*_from_env()` factory (`FACTS_LLM_*` in
`../facts/.env`, **UNVERIFIED against a live server**, GPU-machine work).
`FACTS_VOCAB_APPROVE_MODE=interactive` is the default for pass 2.5's
approval gate (a human approves each proposed new `spec_key` on stdin);
`auto`/`skip` exist for scripted/CI runs or when a human isn't available
right now (see `.env.example`).

After the run, this script writes each record's `facts` block back into
`document_nodes.json` -- `status` (`SUCCESS`/`SKIPPED`), `extracted_at`,
`extracted_from_hash`, `extractor_version`, `prompt_version` -- the same
registry `run_chunk_pipeline.py` writes its `chunk` block into.

```
python run_facts_pipeline.py
```

**`run_facts_pass1_parallel.py`** is a separate, standalone speed/resilience
tool for pass 1 on a GPU machine -- not part of the official chain above
(own docstring says so explicitly). Documented here so it doesn't read as
an unmentioned/orphaned script; see its own docstring for usage.

## Chunker integration test (`run_chunk_test.py`)

The chunk counterpart of `run_e2e_test.py`, but fully offline and fast (no
Ollama, no LLM/VLM): takes real parser IR (default: the newest
`e2e_test_output/` run, or `CHUNK_TEST_INPUT_DIR`), runs chunker over it, and
validates the results -- every IR doc has an entry inside `all_chunks.json`,
the entry round-trips through `ChunkSet.model_validate()` (schema version
gate + leaf contract), its provenance points at the IR sitting next to it
(`source.raw_sha256` match -- proof the chunks aren't left over from an older
IR), no empty leaf text, and the prev/next leaf chain is unbroken. Token
stats, over-limit counts and unresolved cross-refs are reported as info (PDF
cross-refs are expected to stay unresolved).

It does not duplicate chunker's unit suite (`chunker/tests/`, run with
pytest from `chunker/`); this checks the integration boundary with real IR.

```
python run_chunk_test.py
```

Output goes to `chunk_test_output/<timestamp>/` (`all_chunks.json`,
`chunker_run.log`, `summary.json` / `summary.md`). Exit code is non-zero on
any failure, so it doubles as a CI check.