# config/ -- pipeline model/table profiles

Two files, two distinct purposes:

| profile | used by | when |
|---|---|---|
| `cfg_default.env` | `run_extra_corpus.py` (default) | general/ad-hoc corpus runs -- table description enrichment (context + LLM check/retry) fully on, the best-result combination |
| `cfg_e2e_local.env` | `run_e2e_test.py` (default) | this machine's fixed e2e sanity test -- with the local Ollama models actually installed here (see the file's own header) |

Selection is via `--profile <name>` (without extension) or `PIPELINE_PROFILE`
env; if neither is given the defaults above are used. `run_extra_corpus.py`
also accepts the older `--config <path>` flag (when `--profile` is not given).

```powershell
cd src/medrag/pipeline/cli
python run_extra_corpus.py --profile cfg_default --out-dir out/run1
python run_e2e_test.py --profile cfg_e2e_local
```

## History note

This directory used to hold seven separate comparison files
(cfg1_baseline..cfg7_full) -- a one-off test that parsed the same 3 datasheets
with seven different settings to compare table extraction quality. The
combination that produced the best results (formerly `cfg6_table_describe_full`)
is now the single default `cfg_default.env`; the other six were removed
(details in git history). If a new comparison is ever needed, the old
settings are recoverable from git.

## Notes

- Real environment variables always win over the `.env` file (see
  `llm/__init__.py`). `run_e2e_test.py`'s `build_env()` (reused by
  `run_extra_corpus.py`) gives each run a FRESH `env` dict: it strips all known
  keys out of `os.environ` and then adds the profile's own values -- it never
  modifies the process's own environment.
- Model connection (`LLM_*`/`VLM_*`) and table tuning (`TABLE_*`) both live in
  these profile files by design -- they're a bundled "recipe", not split by
  the usual secret/tuning taxonomy (see root `CONFIG.md`), because the whole
  point of a profile is to vary both together and compare/reuse the bundle.