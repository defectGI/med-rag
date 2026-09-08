"""Parse EVERY document in the extra corpus (chatbot-corpus/extra_corpus)
with the parser -- run_e2e_test.py's machinery reused wholesale (imported,
not copied): same preflight checks, same two-phase VLM-then-LLM order, same
resume behavior, same per-document output layout and summary.

The ONLY difference from run_e2e_test.py is where the sample comes from:
extra-corpus files have no document_nodes.json record, so the sample is
"every parseable file in EXTRA_CORPUS_DIR" (doc_type is reported as EXTRA).
`--limit N` draws N at RANDOM rather than the first N by name (same fix as
run_e2e_test.py's select_datasheets/select_mixed, 2026-07-20 -- a fixed
--limit used to spot-check the exact same files every run); reproducible via
the printed seed / --seed / E2E_SAMPLE_SEED, shared with run_e2e_test.py.
Everything downstream -- run_stage, run_phase, checkpointing, summary.json/md
-- is run_e2e_test.py's own code.

Model set comes from the same single source of truth: a config .env file
(default config/cfg_default.env -- the general-purpose recommended profile,
distinct from run_e2e_test.py's own cfg_e2e_local.env), so preflight always
validates the exact models the run uses. Override with --profile/--config.

Directory is read from EXTRA_CORPUS_DIR in pipeline/.env (falls back to
../chatbot-corpus/extra_corpus relative to this file).

Output (default pipeline/extra_corpus_output/<timestamp>/):
    001_EXTRA_<stem>/
        <original file name>     <- raw copy of the source file
        <doc_id>.stage1.json     <- phase-1 checkpoint
        <doc_id>.json            <- parsed IR
        <doc_id>.md              <- rendered Markdown
        run.log                  <- per-stage stdout/stderr
    ...
    summary.json / summary.md

Resume works exactly like run_e2e_test.py: re-running with the SAME
--out-dir skips completed documents and redoes the interrupted one; the
default timestamped out-dir always starts fresh.

Usage:
    python run_extra_corpus.py                     # all files, full checks
    python run_extra_corpus.py --limit 5           # quick spot-check, random 5
    python run_extra_corpus.py --limit 5 --seed 7  # reproduce a specific sample
    python run_extra_corpus.py --out-dir my_run    # again = resume my_run
    python run_extra_corpus.py --single-pass       # per-document stage order
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from medrag.pipeline.cli import (
    run_e2e_test as e2e,  # loads pipeline/.env at import time
)

BASE_DIR = Path(__file__).resolve().parent

# run_e2e_test.py's own fixed e2e test uses cfg_e2e_local (the local Ollama
# models actually installed on this machine); this script handles random
# corpora for general/ad-hoc runs, so it carries a separate default:
# cfg_default (the single file marked as "best result", replacing the old
# 7-way comparison matrix). --profile/--config/PIPELINE_PROFILE always
# overrides.
DEFAULT_PROFILE = "cfg_default"

EXTRA_CORPUS_DIR = Path(os.environ.get("EXTRA_CORPUS_DIR",
                                       "../../../../chatbot-corpus/extra_corpus"))
if not EXTRA_CORPUS_DIR.is_absolute():
    EXTRA_CORPUS_DIR = (BASE_DIR / EXTRA_CORPUS_DIR).resolve()

# Extensions with a registered reader (parser/parsers/registry.py). Anything
# else in the corpus dir is reported up front and skipped, mirroring
# run_e2e_test.py's exclusion of PRODUCT_IMAGE/STP.
SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
                  ".md", ".markdown"}


def check_dependencies() -> bool:
    """run_e2e_test.check_dependencies minus the document_nodes.json /
    BELGELER_DIR requirements -- this script needs neither, only the corpus
    directory itself."""
    print("[1/4] checking dependencies...")
    ok = True
    for mod, pip_name in e2e.REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"  MISSING: {pip_name}  (pip install {pip_name})")
            ok = False

    to_markdown_py = e2e.PARSER_DIR / "scripts" / "to_markdown.py"
    if not to_markdown_py.is_file():
        print(f"  MISSING: {to_markdown_py}  -- is PARSER_DIR correct? ({e2e.PARSER_DIR})")
        ok = False
    if not EXTRA_CORPUS_DIR.is_dir():
        print(f"  MISSING: {EXTRA_CORPUS_DIR}  -- set EXTRA_CORPUS_DIR in pipeline/.env")
        ok = False

    print("  OK" if ok else "  dependency check FAILED")
    return ok


def build_sample(limit: int, rng: random.Random) -> list[dict]:
    """Every parseable file in EXTRA_CORPUS_DIR, as records shaped like
    document_nodes.json entries so run_e2e_test's run_stage can consume them
    unchanged. rel_path is computed relative to BELGELER_DIR (what run_stage
    joins against), so it contains ".." -- that's fine, the path is resolved
    before use.

    `limit > 0` used to take the first N by name -- a spot-check run with the
    same --limit always re-tested the exact same files (run_e2e_test.py had the
    identical problem, fixed 2026-07-20; see its select_datasheets/select_mixed).
    Sorted-then-shuffled with `rng` instead, so the pre-limit order is a pure
    function of (seed, corpus set) -- same seed + same directory contents =>
    same sample -- and `--limit N` now draws a different N each run unless a
    seed is pinned (see main()'s --seed/E2E_SAMPLE_SEED, shared with
    run_e2e_test.py). `limit == 0` (all files) is unaffected by the shuffle
    other than final ordering, since every file is included either way."""
    files = sorted(p for p in EXTRA_CORPUS_DIR.iterdir() if p.is_file())
    unsupported = [p.name for p in files if p.suffix.lower() not in SUPPORTED_EXTS]
    if unsupported:
        print(f"  skipping {len(unsupported)} unsupported file(s): "
              f"{', '.join(unsupported[:10])}{' ...' if len(unsupported) > 10 else ''}")
    files = [p for p in files if p.suffix.lower() in SUPPORTED_EXTS]
    rng.shuffle(files)
    if limit > 0:
        files = files[:limit]
    return [
        {
            "doc_type": "EXTRA",
            "identity": {"doc_id": p.stem, "file_name": p.name},
            "location": {"rel_path": os.path.relpath(p, e2e.BELGELER_DIR)},
            "is_active": True,
        }
        for p in files
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="parse only N randomly-sampled files (0 = all, the default)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for random sample selection (also E2E_SAMPLE_SEED "
                         "env, shared with run_e2e_test.py). Omit for a fresh "
                         "random sample each run; the chosen seed is printed so "
                         "a run can be reproduced with --seed <n>.")
    ap.add_argument("--workers", type=int, default=1,
                    help="documents processed concurrently (default 1 -- one local "
                         "Ollama instance serves one GPU; see run_e2e_test.py)")
    ap.add_argument("--out-dir", default=None,
                    help="default: extra_corpus_output/<timestamp>/. Re-running "
                         "with an out-dir from an interrupted run RESUMES it")
    ap.add_argument("--profile", default=None,
                    help="config profile name under config/ (e.g. cfg_default); "
                         "shorthand for --config config/<name>.env. Or set "
                         "PIPELINE_PROFILE env.")
    ap.add_argument("--config", default=str(e2e.profile_path(DEFAULT_PROFILE)),
                    help=f"model config .env path (default: config/{DEFAULT_PROFILE}.env "
                         "-- the general-purpose recommended profile). Ignored if "
                         "--profile/PIPELINE_PROFILE is given.")
    ap.add_argument("--skip-checks", action="store_true",
                    help="skip steps [1/4]-[3/4] (dependencies/Ollama/thinking)")
    ap.add_argument("--single-pass", action="store_true",
                    help="run each document start-to-finish (--stage all) instead of "
                         "the phased VLM-then-LLM order")
    args = ap.parse_args()

    # Profile selection (additive): --profile / PIPELINE_PROFILE -> config/<name>.env;
    # otherwise the --config path (legacy behavior). The single resolver lives
    # in run_e2e_test.
    profile = args.profile or os.environ.get("PIPELINE_PROFILE")
    if profile:
        config_path = e2e.profile_path(profile)
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (BASE_DIR / config_path).resolve()
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")

    # Same single-source-of-truth model resolution as run_e2e_test.main().
    cfg = e2e.parse_env_file(config_path)
    models = {"LLM": cfg["LLM_MODEL"], "VLM": cfg.get("VLM_MODEL") or cfg["LLM_MODEL"]}
    thinking = {
        "LLM": (cfg.get("LLM_THINKING_ON", "") in ("1", "true", "yes", "on")),
        "VLM": (cfg.get("VLM_THINKING_ON", cfg.get("LLM_THINKING_ON", ""))
                in ("1", "true", "yes", "on")),
    }
    nominal_models = {
        "LLM": {"model": models["LLM"], "thinking": thinking["LLM"],
                "num_ctx": cfg.get("LLM_NUM_CTX")},
        "VLM": {"model": models["VLM"], "thinking": thinking["VLM"],
                "num_ctx": cfg.get("VLM_NUM_CTX", cfg.get("LLM_NUM_CTX"))},
    }

    if not args.skip_checks:
        if not check_dependencies():
            sys.exit("Dependency check failed -- install the missing packages and retry.")
        if not e2e.check_ollama_models(models):
            sys.exit("Ollama model check failed -- see errors above.")
        if not e2e.check_thinking_disabled(models, thinking):
            sys.exit("Thinking-disabled check failed -- refusing to run the corpus "
                     "against a model still burning its budget on hidden reasoning.")

    seed = e2e.resolve_sample_seed(args.seed)
    rng = random.Random(seed)
    print(f"sample seed: {seed}  (reproduce this exact sample with --seed {seed} "
          f"or E2E_SAMPLE_SEED={seed})")

    print("[4/4] assembling extra-corpus sample...")
    sample = build_sample(args.limit, rng)
    if not sample:
        sys.exit(f"No parseable files found in {EXTRA_CORPUS_DIR}")
    print(f"  {len(sample)} file(s) from {EXTRA_CORPUS_DIR}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else BASE_DIR / "extra_corpus_output"
               / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  # noqa: DTZ005 -- output directory name; local wall-clock is intentional
    if not out_dir.is_absolute():  # subprocess cwd is PARSER_DIR; see run_e2e_test.py
        out_dir = (BASE_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    to_markdown_py = e2e.PARSER_DIR / "scripts" / "to_markdown.py"

    print(f"\nRunning {len(sample)} extra-corpus documents -> {out_dir}\n")

    if args.single_pass:
        print("[single-pass] per-document order (models swap per stage)\n")
        final = e2e.run_phase(sample, out_dir, cfg, to_markdown_py, "all", args.workers)
        results = [final.get(i) or e2e.skipped_result(record, i, out_dir)
                   for i, record in enumerate(sample, 1)]
    else:
        print(f"[phase 1/2] parse + classify + OCR (VLM: {models['VLM']})\n")
        parse_results = e2e.run_phase(sample, out_dir, cfg, to_markdown_py,
                                      "parse", args.workers)
        parse_failed = {i for i, r in parse_results.items() if r["status"] != "PASS"}

        print(f"\n[phase 2/2] OCR checks + table descriptions (LLM: {models['LLM']})\n")
        if not args.skip_checks:
            e2e.eom.warm_model(e2e.eom.ollama_root(e2e.OLLAMA_ROOT), models["LLM"])
        final = e2e.run_phase(sample, out_dir, cfg, to_markdown_py, "finish",
                              args.workers, skip_indices=parse_failed)

        results = []
        for i, record in enumerate(sample, 1):
            if i in parse_failed:
                results.append(parse_results[i])
            elif i in final:
                r = dict(final[i])
                if i in parse_results:
                    r["parse_s"] = parse_results[i]["elapsed_s"]
                    r["finish_s"] = r["elapsed_s"]
                    r["elapsed_s"] = round(r["parse_s"] + r["finish_s"], 1)
                results.append(r)
            else:
                results.append(e2e.skipped_result(record, i, out_dir))

    e2e.write_summary(results, out_dir, nominal_models, title="extra-corpus run")

    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
