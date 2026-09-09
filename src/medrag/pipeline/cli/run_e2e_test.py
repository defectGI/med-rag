"""End-to-end local test: `parser` over real `chatbot-corpus` documents,
fully local Ollama models, single run (like run_parse_pipeline.py, but a
fixed small sample with grouped, inspectable output instead of writing back
into document_nodes.json -- and this runs ONE selected profile, not a matrix;
to sweep several profiles, loop run_extra_corpus.py --profile, see config/README).

Model set is defined ONLY in config/cfg_e2e_local.env (this script reads
LLM_MODEL/VLM_MODEL/*_THINKING_ON from there, so preflight always checks the
exact models the run uses). The shipped default:
    LLM  : qwen2.5:32b  (text -- table description, OCR check)
    VLM  : qwen2.5vl:7b (vision -- OCR, hybrid/scanned pages, table grid)
    VLM2 : unset        (no consensus verification, single-model run)
Both are Qwen2.5 models with NO thinking/reasoning mode -- chosen on purpose:
qwen3-vl:8b (the literal "8b" match to what was originally asked for) was
tried first and rejected because step [3/4] below proved its hidden thinking
cannot be turned off on Ollama 0.31.1 by any known parameter
(reasoning_effort="none", reasoning.enabled=false, native "think": false --
all tried directly against the server, all still produced a full <think>
block first), which starves `content` under any realistic token budget
(exactly the failure describe/core.py's own warning describes).
qwen2.5* have no thinking mode at all, so that failure mode doesn't apply.

Sample: N datasheets + N non-datasheet ("mixed") documents, picked at RANDOM
from chatbot-corpus/document_info/document_nodes.json. The selection used to be
the first N by rel_path, which meant "10 datasheets" re-tested the exact same 10
files every run and narrowed the effective sample; now each run draws a fresh
random subset so coverage widens across runs. It is still reproducible: the run
prints its seed and `--seed <n>` (or E2E_SAMPLE_SEED) replays the exact same
sample. Datasheets are the single largest doc_type; "mixed" round-robins across
every other parseable doc_type (BROCHURE, CATALOGUE, CE_DECLARATION,
TECHNICAL_DRAWING, USER_MANUAL, QUICK_START_GUIDE) -- randomly WITHIN each type
and with a shuffled type order -- so the files span as many types as the corpus
actually has while still varying which ones are picked. PRODUCT_IMAGE/STP are
excluded -- the parser has no reader for .png/.jpg/.stp (see
parsers/registry.py), so they'd just be a guaranteed SKIP, not a real parse
test.

Before any document is touched, four checks run in order and the whole run
aborts (non-zero exit) if any of them fails:
    [1/4] Python dependencies (parser/requirements.txt's actual imports)
    [2/4] Ollama reachable + both models pulled and warmed
    [3/4] thinking really is off (live smoke call, see above)
    [4/4] the sample can actually be assembled (enough files exist on disk)

Documents run in TWO PHASES by model affinity (to_markdown.py --stage): phase 1
runs the VLM-only work (parse -- including visual-region classification -- +
image classify/OCR) for EVERY document, phase 2 the LLM-only work (OCR checks
+ table descriptions) for every document. A single
GPU can't hold qwen2.5:32b and qwen2.5vl:7b at once, so the old per-document
order made Ollama reload a model at every stage boundary (and per image!);
phased, each model cold-starts once per run. --single-pass restores the old
per-document order if the phased path ever needs to be ruled out.

Progress is preserved: phase 1 checkpoints each document atomically to
<doc_id>.stage1.json, and both phases skip documents whose artifacts already
exist. After a crash/Ctrl-C, re-running with the SAME --out-dir resumes right
where it stopped (completed documents are skipped, the interrupted one is
redone); the default timestamped out-dir always starts a fresh run.

Output layout -- raw input, rendered Markdown, and IR JSON kept together per
document (this is what makes a single result easy to eyeball), under
--out-dir (default pipeline/e2e_test_output/<timestamp>/):
    01_DATASHEET_<stem>/
        <original file name>          <- raw copy of the source file
        <doc_id>.stage1.json           <- phase-1 checkpoint (IR + raw OCR;
                                          kept so phase 2 alone can be re-run)
        <doc_id>.json                  <- parsed IR
        <doc_id>.md                    <- rendered Markdown
        run.log                        <- that document's stdout/stderr,
                                          appended per stage
    ...
    summary.json                       <- one record per document + totals
    summary.md                         <- the same, as a human-readable table

Usage:
    python run_e2e_test.py
    python run_e2e_test.py --n-datasheets 5 --n-mixed 5
    python run_e2e_test.py --workers 2 --out-dir my_run
    python run_e2e_test.py --out-dir my_run          # again = resume my_run
    python run_e2e_test.py --single-pass             # old per-document order
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from medrag.pipeline.cli import (
    ensure_ollama_models as eom,
)


def _resolve(env_var: str) -> Path:
    p = Path(os.environ[env_var])
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


PARSER_DIR = _resolve("PARSER_DIR")
BELGELER_DIR = _resolve("BELGELER_DIR")
DOCUMENT_NODES_PATH = _resolve("DOCUMENT_NODES_PATH")
PROFILES_DIR = BASE_DIR / "config"
DEFAULT_PROFILE = "cfg_e2e_local"
# Backward-compatible default (run_extra_corpus imports this). Profile
# selection is additive: --profile NAME or PIPELINE_PROFILE env ->
# config/NAME.env; if neither is given this is used (behavior identical to
# before).
CONFIG_PATH = PROFILES_DIR / f"{DEFAULT_PROFILE}.env"


def profile_path(name: str) -> Path:
    """Path to config/<name>.env (passed as-is if extension/absolute is given)."""
    p = Path(name)
    if p.is_absolute():
        return p
    if name.endswith(".env"):
        return PROFILES_DIR / p.name if p.parent == Path(".") else p
    return PROFILES_DIR / f"{name}.env"


def resolve_profile(profile: str | None) -> Path:
    """Profile selection: --profile arg -> PIPELINE_PROFILE env -> DEFAULT_PROFILE."""
    name = profile or os.environ.get("PIPELINE_PROFILE")
    return profile_path(name) if name else CONFIG_PATH

from medrag.pipeline.parser import (
    progress,
)

# (unlike the to_markdown.py subprocess run_stage() launches), so a bar here
# actually renders; see medrag/pipeline/parser/progress.py.

OLLAMA_ROOT = "http://localhost:11434"
# Read from cfg_e2e_local.env at runtime (below) so the model set is defined
# in exactly ONE place -- the config file -- and preflight can never check a
# different model than the one the run actually uses.

MIXED_DOC_TYPES = [
    "BROCHURE", "CATALOGUE", "CE_DECLARATION", "TECHNICAL_DRAWING",
    "USER_MANUAL", "QUICK_START_GUIDE",
]

# module name -> pip package name, for parser/requirements.txt's hard requirements
REQUIRED_PACKAGES = {
    "openpyxl": "openpyxl", "docx": "python-docx", "pptx": "python-pptx",
    "dotenv": "python-dotenv", "pdfplumber": "pdfplumber", "PIL": "pillow",
}

# Every key that appears in config/cfg_e2e_local.env or that this script sets
# itself -- stripped from the subprocess env before that file's values (and
# STORAGE_OUTPUT_DIR) are applied, so neither pipeline/.env's nor parser/.env's
# real values (e.g. the different VLM2 set for the other machine) can leak
# into the selected-profile run. (build_env() below is the reusable env-isolation
# helper; run_extra_corpus.py reuses it for arbitrary corpora.)
ALL_KNOWN_KEYS = [
    "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_THINKING_ON",
    "VLM_PROVIDER", "VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY", "VLM_THINKING_ON",
    "VLM2_PROVIDER", "VLM2_MODEL", "VLM2_BASE_URL", "VLM2_API_KEY", "VLM2_THINKING_ON",
    "VLM_CLASSIFY_PROVIDER", "VLM_CLASSIFY_MODEL", "VLM_CLASSIFY_BASE_URL",
    "VLM_CLASSIFY_API_KEY", "VLM_CLASSIFY_THINKING_ON", "VLM_CLASSIFY_NUM_CTX",
    "TABLE_STRUCT_PROVIDER", "TABLE_STRUCT_MODEL", "TABLE_STRUCT_BASE_URL",
    "TABLE_STRUCT_API_KEY", "TABLE_LLM_CHECK", "DESCRIBE_CONTEXT", "DESCRIBE_CONTEXT_BEFORE",
    "DESCRIBE_CONTEXT_AFTER", "DESCRIBE_CONTEXT_MAX_CHARS", "TABLE_CHECK_RETRIES",
    "STORAGE_OUTPUT_DIR", "STORAGE_RAW_DIR", "STORAGE_IMAGES_DIR",
]
FORCE_BLANK = ["VLM2_PROVIDER", "VLM2_MODEL", "VLM2_BASE_URL", "VLM2_API_KEY", "VLM2_THINKING_ON"]


# ---------------------------------------------------------------------------
# [1/4] dependency check
# ---------------------------------------------------------------------------


def check_dependencies(config_path: Path = CONFIG_PATH) -> bool:
    print("[1/4] checking dependencies...")
    ok = True
    for mod, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"  MISSING: {pip_name}  (pip install {pip_name})")
            ok = False

    to_markdown_py = PARSER_DIR / "scripts" / "to_markdown.py"
    if not to_markdown_py.is_file():
        print(f"  MISSING: {to_markdown_py}  -- is PARSER_DIR correct? ({PARSER_DIR})")
        ok = False
    if not DOCUMENT_NODES_PATH.is_file():
        print(f"  MISSING: {DOCUMENT_NODES_PATH}")
        ok = False
    if not BELGELER_DIR.is_dir():
        print(f"  MISSING: {BELGELER_DIR}")
        ok = False
    if not config_path.is_file():
        print(f"  MISSING: {config_path}")
        ok = False

    print("  OK" if ok else "  dependency check FAILED")
    return ok


# ---------------------------------------------------------------------------
# [2/4] Ollama model readiness
# ---------------------------------------------------------------------------


def check_ollama_models(models: dict[str, str]) -> bool:
    print("[2/4] checking Ollama models...")
    root = eom.ollama_root(OLLAMA_ROOT)
    if not eom.ensure_server_running(root):
        return False

    ok = True
    for role, model in models.items():
        print(f"  [{role}] {model}")
        if eom.model_present(root, model):
            print("    already present")
        elif not eom.pull_model(root, model):
            ok = False
            continue
        eom.warm_model(root, model)
    return ok


# ---------------------------------------------------------------------------
# [3/4] thinking-really-off smoke test
# ---------------------------------------------------------------------------
# check_thinking_disabled/_chat_once moved to ensure_ollama_models.py
# (2026-07-17) -- run_full_corpus.py's preflight needed the exact same live
# check, so it now lives in the one module both scripts already import
# rather than as two copies.


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------


def load_document_nodes() -> list[dict]:
    with open(DOCUMENT_NODES_PATH, encoding="utf-8") as f:
        return json.load(f)["documents"]


def _exists(record: dict) -> bool:
    return (BELGELER_DIR / record["location"]["rel_path"]).is_file()


def resolve_sample_seed(cli_seed: int | None) -> int:
    """The seed for this run's random sampling. Precedence: --seed, then
    E2E_SAMPLE_SEED, then a fresh random seed. Always an explicit int so the run
    can print it and be reproduced exactly (`--seed <n>`): the sampling is
    random by default -- to widen coverage across re-runs instead of hammering
    the same first-N-sorted documents -- but a failing sample must stay
    reproducible, which a printed seed guarantees."""
    if cli_seed is not None:
        return cli_seed
    env = os.environ.get("E2E_SAMPLE_SEED", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            print(f"  ignoring non-integer E2E_SAMPLE_SEED={env!r}")
    return random.randrange(2**32)


def select_datasheets(docs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """N datasheets picked at random from all available ones (was: the first N
    by rel_path, which re-tested the exact same documents every run and narrowed
    the effective sample). `rng` is seeded (see `resolve_sample_seed`) so the
    choice is reproducible from the printed seed."""
    cands = [d for d in docs
             if d.get("is_active", True) and d.get("doc_type") == "DATASHEET" and _exists(d)]
    # Sort first so the shuffle is a pure function of (seed, corpus set) and not
    # of load_document_nodes()'s incidental ordering -- same seed + same corpus
    # => same sample, regardless of how the registry happened to enumerate.
    cands.sort(key=lambda d: d["location"]["rel_path"])
    rng.shuffle(cands)
    return cands[:n]


def select_mixed(docs: list[dict], n: int, exclude_ids: set[str],
                 rng: random.Random) -> list[dict]:
    """Round-robins across MIXED_DOC_TYPES (one from each type per round) so
    the sample spans as many non-datasheet document types as the corpus
    actually has, instead of just taking the first N of whichever type
    happens to sort first. Within each type the order is RANDOM (seeded), and
    the type order is shuffled per run too, so re-runs vary which documents (and
    which types fill the last partial round) are picked -- while the
    round-robin still guarantees type spread. Reproducible from the seed."""
    by_type: dict[str, list[dict]] = {t: [] for t in MIXED_DOC_TYPES}
    for d in docs:
        t = d.get("doc_type")
        if (t in by_type and d.get("is_active", True)
                and d["identity"]["doc_id"] not in exclude_ids and _exists(d)):
            by_type[t].append(d)
    for bucket in by_type.values():
        # Sort then shuffle: make the sample a pure function of (seed, corpus),
        # independent of the registry's enumeration order (see select_datasheets).
        bucket.sort(key=lambda d: d["location"]["rel_path"])
        rng.shuffle(bucket)
    type_order = list(MIXED_DOC_TYPES)
    rng.shuffle(type_order)

    picked: list[dict] = []
    round_idx = 0
    while len(picked) < n:
        progressed = False
        for t in type_order:
            bucket = by_type[t]
            if round_idx < len(bucket):
                picked.append(bucket[round_idx])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        round_idx += 1
    return picked


def check_sample(datasheets: list[dict], mixed: list[dict], n_datasheets: int, n_mixed: int) -> bool:
    print("[4/4] checking sample availability...")
    ok = True
    if len(datasheets) < n_datasheets:
        print(f"  only {len(datasheets)}/{n_datasheets} datasheets available on disk")
        ok = False
    if len(mixed) < n_mixed:
        print(f"  only {len(mixed)}/{n_mixed} mixed documents available on disk")
        ok = False
    if ok:
        print(f"  {len(datasheets)} datasheets, {len(mixed)} mixed "
              f"({dict(Counter(d['doc_type'] for d in mixed))})")
    return ok


# ---------------------------------------------------------------------------
# per-document run
# ---------------------------------------------------------------------------


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def build_env(cfg: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ALL_KNOWN_KEYS}
    env.update(cfg)
    for k in FORCE_BLANK:
        env.setdefault(k, "")
    env.update(extra)
    return env


def _safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return cleaned or "doc"


def _group_dir(record: dict, index: int, out_dir: Path) -> tuple[Path, str]:
    doc_type = record.get("doc_type", "UNKNOWN")
    doc_id = _safe_stem(record["identity"]["file_name"])
    return out_dir / f"{index:02d}_{doc_type}_{doc_id}", doc_id


def _artifacts(group_dir: Path, doc_id: str) -> dict[str, Path]:
    return {
        "stage1": group_dir / f"{doc_id}.stage1.json",
        "json": group_dir / f"{doc_id}.json",
        "md": group_dir / f"{doc_id}.md",
    }


def _stage_done(record: dict, index: int, out_dir: Path, stage: str) -> bool:
    """Resume check: is this document's `stage` already complete in out_dir?
    Trustworthy on existence alone -- the stage1 checkpoint is written
    atomically (tmp + os.replace) and the final .json/.md pair only lands at
    the very end of the finish/all stage, so a killed run can't leave a
    convincing-looking torn artifact behind."""
    group_dir, doc_id = _group_dir(record, index, out_dir)
    art = _artifacts(group_dir, doc_id)
    final_done = art["json"].is_file() and art["md"].is_file()
    if stage == "parse":
        return art["stage1"].is_file() or final_done  # final output implies stage1's work
    return final_done  # "finish" / "all"


_req_lock = threading.Lock()
_req_by_child: dict[int, int] = {}  # id(child's bars dict) -> its last @@REQ count


def _ipc_req_event(line: str, bars: dict[str, object]) -> None:
    """@@REQ <n>: one child's current in-flight model request count (see
    parser/progress.py). Each concurrent run_stage() child reports its own
    number; the terminal gauge shows the sum over the children still alive.
    The bars dict is unique per child, so its id doubles as the child key."""
    try:
        n = int(line.split()[1])
    except (IndexError, ValueError):
        return
    with _req_lock:
        _req_by_child[id(bars)] = n
        total = sum(_req_by_child.values())
    progress.set_inflight(total)


def _ipc_req_clear(bars: dict[str, object]) -> None:
    """Drop a finished child's contribution so the gauge can't get stuck on
    a count from a process that no longer exists."""
    with _req_lock:
        if id(bars) not in _req_by_child:
            return  # child never reported -- don't touch the gauge
        _req_by_child.pop(id(bars))
        total = sum(_req_by_child.values())
    progress.set_inflight(total)


def _ipc_bar_event(line: str, bars: dict[str, object]) -> None:
    """One @@BAR line from a child parser (parser/progress.py's _IpcBar):
    open/advance/close the matching bar on THIS process's terminal. The child
    already prefixes descs with its file name, so bars from concurrent
    workers stay tellable apart. Malformed lines are dropped -- a garbled
    progress event must never kill the run."""
    try:
        parts = line.rstrip("\n").split(" ", 5)
        op, bar_id = parts[1], parts[2]
        if op == "OPEN":
            bars[bar_id] = progress.bar(int(parts[3]), parts[5], unit=parts[4])
        elif op == "UPD" and bar_id in bars:
            bars[bar_id].update(int(parts[3]))
        elif op == "CLOSE":
            b = bars.pop(bar_id, None)
            if b is not None:
                b.close()
    except (IndexError, ValueError):
        pass


def run_stage(record: dict, index: int, out_dir: Path, cfg: dict[str, str],
              to_markdown_py: Path, stage: str) -> dict:
    """Run ONE stage of one document as a subprocess. Never raises -- a
    subprocess failure lands in the returned record's status/exit_code,
    mirroring run_parse_pipeline.py's process_one()."""
    doc_type = record.get("doc_type", "UNKNOWN")
    file_name = record["identity"]["file_name"]
    src = BELGELER_DIR / record["location"]["rel_path"]
    group_dir, doc_id = _group_dir(record, index, out_dir)
    group_dir.mkdir(parents=True, exist_ok=True)

    raw_copy = group_dir / file_name
    if not raw_copy.is_file():  # copy once; the finish stage reuses it
        shutil.copy2(src, raw_copy)

    # PROGRESS_IPC=1: the child's stdout is a pipe (no TTY), so its own tqdm
    # bars can't render -- instead parser/progress.py emits @@BAR protocol
    # lines, and streaming the pipe here lets THIS process (whose stderr is
    # the real terminal) draw the child's page/image/table bars live.
    env = build_env(cfg, {"STORAGE_OUTPUT_DIR": str(group_dir),
                          "PROGRESS_IPC": "1"})
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(to_markdown_py), str(src.resolve()), doc_id,
         "--stage", stage],
        cwd=PARSER_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", bufsize=1,
    )
    bars: dict[str, object] = {}
    # Stream (don't buffer) into the log -- append, not truncate: with two
    # stages per document, one log holds both. @@BAR lines are consumed here
    # and kept out of the log.
    with open(group_dir / "run.log", "a", encoding="utf-8") as fh:
        fh.write(f"===== stage: {stage} =====\n")
        for line in proc.stdout:
            if line.startswith("@@BAR "):
                _ipc_bar_event(line, bars)
            elif line.startswith("@@REQ "):
                _ipc_req_event(line, bars)
            else:
                fh.write(line)
    proc.wait()
    _ipc_req_clear(bars)
    for b in bars.values():  # child died mid-stage: free the terminal lines
        b.close()
    elapsed = time.time() - t0

    art = _artifacts(group_dir, doc_id)
    result: dict = {
        "doc_id": doc_id, "doc_type": doc_type, "file_name": file_name,
        "rel_path": record["location"]["rel_path"], "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1), "group_dir": str(group_dir),
        "json_ok": art["json"].is_file(), "md_ok": art["md"].is_file(),
    }
    ir_path = art["json"] if art["json"].is_file() else (
        art["stage1"] if art["stage1"].is_file() else None)
    if ir_path is not None:
        try:
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            if ir_path == art["stage1"]:
                ir = ir.get("ir", {})
            blocks = ir.get("blocks", [])
            result["blocks"] = len(blocks)
            result["tables"] = sum(1 for b in blocks if b.get("type") == "table")
            result["images"] = sum(1 for b in blocks if b.get("type") == "image")
            # The IR's own metadata.models (set by to_markdown.py's
            # _write_final) is the actual resolved config -- including any
            # auto-applied num_ctx default -- not just the nominal cfg value.
            result["models"] = ir.get("metadata", {}).get("models")
        except Exception:  # noqa: BLE001, S110 -- malformed JSON still counts as a failure below
            pass

    if stage == "parse":
        # phase-1 success = a complete checkpoint, not final output
        result["status"] = ("PASS" if proc.returncode == 0 and art["stage1"].is_file()
                            else "FAIL")
    else:
        result["status"] = ("PASS" if proc.returncode == 0 and result["json_ok"]
                            and result["md_ok"] else "FAIL")
    return result


def skipped_result(record: dict, index: int, out_dir: Path) -> dict:
    """Summary record for a document found already complete (resume path):
    same shape as run_stage's, elapsed 0, counted from the existing IR."""
    group_dir, doc_id = _group_dir(record, index, out_dir)
    art = _artifacts(group_dir, doc_id)
    result: dict = {
        "doc_id": doc_id, "doc_type": record.get("doc_type", "UNKNOWN"),
        "file_name": record["identity"]["file_name"],
        "rel_path": record["location"]["rel_path"], "exit_code": 0,
        "elapsed_s": 0.0, "group_dir": str(group_dir),
        "json_ok": art["json"].is_file(), "md_ok": art["md"].is_file(),
        "resumed": True,
    }
    try:
        ir = json.loads(art["json"].read_text(encoding="utf-8"))
        blocks = ir.get("blocks", [])
        result["blocks"] = len(blocks)
        result["tables"] = sum(1 for b in blocks if b.get("type") == "table")
        result["images"] = sum(1 for b in blocks if b.get("type") == "image")
        result["models"] = ir.get("metadata", {}).get("models")
    except Exception:  # noqa: BLE001, S110 -- unreadable IR leaves counts empty; PASS/FAIL is still decided from json_ok+md_ok
        pass
    result["status"] = "PASS" if result["json_ok"] and result["md_ok"] else "FAIL"
    return result


# ---------------------------------------------------------------------------
# phase runner
# ---------------------------------------------------------------------------


def run_phase(sample: list[dict], out_dir: Path, cfg: dict[str, str],
              to_markdown_py: Path, stage: str, workers: int,
              skip_indices: set[int] = frozenset()) -> dict[int, dict]:
    """Run one stage across the whole sample (that's the point: all VLM work,
    THEN all LLM work). Documents whose artifacts for this stage already
    exist are skipped -- that's the resume path after a crash/Ctrl-C.
    `skip_indices` excludes documents a previous phase already failed.
    Returns {sample index -> result record} for the documents actually run."""
    pending: list[tuple[int, dict]] = []
    resumed = 0
    for i, record in enumerate(sample, 1):
        if i in skip_indices:
            continue
        if _stage_done(record, i, out_dir, stage):
            resumed += 1
            continue
        pending.append((i, record))
    if resumed:
        print(f"  {resumed} document(s) already done for stage '{stage}' -- skipped (resume)")

    results: dict[int, dict] = {}
    n_pending = len(pending)

    def run_stage_announced(n: int, record: dict, i: int) -> dict:
        # Printed from the worker thread right as it picks up the task, so
        # with workers=1 these appear one at a time instead of bursting out
        # at submission. The actual parse runs in a subprocess whose
        # stdout/stderr are piped (not a TTY), so parser/progress.py's tqdm
        # bars never reach this terminal -- these lines, and the bar built
        # around this loop below, are this process's OWN terminal output
        # (a real TTY when run interactively) and render normally.
        progress.write(f"  [{n}/{n_pending}] {record['doc_type']:<16} "
                       f"{record['identity']['file_name']} ...")
        return run_stage(record, i, out_dir, cfg, to_markdown_py, stage)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
         progress.bar(n_pending, f"stage: {stage}", unit="doc") as pbar:
        # dict preserves submission order; calling .result() in that same
        # order (rather than as_completed()) keeps output in sample order
        # regardless of which worker finishes first -- it just blocks on
        # each in turn, which is a no-op cost when workers=1 anyway.
        futures = [(i, record, pool.submit(run_stage_announced, n, record, i))
                   for n, (i, record) in enumerate(pending, 1)]
        for i, record, fut in futures:
            result = fut.result()
            results[i] = result
            pbar.update(1)
            progress.write(f"  [{result['status']}] {record['doc_type']:<16} "
                          f"{result['file_name']} ({result['elapsed_s']}s)")
    return results


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def write_summary(results: list[dict], out_dir: Path, nominal_models: dict | None = None,
                  title: str = "e2e run") -> None:
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # The run's own intended config (from cfg_e2e_local.env, what preflight
        # validated) -- each document's real, resolved config (including any
        # auto-applied num_ctx default) is recorded per-document below in
        # documents[i]["models"], read back from that document's own IR JSON.
        "models": nominal_models,
        "total": len(results), "passed": passed, "failed": len(results) - passed,
        "documents": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {title} summary -- {passed}/{len(results)} passed", "",
    ]
    if nominal_models:
        lines.append("Models: " + ", ".join(
            f"{role}={info.get('model')} (num_ctx={info.get('num_ctx') or 'default'}, "
            f"thinking={'on' if info.get('thinking') else 'off'})"
            for role, info in nominal_models.items()))
        lines.append("")
    lines += [
        "| # | doc_type | file | status | blocks | tables | images | time(s) | parse/finish(s) | models |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        if r.get("resumed"):
            split = "resumed"
        elif "parse_s" in r:
            split = f"{r['parse_s']}/{r['finish_s']}"
        else:
            split = "-"
        doc_models = r.get("models") or {}
        models_note = ", ".join(f"{role}={info.get('model')}"
                                for role, info in doc_models.items()) or "-"
        lines.append(
            f"| {i} | {r['doc_type']} | {r['file_name']} | {r['status']} | "
            f"{r.get('blocks', '-')} | {r.get('tables', '-')} | {r.get('images', '-')} | "
            f"{r['elapsed_s']} | {split} | {models_note} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"{'#':<3} {'doc_type':<16} {'status':<6} {'blk':>4} {'tbl':>4} {'img':>4} {'s':>7}  file")
    for i, r in enumerate(results, 1):
        print(f"{i:<3} {r['doc_type']:<16} {r['status']:<6} "
              f"{r.get('blocks', '-'):>4} {r.get('tables', '-'):>4} {r.get('images', '-'):>4} "
              f"{r['elapsed_s']:>7}  {r['file_name']}")
    print()
    print(f"{passed}/{len(results)} passed. Summary: {out_dir / 'summary.json'} / "
          f"{out_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-datasheets", type=int, default=10)
    ap.add_argument("--n-mixed", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for random sample selection (also E2E_SAMPLE_SEED "
                         "env). Omit for a fresh random sample each run; the "
                         "chosen seed is printed so a run can be reproduced with "
                         "--seed <n>.")
    ap.add_argument("--workers", type=int, default=1,
                    help="documents processed concurrently (default 1 -- a single local "
                         "Ollama instance serves one GPU, so concurrent documents mostly "
                         "just queue behind each other and add contention risk; raise this "
                         "only if the server is known to handle it)")
    ap.add_argument("--out-dir", default=None,
                    help="default: e2e_test_output/<timestamp>/. Re-running with "
                         "an out-dir from an interrupted run RESUMES it: documents "
                         "whose artifacts already exist are skipped")
    ap.add_argument("--skip-checks", action="store_true",
                    help="skip steps [1/4]-[3/4] (dependencies/Ollama/thinking) -- for "
                         "quickly re-running after a checked run; sample availability "
                         "[4/4] always still runs")
    ap.add_argument("--single-pass", action="store_true",
                    help="run each document start-to-finish (--stage all) instead of "
                         "the phased VLM-then-LLM order -- the pre-phasing behavior, "
                         "kept as a fallback/baseline (still resumable per document)")
    ap.add_argument("--profile", default=None,
                    help="config profile under config/ (e.g. cfg_default); "
                         "default cfg_e2e_local. Or set PIPELINE_PROFILE env.")
    args = ap.parse_args()

    config_path = resolve_profile(args.profile)

    # Single source of truth for the model set: the config file. Preflight
    # checks the exact models (and thinking toggles) the run will use, so it
    # can never green-light a different model than the one that actually runs.
    cfg = parse_env_file(config_path)
    models = {"LLM": cfg["LLM_MODEL"], "VLM": cfg.get("VLM_MODEL") or cfg["LLM_MODEL"]}
    thinking = {
        "LLM": (cfg.get("LLM_THINKING_ON", "") in ("1", "true", "yes", "on")),
        "VLM": (cfg.get("VLM_THINKING_ON", cfg.get("LLM_THINKING_ON", ""))
                in ("1", "true", "yes", "on")),
    }
    # VLM_CLASSIFY only gets its own preflight entry when the profile actually
    # sets one of VLM_CLASSIFY_{PROVIDER,MODEL,BASE_URL,API_KEY} -- mirrors
    # llm/__init__.py's get_vlm_classify_client(), which otherwise reuses the
    # primary VLM client outright (nothing extra to check in that case).
    if any(cfg.get(f"VLM_CLASSIFY_{k}") for k in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY")):
        models["VLM_CLASSIFY"] = cfg.get("VLM_CLASSIFY_MODEL") or models["VLM"]
        thinking["VLM_CLASSIFY"] = (cfg.get("VLM_CLASSIFY_THINKING_ON", "") in ("1", "true", "yes", "on")
                                    if "VLM_CLASSIFY_THINKING_ON" in cfg else thinking["VLM"])
    # The run's nominal config, for summary.json/md -- each document's own IR
    # JSON separately records the REAL resolved config (see to_markdown.py's
    # _write_final), which can differ here only in num_ctx (an unset value
    # auto-defaults per-process, see llm/__init__.py; "num_ctx": None below
    # just means "whatever that default turns out to be").
    nominal_models = {
        "LLM": {"model": models["LLM"], "thinking": thinking["LLM"],
                "num_ctx": cfg.get("LLM_NUM_CTX")},
        "VLM": {"model": models["VLM"], "thinking": thinking["VLM"],
                "num_ctx": cfg.get("VLM_NUM_CTX", cfg.get("LLM_NUM_CTX"))},
    }
    if "VLM_CLASSIFY" in models:
        nominal_models["VLM_CLASSIFY"] = {
            "model": models["VLM_CLASSIFY"], "thinking": thinking["VLM_CLASSIFY"],
            "num_ctx": cfg.get("VLM_CLASSIFY_NUM_CTX", cfg.get("VLM_NUM_CTX", cfg.get("LLM_NUM_CTX"))),
        }

    if not args.skip_checks:
        if not check_dependencies(config_path):
            sys.exit("Dependency check failed -- install the missing packages and retry.")
        if not check_ollama_models(models):
            sys.exit("Ollama model check failed -- see errors above.")
        if not eom.check_thinking_disabled(models, thinking, root=OLLAMA_ROOT):
            sys.exit("Thinking-disabled check failed -- refusing to run 20 documents "
                     "against a model that's still burning its budget on hidden reasoning.")

    seed = resolve_sample_seed(args.seed)
    rng = random.Random(seed)
    print(f"sample seed: {seed}  (reproduce this exact sample with --seed {seed} "
          f"or E2E_SAMPLE_SEED={seed})")

    docs = load_document_nodes()
    datasheets = select_datasheets(docs, args.n_datasheets, rng)
    mixed = select_mixed(docs, args.n_mixed,
                         {d["identity"]["doc_id"] for d in datasheets}, rng)
    if not check_sample(datasheets, mixed, args.n_datasheets, args.n_mixed):
        sys.exit("Sample check failed -- not enough matching documents found on disk.")

    sample = datasheets + mixed

    out_dir = (Path(args.out_dir) if args.out_dir
              else BASE_DIR / "e2e_test_output" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  # noqa: DTZ005 -- output directory name; local wall-clock is intentional
    # Must be absolute: it's handed to each subprocess as STORAGE_OUTPUT_DIR,
    # and that subprocess's cwd is PARSER_DIR, not BASE_DIR -- a relative
    # path would resolve against the wrong directory and silently write
    # output where nothing expects it.
    if not out_dir.is_absolute():
        out_dir = (BASE_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    to_markdown_py = PARSER_DIR / "scripts" / "to_markdown.py"

    print(f"\nRunning {len(sample)} documents ({len(datasheets)} datasheet + {len(mixed)} mixed) "
          f"-> {out_dir}\n")

    if args.single_pass:
        print("[single-pass] per-document order (models swap per stage)\n")
        final = run_phase(sample, out_dir, cfg, to_markdown_py, "all", args.workers)
        results = [final.get(i) or skipped_result(record, i, out_dir)
                   for i, record in enumerate(sample, 1)]
    else:
        # Phase 1 -- every document's VLM work (parse + visual-region
        # classify + image classify/OCR). The VLM was warmed by preflight;
        # nothing here needs the big LLM yet.
        print(f"[phase 1/2] parse + classify + OCR (VLM: {models['VLM']})\n")
        parse_results = run_phase(sample, out_dir, cfg, to_markdown_py,
                                  "parse", args.workers)
        parse_failed = {i for i, r in parse_results.items() if r["status"] != "PASS"}

        # Phase 2 -- every document's LLM work. Swap the models once, here,
        # instead of at every stage boundary of every document.
        print(f"\n[phase 2/2] OCR checks + table descriptions (LLM: {models['LLM']})\n")
        if not args.skip_checks:
            eom.warm_model(eom.ollama_root(OLLAMA_ROOT), models["LLM"])
        final = run_phase(sample, out_dir, cfg, to_markdown_py, "finish",
                          args.workers, skip_indices=parse_failed)

        results = []
        for i, record in enumerate(sample, 1):
            if i in parse_failed:  # no checkpoint -> finish never attempted
                results.append(parse_results[i])
            elif i in final:
                r = dict(final[i])
                if i in parse_results:  # not resumed: both stages ran now
                    r["parse_s"] = parse_results[i]["elapsed_s"]
                    r["finish_s"] = r["elapsed_s"]
                    r["elapsed_s"] = round(r["parse_s"] + r["finish_s"], 1)
                results.append(r)
            else:  # fully done in an earlier run
                results.append(skipped_result(record, i, out_dir))

    write_summary(results, out_dir, nominal_models)

    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
