"""Benchmark CLI: score a parser's Markdown output against the source PDF.

Config is split by KIND (repo-wide taxonomy, see root CONFIG.md):

  * GRADING TUNING (dpi, weights, max_pages, chunk, token budgets) lives in
    `config/default.toml`, loaded and validated by `benchmark/config.py`. A
    `--config <file.toml>` merges a partial override on top; a CLI flag beats
    both. Precedence: default.toml < --config toml < flag.
  * MODEL CONNECTION identity (provider/model/base_url/api_key/thinking/
    num_ctx) lives in the ENVIRONMENT, not here -- the same `VLM_*`/`LLM_*`
    variables the parser uses. .env files are loaded LAYERED (highest first):
    --env-file, then benchmark/.env, then ./.env, then parser/.env; each fills
    only what higher-precedence ones leave unset (so benchmark/.env can override
    just the judge model while everything else still comes from parser/.env). A
    flag (--model, --provider, ...) is the top override for a single run.
  * RUN INPUTS (folder/scope/out) are CLI flags (-f/-s/-o), each with an
    optional .env default (BENCHMARK_FOLDER/SCOPE/OUT) the flag overrides --
    the same way parser/pipeline take their input paths from the env.

Model selection reuses the parser's provider-agnostic client
(`parser/llm/`), so any provider it supports -- ollama, anthropic,
openai-compatible, openrouter -- works here. The model must be multimodal
(the PDF is graded from page images).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Repo root = grandparent of this package (repo_root/tools/benchmark/cli.py).
# `tools/` is a real installed package now (D-46/D-62, pyproject.toml
# include=["medrag*", "tools*"]) — no sys.path bootstrap needed to import
# `tools.benchmark.*`; run this as `python -m tools.benchmark ...`, not as a
# bare script.
_HERE = Path(__file__).resolve().parent          # the benchmark/ package dir
_REPO_ROOT = _HERE.parent.parent
_PARSER_DIR = _REPO_ROOT / "src" / "medrag" / "pipeline" / "parser"


def _default_report_path(folder: Path, reports_dir: Path | None = None) -> Path:
    """Default `-o/--out`: `<reports_dir>/<timestamp>__<graded folder
    name>.json` -- never dropped directly into the graded folder (that
    clutters pipeline output dirs and a later run silently overwrites an
    earlier one at the same fixed name). The timestamp makes "when was this
    run" visible at a glance; the graded folder's own name makes "what was
    graded" visible too. Same readable %Y-%m-%d_%H-%M-%S style
    pipeline/run_e2e_test.py uses for its own output dirs -- LOCAL system
    time, so the name matches the wall clock; still sorts correctly
    (zero-padded, left-to-right) and stays filesystem-safe on Windows (no
    colons). `reports_dir` defaults to `benchmark/benchmark_reports`; a
    directory-valued `-o/BENCHMARK_OUT` passes its own dir here instead."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005 -- rapor dosya adi; docstring'de kilitlendigi gibi yerel duvar saati kasitli
    graded = folder.resolve().name or "root"
    if reports_dir is None:
        reports_dir = Path(__file__).resolve().parent / "benchmark_reports"
    return reports_dir / f"{stamp}__{graded}.json"


def _resolve_out_path(out_in: str | None, folder: Path) -> Path:
    """Resolve `-o/--out`/`BENCHMARK_OUT` to a concrete file path. A value that
    is (or looks like) a directory -- an existing dir, or a trailing slash --
    is treated as the reports directory to drop the usual auto-named report
    into, not a literal filename (a directory can't be `write_text`'d to)."""
    if not out_in:
        return _default_report_path(folder)
    out_path = Path(out_in)
    if out_path.is_dir() or out_in.endswith(("/", "\\")):
        return _default_report_path(folder, reports_dir=out_path)
    return out_path


def _load_env(explicit: str | None) -> list[Path]:
    """Load .env files LAYERED by precedence (highest first), so benchmark's own
    file overrides the shared ones per-variable while still inheriting whatever
    it doesn't set -- no file has to be complete on its own. Real environment
    variables always win over every file.

    Order, highest precedence first:
        1. --env-file <path>   (explicit)
        2. benchmark/.env      (this package's own file -- the default home for
                                a judge-specific model and default run inputs)
        3. ./.env              (repo root)
        4. parser/.env         (shared parser model config)

    `load_dotenv(override=False)` means a higher-precedence file's value is not
    overwritten by a lower one -- values cascade DOWN the list filling only
    what's still unset. So benchmark/.env with just `VLM_MODEL=...` overrides
    the judge model while provider/base_url still come from parser/.env. Returns
    the files actually loaded (highest precedence first) for the run banner."""
    from dotenv import load_dotenv

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [_HERE / ".env", _REPO_ROOT / ".env", _PARSER_DIR / ".env"]

    loaded: list[Path] = []
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


def _apply_model_overrides(args) -> None:
    """Push --provider/--model/... onto VLM_* env vars (flags beat the .env)."""
    mapping = {
        "provider": "VLM_PROVIDER",
        "model": "VLM_MODEL",
        "base_url": "VLM_BASE_URL",
        "api_key": "VLM_API_KEY",
    }
    for attr, env in mapping.items():
        val = getattr(args, attr)
        if val:
            os.environ[env] = val
    # "thinking" is its own on/off setting (not a pass-through string like the
    # ones above) -- explicit here rather than left to inherit whatever
    # parser/.env happens to have, since a hybrid-reasoning judge model
    # silently swallowing its own answer is exactly the failure _preflight
    # below is built to catch.
    if args.thinking is not None:
        os.environ["VLM_THINKING_ON"] = "1" if args.thinking == "on" else "0"
    # Ollama-only: forces the context window via native /api/chat instead of
    # /v1 (see parser/llm/openai_compat.py) -- the only way to guarantee the
    # judge's own prompt (page images + full Markdown) isn't silently
    # truncated. No effect on any other provider. Left unset/0, llm/__init__.py's
    # _build_client applies its own safe default for ollama -- this only needs
    # to push an EXPLICIT choice through.
    if args.num_ctx:
        os.environ["VLM_NUM_CTX"] = str(args.num_ctx)


def _parse_weights(spec: str | None, config_weights: dict | None) -> dict[str, float]:
    from tools.benchmark.judge import DIMENSIONS

    weights = {d: 1.0 for d in DIMENSIONS}
    if config_weights:
        for k, v in config_weights.items():
            if k not in weights:
                raise SystemExit(f"unknown weight dimension in config: {k!r}")
            weights[k] = float(v)
    if spec:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            key, _, val = part.partition("=")
            key = key.strip()
            if key not in weights:
                raise SystemExit(f"unknown weight dimension: {key!r} "
                                 f"(expected one of {', '.join(weights)})")
            weights[key] = float(val)
    return weights


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-f", "--folder",
                    help="folder to grade: either one holding a .pdf + .md, or a "
                         "folder of such subfolders (batch). Required unless "
                         "BENCHMARK_FOLDER is set in the .env.")
    ap.add_argument("-s", "--scope",
                    help="path to a scope .md; grading is restricted to what it "
                         "describes. Omit to grade the whole document (or set "
                         "BENCHMARK_SCOPE in the .env).")
    ap.add_argument("--config",
                    help="TOML file overriding grading tuning (dpi, weights, "
                         "max_pages, chunk, token budgets); partial file ok, "
                         "merged over config/default.toml. Model/provider come "
                         "from .env, not here.")
    ap.add_argument("-o", "--out",
                    help="write the full JSON report here (or set BENCHMARK_OUT in "
                         "the .env; default: "
                         "benchmark/benchmark_reports/<timestamp>__<graded folder>.json). "
                         "A directory (existing, or ending in / or \\) is treated as the "
                         "reports dir -- the auto-named report is written inside it.")
    ap.add_argument("--provider", help="override VLM_PROVIDER (e.g. ollama, anthropic)")
    ap.add_argument("--model", help="override VLM_MODEL (must be multimodal)")
    ap.add_argument("--base-url", help="override VLM_BASE_URL (openai-compatible root)")
    ap.add_argument("--api-key", help="override VLM_API_KEY")
    ap.add_argument("--thinking", choices=("on", "off"),
                    help="override VLM_THINKING_ON explicitly (on/off); default: "
                         "whatever parser/.env has, or off if unset. A judge model "
                         "left thinking ON can burn its whole token budget on hidden "
                         "reasoning and return empty/unparseable content -- see the "
                         "[2/2] preflight check.")
    ap.add_argument("--num-ctx", type=int,
                    help="ollama only: force this context window (tokens) via Ollama's "
                         "native /api/chat instead of /v1 -- the only way to guarantee "
                         "the judge's own prompt (page images + Markdown) isn't silently "
                         "truncated. Unset = /v1 as before (--assume-context just warns). "
                         "No effect on any other provider.")
    ap.add_argument("--env-file", help="explicit .env to load, highest precedence "
                    "(default layered order: benchmark/.env, then ./.env, then "
                    "parser/.env -- each fills only what higher ones leave unset)")
    ap.add_argument("--dpi", type=int, help="PDF render resolution (default from config)")
    ap.add_argument("--max-pages", type=int,
                    help="cap pages sent to the model per document (default from config; 0=all)")
    ap.add_argument("--chunk", action=argparse.BooleanOptionalAction, default=None,
                    help="grade a document too big for one context window in "
                         "page-aligned chunks (default: on). A document whose estimated "
                         "prompt exceeds 80%% of the context window (--num-ctx, else "
                         "--assume-context) and has a parser IR JSON next to its .md is "
                         "split into page windows, each graded on its own slice of the "
                         "Markdown, then aggregated page-weighted. --no-chunk forces the "
                         "old single-call behavior (a big document then just warns).")
    ap.add_argument("--chunk-pages", type=int,
                    help="pages per chunk window (default: auto -- the most that fit the "
                         "context budget). An explicit value caps the window height.")
    ap.add_argument("--max-tokens", type=int,
                    help="model response budget per document (default from config)")
    ap.add_argument("--assume-context", type=int,
                    help="context window (tokens) assumed ONLY as a last resort -- when "
                         "neither --num-ctx nor the client's own enforced window is known "
                         "(default 4096). For ollama the real enforced window (an explicit "
                         "--num-ctx, else the auto-applied 16384 default) is used instead, "
                         "for both the chunk trigger and the size warning, so this rarely "
                         "applies there.")
    ap.add_argument("--weights",
                    help="overall weights, e.g. 'accuracy=2,coverage=1,clarity=1' "
                         "(default: equal)")
    return ap


def _load_tuning(config_path: str | None):
    """Load grading tuning: config/default.toml, optionally overridden by a
    partial `--config <file.toml>`. Model connection and run inputs are NOT
    here (env and CLI flags respectively) -- see the module docstring."""
    from pydantic import ValidationError

    from tools.benchmark.config import load_config

    try:
        return load_config(config_path or None)
    except FileNotFoundError as exc:
        raise SystemExit(f"config file not found: {exc}")
    except ValidationError as exc:
        raise SystemExit(f"invalid benchmark config:\n{exc}")


def _check_thinking(client, want_thinking: bool) -> tuple[bool, str]:
    """One trivial vision call to catch a reasoning model swallowing its own
    answer -- a hybrid-reasoning model can be "up" (HTTP 200, health.probe_vlm
    passes) yet still burn the whole max_tokens budget on hidden <think>
    tokens and return empty content, or leak the block into the reply the
    judge is supposed to parse as JSON. Mirrors run_e2e_test.py's
    check_thinking_disabled, but through the provider-agnostic VLMClient
    (so it works for openrouter/ollama/openai-compat alike) rather than a
    raw Ollama HTTP call."""
    import io

    from PIL import Image

    from medrag.pipeline.parser.llm import LLMError

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=255).save(buf, format="PNG")
    try:
        reply = client.complete_vision(
            system="Reply with exactly one word.", user="Reply with exactly: OK",
            images=[("image/png", buf.getvalue())], max_tokens=20)
    except LLMError as exc:
        return False, f"transport: {exc}"
    if want_thinking:
        return True, ""  # empty content is expected under a tiny budget here
    stripped = reply.strip()
    if "<think" in reply.lower() or not stripped:
        return False, f"thinking not suppressed -- raw content: {reply!r}"
    return True, ""


def _preflight(client) -> None:
    """Two known-answer round trips before any document is touched -- a
    reachable endpoint (HTTP 200) is not the same as a usable model. Aborts
    the whole run (no document graded) rather than letting every document
    fail the same way one at a time."""
    from medrag.pipeline.parser.llm.health import probe_vlm

    print("model preflight:", file=sys.stderr)
    print("  [1/2] known-answer image read (canary)...", file=sys.stderr)
    ok, detail = probe_vlm(client)
    print(f"    {'OK' if ok else 'FAILED: ' + detail}", file=sys.stderr)
    if not ok:
        raise SystemExit(f"model preflight failed (image read): {detail}")

    thinking_on = (os.getenv("VLM_THINKING_ON") or os.getenv("LLM_THINKING_ON")
                  or "").strip().lower() in ("1", "true", "yes", "on")
    print(f"  [2/2] thinking/reasoning behaves as configured "
          f"(want {'ON' if thinking_on else 'OFF'})...", file=sys.stderr)
    ok, detail = _check_thinking(client, thinking_on)
    print(f"    {'OK' if ok else 'FAILED: ' + detail}", file=sys.stderr)
    if not ok:
        raise SystemExit(f"model preflight failed (thinking): {detail}")
    print(file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _load_tuning(args.config)

    # Load the .env FIRST: besides the model connection it can also supply the
    # default run inputs (BENCHMARK_FOLDER/SCOPE/OUT), so it must be read before
    # the folder is resolved and validated below. Model flags are pushed on top
    # right after, so a --model/--provider/... flag beats whatever the .env set.
    env_used = _load_env(args.env_file)
    _apply_model_overrides(args)

    # Run inputs: a CLI flag wins; else a BENCHMARK_* env var (parser/pipeline
    # already take input paths from the env this way, see root CONFIG.md); else
    # folder is an error, scope means whole-document, out is the default path.
    folder = args.folder or os.getenv("BENCHMARK_FOLDER")
    if not folder:
        raise SystemExit("error: no folder given (use -f/--folder, or set "
                         "BENCHMARK_FOLDER in the .env)")
    scope_in = args.scope or os.getenv("BENCHMARK_SCOPE")
    out_in = args.out or os.getenv("BENCHMARK_OUT")

    from tools.benchmark.chunk import build_windows, pages_for_budget
    from tools.benchmark.discover import Skipped, discover
    from tools.benchmark.judge import (
        JudgeError,
        estimate_prompt_tokens,
        evaluate_document,
        evaluate_windows,
    )
    from tools.benchmark.render import render_pdf
    from tools.benchmark.report import build_report, format_console, write_report

    # Context caps common enough (Ollama defaults, round power-of-two sizes)
    # that a reported prompt_tokens landing within a few tokens of one is
    # itself evidence -- not just a guess -- that the server silently
    # truncated the prompt to fit.
    _COMMON_CONTEXT_CAPS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)

    # Resolve grading tuning: CLI flag wins if given, else config value.
    weights = _parse_weights(args.weights, cfg.weights.as_dict())
    dpi = args.dpi if args.dpi is not None else cfg.render.dpi
    max_pages = (args.max_pages or None) if args.max_pages is not None \
        else cfg.grading.max_pages_or_none()  # 0/None = all
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.grading.max_tokens
    assume_context = args.assume_context if args.assume_context is not None \
        else cfg.grading.assume_context
    chunk_enabled = args.chunk if args.chunk is not None else cfg.chunking.enabled
    chunk_pages = args.chunk_pages if args.chunk_pages is not None \
        else cfg.chunking.pages_per_window_or_none()

    documents, skipped = discover(Path(folder))
    if not documents:
        print(format_console([], skipped))
        raise SystemExit("no gradable PDF/Markdown pair found")

    scope_md = None
    if scope_in:
        scope_path = Path(scope_in)
        if not scope_path.is_file():
            raise SystemExit(f"scope file not found: {scope_path}")
        scope_md = scope_path.read_text(encoding="utf-8")

    from medrag.pipeline.parser.llm import LLMError, get_vlm_client

    try:
        client = get_vlm_client("primary")
    except LLMError as exc:
        raise SystemExit(f"could not build model client: {exc}")

    model_id = os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL")
    thinking_id = os.getenv("VLM_THINKING_ON") or os.getenv("LLM_THINKING_ON") or "(unset, off)"
    num_ctx_id = os.getenv("VLM_NUM_CTX") or os.getenv("LLM_NUM_CTX")
    ctx_note = f"  num_ctx: {num_ctx_id} (enforced, native /api/chat)" if num_ctx_id else "  num_ctx: (unenforced, /v1)"
    env_note = ("  env: " + ", ".join(str(p) for p in env_used)) if env_used else ""
    print(f"model: {model_id}  provider: {os.getenv('VLM_PROVIDER') or os.getenv('LLM_PROVIDER')}"
          f"  thinking: {thinking_id}{ctx_note}{env_note}", file=sys.stderr)

    # The context window the judge call will REALLY run under -- what the chunk
    # trigger and the pre-request size warning must both compare against so they
    # reflect reality rather than a stale assumption:
    #   * an explicit --num-ctx (args.num_ctx) wins; else
    #   * the client's own enforced window: for ollama, get_vlm_client above has
    #     ALREADY auto-applied _DEFAULT_OLLAMA_NUM_CTX (16384) and stored it on
    #     client.num_ctx even when nothing was set anywhere -- so the trigger
    #     uses that real 16384, not the 4096 assume-context default, and a
    #     document that actually fits is not needlessly chunked; else
    #   * --assume-context, for a provider that exposes no enforced window.
    effective_num_ctx = args.num_ctx or getattr(client, "num_ctx", None)
    ctx_enforced = effective_num_ctx is not None
    ctx_limit = effective_num_ctx or assume_context
    usable = ctx_limit * 0.8

    _preflight(client)  # aborts before any document is touched if it fails

    print(f"grading {len(documents)} document(s), scope: "
          f"{scope_in or '(whole document)'}\n", file=sys.stderr)

    evaluations = []
    for doc in documents:
        try:
            images, total = render_pdf(doc.pdf, dpi=dpi, max_pages=max_pages)
            markdown = doc.md.read_text(encoding="utf-8")

            est = estimate_prompt_tokens(markdown, scope_md, len(images))

            # An over-budget document is graded in page-aligned chunks instead of
            # one silently-truncated call -- but only when a parser IR JSON sits
            # next to the .md (chunk.py needs its per-block page numbers to slice
            # the Markdown to match each page window). Without one, fall through
            # to the single call and the same warning as before.
            windows = None
            if chunk_enabled and est > usable and doc.ir_json is not None:
                ppw = chunk_pages or pages_for_budget(
                    markdown=markdown, scope_md=scope_md, total_pages=total,
                    budget_tokens=usable)
                windows = build_windows(
                    doc.ir_json, images, pages_per_window=ppw,
                    warn=lambda reason, _doc=doc: print(
                        f"  !! {_doc.doc_id}: over budget but CANNOT be chunked: "
                        f"{reason}; falling back to one full-document call.",
                        file=sys.stderr))

            if windows:
                print(f"  [{len(evaluations) + 1}/{len(documents)}] {doc.doc_id}: "
                      f"~{est} est. tokens over budget (~{int(usable)}); grading in "
                      f"{len(windows)} chunk(s) of <= {ppw} page(s)...", file=sys.stderr)

                def _on_window(idx, n, win, sub):
                    print(f"      chunk {idx + 1}/{n} p{win.page_lo}-{win.page_hi}: "
                          f"overall {sub.overall}", file=sys.stderr)

                ev = evaluate_windows(
                    client, doc_id=doc.doc_id, windows=windows, scope_md=scope_md,
                    total_pages=total, weights=weights, max_tokens=max_tokens,
                    ir_json=doc.ir_json, on_window=_on_window)
                evaluations.append(ev)
                parsed_with = ", ".join(f"{role}={info.get('model')}"
                                        for role, info in (ev.parse_models or {}).items())
                parse_note = f"  (parsed with {parsed_with})" if parsed_with else ""
                print(f"    -> {doc.doc_id}: overall {ev.overall} "
                      f"(page-weighted over {len(windows)} chunk(s)){parse_note}",
                      file=sys.stderr)
                continue

            if est >= usable:
                why = "" if doc.ir_json is not None or not chunk_enabled else \
                    " (no IR JSON beside the .md, so it can't be chunked -- see --chunk)"
                if ctx_enforced:
                    print(f"  !! WARNING {doc.doc_id}: ~{est} estimated input tokens "
                          f"({len(images)} page image(s)) -- close to or over the enforced "
                          f"--num-ctx ({ctx_limit}).{why} This is a rough estimate, not exact; "
                          f"Ollama's native /api/chat is being used so the window IS actually "
                          f"applied, but if the estimate is right the request may fail or run "
                          f"very slow instead of truncating. Consider --max-pages or a bigger "
                          f"--num-ctx.", file=sys.stderr)
                else:
                    print(f"  !! WARNING {doc.doc_id}: ~{est} estimated input tokens "
                          f"({len(images)} page image(s)) -- close to or over the assumed "
                          f"context window ({ctx_limit}).{why} This is a rough estimate, not "
                          f"exact. A server that silently truncates an over-budget prompt "
                          f"(Ollama's /v1 endpoint does, with no error) would grade against "
                          f"content it never saw. Consider --max-pages, or --num-ctx to "
                          f"actually enforce (not just warn about) the window.", file=sys.stderr)

            ev = evaluate_document(
                client, doc_id=doc.doc_id, markdown=markdown, scope_md=scope_md,
                images=images, total_pages=total, weights=weights,
                max_tokens=max_tokens, ir_json=doc.ir_json)
            evaluations.append(ev)

            tok_note = f"  [{ev.prompt_tokens}tok]" if ev.prompt_tokens is not None else ""
            parsed_with = ", ".join(f"{role}={info.get('model')}"
                                    for role, info in (ev.parse_models or {}).items())
            parse_note = f"  (parsed with {parsed_with})" if parsed_with else ""
            print(f"  [{len(evaluations)}/{len(documents)}] {doc.doc_id}: "
                  f"overall {ev.overall}{tok_note}{parse_note}", file=sys.stderr)

            near_cap = next((c for c in _COMMON_CONTEXT_CAPS
                             if ev.prompt_tokens is not None and abs(ev.prompt_tokens - c) <= 8),
                            None)
            if near_cap:
                print(f"    !! {doc.doc_id}: actual prompt_tokens={ev.prompt_tokens} is "
                      f"suspiciously close to a common context cap ({near_cap}) -- the "
                      f"server may have silently truncated the input; treat this score "
                      f"with caution.", file=sys.stderr)
        except (JudgeError, LLMError) as exc:
            skipped.append(Skipped(doc.pdf.parent, f"grading failed: {exc}"))
            print(f"  !! {doc.doc_id}: {exc}", file=sys.stderr)

    print()
    print(format_console(evaluations, skipped))

    report = build_report(evaluations, skipped, scope_path=scope_in, model=model_id)
    out = _resolve_out_path(out_in, Path(folder))
    write_report(report, out)
    print(f"\nreport -> {out}", file=sys.stderr)
    return 0 if evaluations else 1


if __name__ == "__main__":
    raise SystemExit(main())
