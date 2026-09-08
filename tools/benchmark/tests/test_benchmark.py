"""Offline tests for the benchmark plumbing -- no model, no network.

Mirrors the parser's testing philosophy: hand-built folders + a fake VLM client
whose canned reply exercises discovery, prompt assembly, JSON parsing, weighted
overall, and reporting. Run with `pytest tools/benchmark/tests` from the repo
root (`tools/` is a real installed package now, D-46/D-62 -- no sys.path
bootstrap needed; bare `python tools/benchmark/tests/test_benchmark.py` is no
longer supported).
"""

from __future__ import annotations

import os
from pathlib import Path

from tools.benchmark.chunk import build_windows, pages_for_budget
from tools.benchmark.discover import discover
from tools.benchmark.judge import (
    JudgeError,
    build_user_prompt,
    evaluate_document,
    evaluate_windows,
    parse_evaluation,
)
from tools.benchmark.report import build_report, format_console

EQUAL = {"accuracy": 1.0, "coverage": 1.0, "clarity": 1.0}


class FakeVLM:
    """Records each call and returns a canned JSON reply.

    Pass a single string for one fixed reply, or a list to return a different
    reply per call (for multi-window chunked grading)."""

    def __init__(self, reply):
        self._replies = reply if isinstance(reply, list) else None
        self.reply = reply if self._replies is None else None
        self.last = None
        self.calls = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.last = {"system": system, "user": user, "n_images": len(images)}
        self.calls.append(self.last)
        if self._replies is not None:
            return self._replies[len(self.calls) - 1]
        return self.reply


def _canned(acc=80, cov=70, cla=90):
    return (
        f'{{"accuracy": {{"score": {acc}, "reasoning": "ok", "issues": ["x"]}},'
        f' "coverage": {{"score": {cov}, "reasoning": "gaps", "issues": []}},'
        f' "clarity": {{"score": {cla}, "reasoning": "clean", "issues": []}},'
        ' "summary": "decent"}'
    )


def test_discover_single(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "doc.md").write_text("# hi", encoding="utf-8")
    docs, skipped = discover(tmp_path)
    assert len(docs) == 1 and not skipped
    assert docs[0].pdf.name == "doc.pdf" and docs[0].md.name == "doc.md"


def test_discover_batch_and_skips(tmp_path):
    good = tmp_path / "a"
    good.mkdir()
    (good / "a.pdf").write_bytes(b"%PDF")
    (good / "a.md").write_text("x", encoding="utf-8")
    bad = tmp_path / "b"
    bad.mkdir()
    (bad / "b.pdf").write_bytes(b"%PDF")  # no md
    docs, skipped = discover(tmp_path)
    assert [d.doc_id for d in docs] == ["a"]
    assert len(skipped) == 1 and "no .md" in skipped[0].reason


def test_parse_and_weighted_overall():
    ev = parse_evaluation(_canned(80, 60, 100), "d", EQUAL)
    assert ev.dimensions["accuracy"].score == 80
    assert ev.overall == 80.0  # (80+60+100)/3
    weighted = parse_evaluation(_canned(80, 60, 100), "d",
                                {"accuracy": 2, "coverage": 1, "clarity": 1})
    assert weighted.overall == round((80 * 2 + 60 + 100) / 4, 1)  # 80.0


def test_parse_tolerates_fences_and_clamps():
    reply = "```json\n" + _canned(150, -5, 50) + "\n```"
    ev = parse_evaluation(reply, "d", EQUAL)
    assert ev.dimensions["accuracy"].score == 100  # clamped
    assert ev.dimensions["coverage"].score == 0     # clamped


def test_parse_rejects_garbage():
    try:
        parse_evaluation("not json at all", "d", EQUAL)
    except JudgeError:
        return
    raise AssertionError("expected JudgeError")


def test_prompt_includes_scope_and_truncation_note():
    p = build_user_prompt("MD BODY", "ONLY grade tables.",
                          total_pages=10, rendered_pages=3)
    assert "ONLY grade tables." in p
    assert "MD BODY" in p
    assert "3 of 10" in p  # truncation disclosed, never silent


def test_evaluate_document_end_to_end():
    fake = FakeVLM(_canned())
    imgs = [("image/png", b"\x89PNG")]
    ev = evaluate_document(fake, doc_id="d", markdown="# t", scope_md=None,
                           images=imgs, total_pages=1, weights=EQUAL)
    assert ev.rendered_pages == 1 and ev.total_pages == 1
    assert fake.last["n_images"] == 1
    report = build_report([ev], [], scope_path=None, model="fake")
    assert report["aggregate"]["count"] == 1
    assert "overall" in format_console([ev], [])


def _write_ir(path: Path, n_pages: int) -> Path:
    """A minimal parser IR JSON with a heading+paragraph on each of `n_pages`."""
    from medrag.pipeline.parser.parsers.base import (
        HeadingBlock,
        ParagraphBlock,
        ParsedDocument,
        Span,
    )

    blocks = []
    for p in range(1, n_pages + 1):
        blocks.append(HeadingBlock(id=f"h{p}", span=Span(page=p),
                                   text=f"Page {p} heading", level=1))
        blocks.append(ParagraphBlock(id=f"b{p}", span=Span(page=p),
                                     text=f"Body text for page {p}."))
    ParsedDocument(doc_id="d", source_path="d.pdf", fmt="pdf", blocks=blocks).save(path)
    return path


def test_build_windows_page_aligned(tmp_path):
    ir = _write_ir(tmp_path / "d.json", n_pages=5)
    images = [("image/png", bytes([i])) for i in range(5)]  # page i+1 -> images[i]
    windows = build_windows(ir, images, pages_per_window=2)
    assert [(w.page_lo, w.page_hi) for w in windows] == [(1, 2), (3, 4), (5, 5)]
    assert [w.n_pages for w in windows] == [2, 2, 1]
    assert windows[0].images == images[0:2]
    assert windows[2].images == images[4:5]
    # each slice's Markdown carries only its own pages' content
    assert "Page 1 heading" in windows[0].markdown
    assert "Page 3 heading" not in windows[0].markdown
    assert "Page 5 heading" in windows[2].markdown


def test_build_windows_none_when_small(tmp_path):
    ir = _write_ir(tmp_path / "d.json", n_pages=3)
    images = [("image/png", b"x")] * 3
    assert build_windows(ir, images, pages_per_window=8) is None  # fits one window


def test_build_windows_none_without_page_provenance(tmp_path):
    from medrag.pipeline.parser.parsers.base import ParagraphBlock, ParsedDocument
    p = tmp_path / "d.json"
    ParsedDocument(doc_id="d", source_path="d.md", fmt="markdown",
                   blocks=[ParagraphBlock(id="b0", text="no page")]).save(p)
    images = [("image/png", b"x")] * 5
    assert build_windows(p, images, pages_per_window=2) is None  # can't align


def test_build_windows_slices_truncated_render(tmp_path):
    """A --max-pages-truncated render (fewer images than IR pages) must still
    be windowed even when the rendered pages fit one window: the window slices
    the Markdown to the rendered pages, while the single-call fallback would
    send the ENTIRE Markdown (the unrendered tail included) and blow the very
    budget that triggered chunking."""
    ir = _write_ir(tmp_path / "d.json", n_pages=10)   # IR spans 10 pages...
    images = [("image/png", bytes([i])) for i in range(4)]  # ...4 rendered
    windows = build_windows(ir, images, pages_per_window=8)
    assert [(w.page_lo, w.page_hi) for w in windows] == [(1, 4)]
    assert windows[0].images == images
    assert "Page 4 heading" in windows[0].markdown
    assert "Page 5 heading" not in windows[0].markdown  # unrendered tail dropped


def test_build_windows_fallback_reason_reported(tmp_path):
    """Every silent None must explain itself through `warn` -- a benchmark run
    where chunking was needed but impossible has to say WHY on stderr."""
    from medrag.pipeline.parser.parsers.base import ParagraphBlock, ParsedDocument
    reasons = []

    p = tmp_path / "no_pages.json"
    ParsedDocument(doc_id="d", source_path="d.md", fmt="markdown",
                   blocks=[ParagraphBlock(id="b0", text="no page")]).save(p)
    images = [("image/png", b"x")] * 5
    assert build_windows(p, images, pages_per_window=2, warn=reasons.append) is None
    assert "no per-block page numbers" in reasons[-1]

    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")
    assert build_windows(tmp_path / "corrupt.json", images,
                         pages_per_window=2, warn=reasons.append) is None
    assert "failed to load" in reasons[-1]

    ir = _write_ir(tmp_path / "small.json", n_pages=3)
    assert build_windows(ir, [("image/png", b"x")] * 3,
                         pages_per_window=8, warn=reasons.append) is None
    assert "fits one" in reasons[-1]


def test_evaluate_windows_page_weighted(tmp_path):
    ir = _write_ir(tmp_path / "d.json", n_pages=5)
    images = [("image/png", bytes([i])) for i in range(5)]
    windows = build_windows(ir, images, pages_per_window=2)  # [1-2],[3-4],[5-5]
    fake = FakeVLM([_canned(90, 90, 90), _canned(60, 60, 60), _canned(0, 0, 0)])
    ev = evaluate_windows(fake, doc_id="d", windows=windows, scope_md=None,
                          total_pages=5, weights=EQUAL)
    assert len(fake.calls) == 3
    # page-weighted mean: (90*2 + 60*2 + 0*1) / 5 == 60.0, not the plain mean 50
    assert ev.dimensions["accuracy"].score == 60.0
    assert ev.overall == 60.0
    assert ev.rendered_pages == 5 and ev.total_pages == 5
    assert [w["pages"] for w in ev.windows] == [[1, 2], [3, 4], [5, 5]]
    assert "pages 1-2 of a 5-page" in fake.calls[0]["user"]  # window note reached model
    # issues stay page-tagged so nothing is lost in the aggregate
    assert any(i.startswith("[p1-2]") for i in ev.dimensions["accuracy"].issues)


def test_prompt_windowed_note():
    p = build_user_prompt("MD", None, total_pages=10, rendered_pages=3,
                          page_lo=4, page_hi=6)
    assert "pages 4-6 of a 10-page" in p
    assert "CHUNKED" in p


def test_pages_for_budget_scales():
    md = "word " * 1000
    few = pages_for_budget(markdown=md, scope_md=None, total_pages=50, budget_tokens=2000)
    many = pages_for_budget(markdown=md, scope_md=None, total_pages=50, budget_tokens=40000)
    assert 1 <= few < many
    assert pages_for_budget(markdown=md, scope_md=None, total_pages=50,
                            budget_tokens=40000, cap=3) == 3


def test_config_defaults_load():
    from tools.benchmark.config import load_config
    cfg = load_config()
    assert cfg.render.dpi == 150
    assert cfg.grading.max_tokens == 2048
    assert cfg.grading.assume_context == 4096
    assert cfg.chunking.enabled is True
    assert cfg.grading.max_pages_or_none() is None          # 0 -> all
    assert cfg.chunking.pages_per_window_or_none() is None   # 0 -> auto
    assert cfg.weights.as_dict() == {"accuracy": 2.0, "coverage": 2.0, "clarity": 1.0}


def test_config_override_merges(tmp_path):
    """A partial --config TOML overrides only its keys; the rest fall back."""
    from tools.benchmark.config import load_config
    ov = tmp_path / "ov.toml"
    ov.write_text("[grading]\nmax_pages = 20\n\n[weights]\naccuracy = 3.0\n",
                  encoding="utf-8")
    cfg = load_config(ov)
    assert cfg.grading.max_pages == 20      # overridden
    assert cfg.grading.max_tokens == 2048   # untouched default
    assert cfg.weights.accuracy == 3.0      # overridden
    assert cfg.weights.clarity == 1.0       # untouched default


def test_config_rejects_unknown_key(tmp_path):
    """extra='forbid': an unknown key is a loud error, not silently ignored."""
    from pydantic import ValidationError

    from tools.benchmark.config import load_config
    ov = tmp_path / "bad.toml"
    ov.write_text("[grading]\nbogus = 1\n", encoding="utf-8")
    try:
        load_config(ov)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for unknown key")


def test_load_env_layers_benchmark_over_parser(tmp_path):
    """benchmark/.env wins per-variable but parser/.env fills what it leaves
    unset -- the layered load, not a single-file replace. (No monkeypatch: the
    direct `python test_benchmark.py` runner passes only one arg.)"""
    from tools.benchmark import cli

    here = tmp_path / "benchmark"
    parser = tmp_path / "parser"
    here.mkdir()
    parser.mkdir()
    # benchmark/.env sets only the model; parser/.env carries provider+base_url.
    (here / ".env").write_text("VLM_MODEL=judge-model\n", encoding="utf-8")
    (parser / ".env").write_text(
        "VLM_MODEL=parser-model\nVLM_PROVIDER=ollama\n"
        "VLM_BASE_URL=http://x:11434/v1\n", encoding="utf-8")

    saved_attrs = {k: getattr(cli, k) for k in ("_HERE", "_REPO_ROOT", "_PARSER_DIR")}
    saved_env = {v: os.environ.get(v) for v in ("VLM_MODEL", "VLM_PROVIDER", "VLM_BASE_URL")}
    try:
        cli._HERE, cli._REPO_ROOT, cli._PARSER_DIR = here, tmp_path, parser
        for v in saved_env:
            os.environ.pop(v, None)

        loaded = cli._load_env(None)
        assert loaded == [here / ".env", parser / ".env"]  # both, benchmark first
        assert os.environ["VLM_MODEL"] == "judge-model"      # benchmark/.env wins
        assert os.environ["VLM_PROVIDER"] == "ollama"        # filled from parser/.env
        assert os.environ["VLM_BASE_URL"] == "http://x:11434/v1"
    finally:
        for k, val in saved_attrs.items():
            setattr(cli, k, val)
        for v, val in saved_env.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def test_config_missing_key_fails():
    """No Python defaults: a missing required key blows up at validation."""
    from pydantic import ValidationError

    from tools.benchmark.config import BenchmarkConfig
    try:
        BenchmarkConfig.model_validate({"render": {"dpi": 150}})
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for missing sections")


def _run_all():
    import tempfile

    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        argcount = fn.__code__.co_argcount
        if argcount:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f"  ok  {name}")
    print(f"{passed} tests passed")


if __name__ == "__main__":
    _run_all()
