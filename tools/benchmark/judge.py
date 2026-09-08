"""LLM-as-judge scoring of a parsed Markdown against its source PDF.

One vision call per document: the rendered PDF pages are the ground truth, the
Markdown is the candidate, and an optional scope narrows *what* is graded. The
model returns a per-dimension score (accuracy / coverage / clarity) with
reasoning; the overall is computed here from configurable weights, never asked
of the model, so the aggregate is deterministic given the sub-scores.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Turkish name -> internal key. The user speaks of dogruluk / kapsayicilik /
# aciklik; keys stay ASCII for JSON and code.
DIMENSIONS = ("accuracy", "coverage", "clarity")
DIMENSION_TR = {"accuracy": "dogruluk", "coverage": "kapsayicilik", "clarity": "aciklik"}

_SYSTEM = """\
You are a meticulous evaluator grading how faithfully a Markdown document
reproduces a source PDF. The PDF was converted to Markdown by an automated
document parser. The rendered PDF pages (attached as images) are the GROUND
TRUTH; the Markdown is the candidate to be graded against them.

The user message below carries an EVALUATION SCOPE. It OVERRIDES the
dimension definitions below wherever they conflict: if the scope places a
category of content or issue out of bounds, that must not affect ANY of the
three scores -- accuracy, coverage, AND clarity are all subject to the
scope, not just coverage. Before penalizing anything, check it against the
scope first; when in doubt whether something is in scope, do not penalize it.

Grade three independent dimensions, each on a 0-100 integer scale:

- accuracy: Of the in-scope content that IS present in the Markdown, how
  correct is it versus the PDF? Penalize wrong numbers, garbled or
  mis-transcribed text, mislabeled or wrong-level headings, corrupted/
  misaligned table cells, and any hallucinated content the PDF does not
  contain -- but only for content the scope covers. Do NOT penalize omissions
  here (that is coverage), and do NOT penalize anything the scope excludes
  (e.g. a limitation the scope explicitly says is expected behavior).

- coverage: How much of the PDF's in-scope content made it into the Markdown?
  Penalize dropped text, missing headings/lists/tables, and figure/image
  content that was lost -- only when the scope counts that content as in
  scope. Do NOT penalize content that is merely reworded, as long as it is
  present and correct, and do NOT penalize the absence of anything the scope
  places out of bounds.

- clarity: How clean and usable is the Markdown as a document -- correct
  heading hierarchy, well-formed tables, sensible reading order, and no leaked
  noise (page headers/footers, page numbers, artifacts). Judge structure and
  readability, not whether facts are right (that is accuracy) -- and, again,
  never penalize a structural quirk the scope explicitly excludes.

Scoring bands (apply to every dimension): 90-100 excellent, near-perfect;
75-89 good, minor issues; 60-74 usable, several noticeable problems; 40-59
poor, significant loss or errors; 0-39 unusable for this dimension.

Return ONLY a JSON object, no prose before or after, with exactly this shape:
{"accuracy":  {"score": <int 0-100>, "reasoning": "<1-3 sentences>", "issues": ["<short concrete issue>", ...]},
 "coverage":  {"score": <int 0-100>, "reasoning": "...", "issues": [...]},
 "clarity":   {"score": <int 0-100>, "reasoning": "...", "issues": [...]},
 "summary": "<one paragraph overall assessment>"}

Each `issues` entry is ONE concrete thing that pulled THIS dimension's score
down -- one finding per entry, no summaries or generalities ("some values are
wrong" is useless). Every entry MUST pin down WHERE and show the BEFORE ->
AFTER, in this shape:

  "<location>: PDF '<what the page shows>' -> output '<what the Markdown shows>'"

- location: the most specific anchor you can give -- page number, and the
  section heading / table name+cell (row x col) / list the content sits in
  (e.g. "Table 2 'Fees', row 3 col 2, p.4" or "heading 'Warranty', p.6").
- accuracy issue: quote the PDF's value/text and the wrong output value/text
  side by side (e.g. "...: PDF 'Toplam: 1.240' -> output 'Toplam: 1240'").
- coverage issue: quote the in-scope PDF content that is MISSING and mark the
  output side "(absent)" (e.g. "heading 'Warranty' + its paragraph, p.6: PDF
  present -> output (absent)").
- clarity issue: name the structural defect and where (e.g. "p.2: PDF is a
  2-column table -> output collapsed all cells into one line").

If you cannot point to a specific location or cannot show what the output
actually did, do not invent one -- leave that finding out. Use an empty list
when a dimension has no score-lowering issues."""

_DEFAULT_SCOPE = """\
No explicit scope was given. Grade the whole document: all body text, headings,
lists, tables, and the textual/figure content of images. Judge the Markdown as
a faithful, readable reproduction of everything in the PDF."""

# Rough (NOT exact) input-token heuristics, used only to warn before a call
# that looks likely to exceed a small context window -- real tokenization
# varies by model. This matters because Ollama's default context is often
# 4096 tokens, its OpenAI-compatible endpoint has no per-request way to raise
# it (see benchmark/README.md), and an over-budget prompt is truncated
# silently rather than rejected -- the judge would then score pages it never
# actually saw.
_CHARS_PER_TOKEN = 4       # rough English/Latin-script average
# One rendered PDF page image. 700 proved far too low in practice: a real
# 20-page grading call against nemotron3:33b came to 193655 actual prompt
# tokens vs ~153000 estimated -- a 27% shortfall that exceeds the 20% safety
# margin the chunk trigger leaves (0.8 * num_ctx), i.e. a "fits" verdict could
# still be a runtime 400. High-res OCR-oriented vision encoders spend several
# thousand tokens per page; overestimating merely makes windows a bit shorter.
_TOKENS_PER_IMAGE = 2500


def estimate_prompt_tokens(markdown: str, scope_md: str | None, n_images: int) -> int:
    """A rough estimate of one judge call's input token cost (text + images)."""
    text_chars = len(_SYSTEM) + len(_scope_block(scope_md)) + len(markdown)
    return text_chars // _CHARS_PER_TOKEN + n_images * _TOKENS_PER_IMAGE


def _scope_block(scope_md: str | None) -> str:
    if scope_md and scope_md.strip():
        return (
            "EVALUATION SCOPE -- grade STRICTLY with respect to the following. "
            "Anything this scope places out of bounds must not affect any score "
            "(neither reward nor penalize its presence or absence):\n\n"
            f"{scope_md.strip()}"
        )
    return _DEFAULT_SCOPE


def build_user_prompt(markdown: str, scope_md: str | None,
                      *, total_pages: int, rendered_pages: int,
                      page_lo: int | None = None, page_hi: int | None = None) -> str:
    """Assemble the text half of the judge turn (images are attached separately).

    `page_lo`/`page_hi` (1-based, inclusive) mark a chunked window: when set, the
    attached images and Markdown are one page-aligned SLICE of a larger document
    graded across several passes. The note then tells the judge to grade only
    this slice against only these pages -- content on other pages is graded in
    other passes, so its absence here is not a coverage gap."""
    windowed = page_lo is not None and page_hi is not None
    truncation = ""
    if windowed:
        truncation = (
            f"\n\nNOTE: this is a CHUNKED evaluation. The attached {rendered_pages} "
            f"image(s) are pages {page_lo}-{page_hi} of a {total_pages}-page document, "
            "and the Markdown below is the slice of the parser's output for exactly "
            "those pages. Grade ONLY this slice against ONLY these pages. Content that "
            "belongs to other pages is graded in separate passes -- do not penalize "
            "coverage for anything outside pages "
            f"{page_lo}-{page_hi}, and do not expect this slice to read as a complete "
            "standalone document (a section may start before or continue after it).")
    elif rendered_pages < total_pages:
        truncation = (
            f"\n\nNOTE: only the first {rendered_pages} of {total_pages} PDF pages "
            "are attached (page cap). Grade coverage/accuracy against the attached "
            "pages only; do not penalize the Markdown for pages you cannot see.")
    attached = (f"pages {page_lo}-{page_hi} of the source PDF, attached as "
                f"{rendered_pages} page image(s), in order."
                if windowed else
                f"The source PDF is attached as {rendered_pages} page image(s), in order.")
    return (
        f"{_scope_block(scope_md)}\n\n"
        f"{attached}"
        f"{truncation}\n\n"
        "=== CANDIDATE MARKDOWN (parser output) ===\n"
        f"{markdown}\n"
        "=== END CANDIDATE MARKDOWN ===\n\n"
        "Before scoring: re-read the EVALUATION SCOPE at the top of this message. "
        "Check every issue you are about to list against it -- do not let anything "
        "it places out of bounds reduce accuracy, coverage, or clarity, no matter "
        "which dimension it would otherwise seem to fall under. Now grade accuracy, "
        "coverage, and clarity as instructed and return the JSON object."
    )


@dataclass
class DimensionResult:
    score: float
    reasoning: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class Evaluation:
    doc_id: str
    dimensions: dict[str, DimensionResult]
    overall: float
    summary: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    rendered_pages: int = 0
    total_pages: int = 0
    prompt_tokens: int | None = None  # from the server's `usage` block, if it sent one
    # Which model(s)/context window PARSED this document (from its own IR
    # JSON's metadata.models, see llm.describe_configured_models) -- distinct
    # from the JUDGE model (that's report["model"], one per whole run). None
    # when no IR JSON sits next to the .md, or it predates this field.
    parse_models: dict | None = None
    # Per-window breakdown when this document was graded in page-aligned chunks
    # (see benchmark/chunk.py); None for a single-call grade. Each entry is
    # {"pages": [lo, hi], "overall": float, "scores": {dim: float}}. The
    # top-level scores above are the page-weighted aggregate of these.
    windows: list[dict] | None = None

    def to_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "overall": self.overall,
            "scores": {k: round(v.score, 1) for k, v in self.dimensions.items()},
            "weights": self.weights,
            "pages": {"rendered": self.rendered_pages, "total": self.total_pages},
            "prompt_tokens": self.prompt_tokens,
            "parse_models": self.parse_models,
            "summary": self.summary,
            "detail": {
                k: {"score": round(v.score, 1), "reasoning": v.reasoning,
                    "issues": v.issues}
                for k, v in self.dimensions.items()
            },
        }
        if self.windows is not None:
            d["windows"] = self.windows
        return d


class JudgeError(Exception):
    """Raised when the model reply cannot be parsed into scores."""


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (tolerates code fences/prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise JudgeError(f"no JSON object in model reply: {text[:300]!r}")
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"model reply was not valid JSON: {exc}: {candidate[:300]!r}")


def _coerce_score(raw) -> float:
    try:
        score = float(raw)
    except (TypeError, ValueError):
        raise JudgeError(f"non-numeric score: {raw!r}")
    # Clamp defensively; a model occasionally emits 0-10 or over-100.
    return max(0.0, min(100.0, score))


def parse_evaluation(reply: str, doc_id: str, weights: dict[str, float]) -> Evaluation:
    """Turn a raw model reply into a validated Evaluation with computed overall."""
    data = _extract_json(reply)
    dims: dict[str, DimensionResult] = {}
    for key in DIMENSIONS:
        block = data.get(key)
        if not isinstance(block, dict) or "score" not in block:
            raise JudgeError(f"missing/invalid '{key}' block in reply")
        issues = block.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        dims[key] = DimensionResult(
            score=_coerce_score(block["score"]),
            reasoning=str(block.get("reasoning", "")).strip(),
            issues=[str(i).strip() for i in issues if str(i).strip()],
        )
    total_w = sum(weights[k] for k in DIMENSIONS)
    if total_w <= 0:
        raise ValueError("dimension weights must sum to a positive number")
    overall = sum(dims[k].score * weights[k] for k in DIMENSIONS) / total_w
    return Evaluation(
        doc_id=doc_id,
        dimensions=dims,
        overall=round(overall, 1),
        summary=str(data.get("summary", "")).strip(),
        weights=dict(weights),
    )


def read_parse_models(ir_json) -> dict | None:
    """Best-effort read of `metadata.models` from a document's own IR JSON --
    which model(s)/context window actually PARSED it (see
    llm.describe_configured_models). Never raises: a missing/corrupt/older
    (pre-this-field) IR JSON just means "unknown", not a grading failure."""
    if ir_json is None:
        return None
    try:
        data = json.loads(Path(ir_json).read_text(encoding="utf-8"))
        return data.get("metadata", {}).get("models") or None
    except Exception:  # noqa: BLE001
        return None


def evaluate_document(client, *, doc_id: str, markdown: str, scope_md: str | None,
                      images: Sequence[tuple[str, bytes]], total_pages: int,
                      weights: dict[str, float], max_tokens: int = 2048,
                      ir_json=None) -> Evaluation:
    """Score one document: one vision call + parse. `client` is a VLMClient.
    `ir_json`, if given, is read (best-effort) for parse-time model metadata --
    never sent to the judge model, purely attached to the returned report."""
    if not images:
        raise JudgeError(f"{doc_id}: no PDF pages were rendered")
    user = build_user_prompt(markdown, scope_md,
                             total_pages=total_pages, rendered_pages=len(images))
    reply = client.complete_vision(system=_SYSTEM, user=user, images=list(images),
                                   max_tokens=max_tokens)
    ev = parse_evaluation(reply, doc_id, weights)
    ev.rendered_pages = len(images)
    ev.total_pages = total_pages
    usage = getattr(client, "last_usage", None) or {}
    ev.prompt_tokens = usage.get("prompt_tokens")
    ev.parse_models = read_parse_models(ir_json)
    return ev


def _aggregate_windows(doc_id: str, per_window: list[tuple[int, int, int, Evaluation]],
                       weights: dict[str, float], *, total_pages: int,
                       prompt_tokens: int | None, parse_models: dict | None) -> Evaluation:
    """Fold per-window Evaluations into one document Evaluation.

    `per_window` is (page_lo, page_hi, n_pages, Evaluation) per graded window.
    Each dimension score is the PAGE-WEIGHTED mean across windows (a 30-page
    window counts thirty times a 1-page one), so the aggregate matches what a
    single call over the whole document would weigh -- not a plain per-window
    mean that would let a tiny tail window swing the score. The overall is
    recomputed from `weights` on the aggregated dimension means, exactly as
    parse_evaluation does, so single-call and chunked grades are comparable.
    Issues and summaries are kept per window, page-tagged, so nothing is lost."""
    total_w_pages = sum(n for _, _, n, _ in per_window) or 1
    dims: dict[str, DimensionResult] = {}
    for key in DIMENSIONS:
        weighted = sum(ev.dimensions[key].score * n
                       for _, _, n, ev in per_window) / total_w_pages
        issues: list[str] = []
        for lo, hi, _, ev in per_window:
            issues.extend(f"[p{lo}-{hi}] {i}" for i in ev.dimensions[key].issues)
        dims[key] = DimensionResult(score=weighted,
                                    reasoning=f"page-weighted mean over {len(per_window)} window(s)",
                                    issues=issues)
    total_wt = sum(weights[k] for k in DIMENSIONS)
    overall = sum(dims[k].score * weights[k] for k in DIMENSIONS) / total_wt
    summary = " ".join(f"[p{lo}-{hi}] {ev.summary}" for lo, hi, _, ev in per_window
                       if ev.summary)
    return Evaluation(
        doc_id=doc_id,
        dimensions={k: DimensionResult(round(v.score, 1), v.reasoning, v.issues)
                    for k, v in dims.items()},
        overall=round(overall, 1),
        summary=summary,
        weights=dict(weights),
        rendered_pages=sum(n for _, _, n, _ in per_window),
        total_pages=total_pages,
        prompt_tokens=prompt_tokens,
        parse_models=parse_models,
        windows=[{"pages": [lo, hi], "overall": ev.overall,
                  "scores": {k: round(ev.dimensions[k].score, 1) for k in DIMENSIONS}}
                 for lo, hi, _, ev in per_window],
    )


def evaluate_windows(client, *, doc_id: str, windows, scope_md: str | None,
                     total_pages: int, weights: dict[str, float],
                     max_tokens: int = 2048, ir_json=None,
                     on_window=None) -> Evaluation:
    """Score a document in page-aligned windows: one vision call per window, then
    a page-weighted aggregate (see `_aggregate_windows`). `windows` is a list of
    benchmark.chunk.PageWindow. `on_window(index, total, window, sub_eval)` is an
    optional progress callback. Raises JudgeError if a window has no images."""
    if not windows:
        raise JudgeError(f"{doc_id}: no windows to grade")
    per_window: list[tuple[int, int, int, Evaluation]] = []
    tok_sum = 0
    tok_seen = False
    for idx, win in enumerate(windows):
        if not win.images:
            raise JudgeError(f"{doc_id}: window p{win.page_lo}-{win.page_hi} has no images")
        user = build_user_prompt(win.markdown, scope_md, total_pages=total_pages,
                                 rendered_pages=len(win.images),
                                 page_lo=win.page_lo, page_hi=win.page_hi)
        reply = client.complete_vision(system=_SYSTEM, user=user,
                                       images=list(win.images), max_tokens=max_tokens)
        sub = parse_evaluation(reply, f"{doc_id}#p{win.page_lo}-{win.page_hi}", weights)
        sub.rendered_pages = len(win.images)
        sub.total_pages = total_pages
        usage = getattr(client, "last_usage", None) or {}
        pt = usage.get("prompt_tokens")
        if pt is not None:
            tok_sum += pt
            tok_seen = True
        per_window.append((win.page_lo, win.page_hi, win.n_pages, sub))
        if on_window is not None:
            on_window(idx, len(windows), win, sub)
    return _aggregate_windows(
        doc_id, per_window, weights, total_pages=total_pages,
        prompt_tokens=tok_sum if tok_seen else None,
        parse_models=read_parse_models(ir_json))
