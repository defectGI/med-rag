"""Format benchmark results for the terminal and for a JSON report file."""

from __future__ import annotations

import json
from pathlib import Path

from .judge import DIMENSION_TR, DIMENSIONS, Evaluation


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "#" * filled + "-" * (width - filled)


def _parse_models_note(evaluations: list[Evaluation]) -> str:
    """One line summarizing which model(s) PARSED the batch (distinct from
    the judge model, printed separately) -- read from each document's own IR
    JSON, if it had one. Collapses to a single line when every document
    agrees (the common case); says "(mixed" and shows the distinct sets
    otherwise, so a mixed-model batch is never silently averaged together
    as if it were one consistent run."""
    seen: dict[str, str] = {}  # rendered summary string -> that string (dedup)
    missing = False
    for ev in evaluations:
        if not ev.parse_models:
            missing = True
            continue
        rendered = ", ".join(f"{role}={info.get('model')}"
                             for role, info in ev.parse_models.items())
        seen[rendered] = rendered
    if not seen:
        return ""
    if len(seen) == 1 and not missing:
        return f"parsed with: {next(iter(seen))}"
    parts = list(seen)
    if missing:
        parts.append("unknown (no IR JSON)")
    return "parsed with (mixed across documents): " + " | ".join(parts)


def format_console(evaluations: list[Evaluation], skipped: list) -> str:
    """A compact human-readable summary of a run."""
    lines: list[str] = []
    note = _parse_models_note(evaluations)
    if note:
        lines.append(note)
        lines.append("")
    header = f"{'document':<40} " + " ".join(f"{DIMENSION_TR[d][:6]:>7}" for d in DIMENSIONS) + f"{'overall':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for ev in evaluations:
        scores = " ".join(f"{ev.dimensions[d].score:>7.1f}" for d in DIMENSIONS)
        doc = ev.doc_id if len(ev.doc_id) <= 40 else ev.doc_id[:37] + "..."
        note = "" if ev.rendered_pages == ev.total_pages else f"  ({ev.rendered_pages}/{ev.total_pages}p)"
        lines.append(f"{doc:<40} {scores} {ev.overall:>8.1f}{note}")

    if len(evaluations) > 1:
        lines.append("-" * len(header))
        n = len(evaluations)
        avg = {d: sum(e.dimensions[d].score for e in evaluations) / n for d in DIMENSIONS}
        avg_overall = sum(e.overall for e in evaluations) / n
        scores = " ".join(f"{avg[d]:>7.1f}" for d in DIMENSIONS)
        lines.append(f"{'AVERAGE (' + str(n) + ' docs)':<40} {scores} {avg_overall:>8.1f}")

    if skipped:
        lines.append("")
        lines.append(f"Skipped {len(skipped)} folder(s):")
        for s in skipped:
            lines.append(f"  - {s.where}: {s.reason}")
    return "\n".join(lines)


def build_report(evaluations: list[Evaluation], skipped: list, *,
                 scope_path: str | None, model: str | None) -> dict:
    report: dict = {
        "scope": scope_path,
        "model": model,
        "documents": [ev.to_dict() for ev in evaluations],
        "skipped": [{"folder": str(s.where), "reason": s.reason} for s in skipped],
    }
    if evaluations:
        n = len(evaluations)
        report["aggregate"] = {
            "count": n,
            "overall": round(sum(e.overall for e in evaluations) / n, 1),
            "scores": {d: round(sum(e.dimensions[d].score for e in evaluations) / n, 1)
                       for d in DIMENSIONS},
        }
    return report


def write_report(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
