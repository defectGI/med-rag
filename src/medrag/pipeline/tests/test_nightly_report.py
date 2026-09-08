"""N-08/N-09 (I-19, I-21) kilitleme testleri + N-10/N-11/N-12/N-13.

N-08 kabul: ayni gece icin yapilandirilmis cikti (JSON) ve okunabilir sayfa
(markdown) AYNI sayilari gosteriyor -- iki ayri uretici degil, TEK
`NightlyReport` nesnesinden iki gorunum.

N-09 kabul: dokuz bolumun hepsi dolu; bir asama kasten bozuldugunda ilgili
satir Basarisizliklar bolumunde gorunuyor.

N-10 kabul (I-20): tek bir basarisizlik varsa `full_success` imkansiz;
gece penceresi tavanina carpilirsa sonuc HER ZAMAN `partial`.

N-11 kabul (I-22): her gecenin raporu tarihli bir dosyada; "dun bu sayi
kacti" gecmis dosyayi okuyarak cevaplanabiliyor.

N-12/N-13 kabul (I-24, I-25, K-64): durum uc kaynaktan (scan_status, turev
var mi, kayitli-surum vs guncel-surum) turer; surum artirilip kaynaga
dokunulmazsa "asama_degisti", tek dokuman yeniden islenirse digerleri
"taze" kalir.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from medrag.pipeline import nightly_report
from medrag.pipeline.nightly_report import (
    NIGHTLY_WINDOW_MAX_HOURS,
    ChunkSection,
    FailureEntry,
    IntegritySection,
    NightlyReport,
    OwnershipSection,
    ParseSection,
    ProductFeaturesSection,
    SkippedSection,
    SourceDiffSection,
    VectorizeSection,
    derive_derivative_status,
    load_nightly_report_dict,
    render_markdown,
    report_history_filename,
    save_nightly_report,
)


def _make_report(**overrides) -> NightlyReport:
    base = {
        "run_id": "2026-08-20T23-00-00Z",
        "started_at": datetime(2026, 8, 20, 23, 0, 0, tzinfo=UTC),
        "finished_at": datetime(2026, 8, 21, 0, 12, 0, tzinfo=UTC),
        "source_diff": SourceDiffSection(
            changed=["a.pdf"], added=["b.pdf"], removed=["c.pdf"], moved=["d.pdf -> e.pdf"]
        ),
        "skipped": SkippedSection(unchanged_count=42, unchanged_paths=["f.pdf", "g.pdf"]),
        "parse": ParseSection(processed_count=2, duration_seconds=123.5, model_call_count=6),
        "chunk": ChunkSection(produced_count=17, per_document_counts={"a.pdf": 9, "b.pdf": 8}, removed_count=3),
        "vectorize": VectorizeSection(
            points_written=17, points_deleted=3, total_points_after=5000, embedding_model="bge-m3"
        ),
        "ownership": OwnershipSection(deterministic_count=10, model_routed_count=5, unresolved_count=2),
        "product_features": ProductFeaturesSection(
            features_added=4, evidence_added=6, evidence_removed=1, features_removed_for_no_evidence=1
        ),
        "failures": [],
        "integrity": IntegritySection(
            pre_write_check_passed=True,
            post_write_check_passed=True,
            backup_completed=True,
            backup_location="/backups/2026-08-20",
        ),
    }
    base.update(overrides)
    return NightlyReport(**base)


def test_nine_sections_all_populated():
    """N-09: her bolum modelde ayri ve zorunlu alanlar olarak var."""
    report = _make_report()
    d = report.to_json_dict()
    for section in (
        "source_diff",
        "skipped",
        "parse",
        "chunk",
        "vectorize",
        "ownership",
        "product_features",
        "failures",
        "integrity",
    ):
        assert section in d


def test_json_and_markdown_agree_on_numbers():
    """N-08 kabul: JSON ve markdown ayni sayilari gosteriyor."""
    report = _make_report()
    d = report.to_json_dict()
    md = render_markdown(report)

    assert str(d["skipped"]["unchanged_count"]) in md
    assert str(d["parse"]["processed_count"]) in md
    assert str(d["parse"]["duration_seconds"]) in md
    assert str(d["parse"]["model_call_count"]) in md
    assert str(d["chunk"]["produced_count"]) in md
    assert str(d["chunk"]["removed_count"]) in md
    assert str(d["vectorize"]["points_written"]) in md
    assert str(d["vectorize"]["points_deleted"]) in md
    assert str(d["vectorize"]["total_points_after"]) in md
    assert d["vectorize"]["embedding_model"] in md
    assert str(d["ownership"]["deterministic_count"]) in md
    assert str(d["ownership"]["model_routed_count"]) in md
    assert str(d["ownership"]["unresolved_count"]) in md
    assert str(d["product_features"]["features_added"]) in md
    assert str(d["product_features"]["evidence_added"]) in md
    assert str(d["product_features"]["evidence_removed"]) in md
    assert str(d["product_features"]["features_removed_for_no_evidence"]) in md
    for path in d["source_diff"]["changed"] + d["source_diff"]["added"] + d["source_diff"]["removed"] + d[
        "source_diff"
    ]["moved"]:
        assert path in md


def test_deliberately_broken_stage_shows_up_in_failures_section():
    """N-09 kabul: bir asama kasten bozulunca ilgili satir Basarisizliklar'da."""
    report = _make_report(
        failures=[
            FailureEntry(
                file="broken.pdf",
                stage="parse",
                reason="model_timeout",
                error_text="Ollama request timed out after 120s",
            )
        ]
    )
    md = render_markdown(report)
    d = report.to_json_dict()

    assert len(d["failures"]) == 1
    assert "broken.pdf" in md
    assert "parse" in md
    assert "model_timeout" in md
    assert "Ollama request timed out after 120s" in md
    # "yok" placeholder should NOT appear once there is a real failure
    assert "(yok)" not in md.split("## Butunluk")[0].split("## Basarisizliklar")[1]


def test_no_failures_renders_placeholder():
    report = _make_report(failures=[])
    md = render_markdown(report)
    section = md.split("## Basarisizliklar")[1].split("## Butunluk")[0]
    assert "(yok)" in section


def test_integrity_section_is_interface_for_n05_and_o08():
    """Butunluk bolumu N-05 bitmeden de dolu bir arayuz olarak calisir."""
    report = _make_report(
        integrity=IntegritySection(
            pre_write_check_passed=False,
            post_write_check_passed=False,
            backup_completed=False,
            backup_location=None,
            notes="N-05 henuz entegre edilmedi (TODO)",
        )
    )
    md = render_markdown(report)
    d = report.to_json_dict()

    assert d["integrity"]["backup_completed"] is False
    assert d["integrity"]["backup_location"] is None
    assert "BASARISIZ" in md
    assert "TODO" in md


def test_extra_field_rejected():
    """`extra=\"forbid\"` -- semaya olmayan alan sessizce yutulmaz."""
    with pytest.raises(ValidationError):
        SourceDiffSection(changed=[], added=[], removed=[], moved=[], unexpected_field=1)


def test_round_trip_json_str_is_valid_and_matches_dict():
    import json

    report = _make_report()
    parsed = json.loads(report.to_json_str())
    assert parsed == report.to_json_dict()


# ---------------------------------------------------------------------------
# N-10 (I-20): uc degerli sonuc.
# ---------------------------------------------------------------------------


def test_no_failures_and_normal_duration_is_full_success():
    report = _make_report()
    assert report.outcome() == "full_success"
    assert report.to_json_dict()["outcome"] == "full_success"


def test_single_deliberately_broken_document_yields_partial_and_shows_in_failures():
    """N-10 kabul senaryosu: tek bir dokumanin parse'ini kasten boz."""
    report = _make_report(
        parse=ParseSection(processed_count=1, duration_seconds=10.0, model_call_count=1),
        failures=[
            FailureEntry(
                file="broken.pdf",
                stage="parse",
                reason="model_timeout",
                error_text="Ollama request timed out after 120s",
            )
        ],
    )
    assert report.outcome() == "partial"
    d = report.to_json_dict()
    assert d["outcome"] == "partial"
    assert any(f["file"] == "broken.pdf" for f in d["failures"])


def test_failures_with_zero_progress_yield_failed():
    report = _make_report(
        parse=ParseSection(processed_count=0, duration_seconds=1.0, model_call_count=0),
        chunk=ChunkSection(produced_count=0, per_document_counts={}, removed_count=0),
        vectorize=VectorizeSection(
            points_written=0, points_deleted=0, total_points_after=0, embedding_model="bge-m3"
        ),
        product_features=ProductFeaturesSection(
            features_added=0, evidence_added=0, evidence_removed=0, features_removed_for_no_evidence=0
        ),
        failures=[
            FailureEntry(file="x.pdf", stage="scan", reason="io_error", error_text="disk full")
        ],
    )
    assert report.outcome() == "failed"


def test_window_ceiling_breach_is_always_partial_even_without_failures():
    started = datetime(2026, 8, 20, 23, 0, 0, tzinfo=UTC)
    finished = started + timedelta(hours=NIGHTLY_WINDOW_MAX_HOURS)
    report = _make_report(started_at=started, finished_at=finished, failures=[])
    assert report.outcome() == "partial"


def test_outcome_appears_in_rendered_markdown():
    report = _make_report()
    md = render_markdown(report)
    assert "full_success" in md


# ---------------------------------------------------------------------------
# N-11 (I-22): rapor gecmisi.
# ---------------------------------------------------------------------------


def test_save_and_load_report_history_by_date(tmp_path: Path):
    report = _make_report(
        started_at=datetime(2026, 8, 19, 23, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, 0, 30, 0, tzinfo=UTC),
    )
    path = save_nightly_report(report, reports_dir=tmp_path)

    assert path.name == "nightly_2026-08-19.json"
    assert path == tmp_path / "nightly_2026-08-19.json"

    # "dun bu sayi kacti" -- gecmis dosyayi okuyarak cevaplanabiliyor
    loaded = load_nightly_report_dict("2026-08-19", reports_dir=tmp_path)
    assert loaded is not None
    assert loaded["chunk"]["produced_count"] == report.chunk.produced_count
    assert loaded["outcome"] == report.outcome()


def test_load_missing_date_returns_none(tmp_path: Path):
    assert load_nightly_report_dict("2020-01-01", reports_dir=tmp_path) is None


def test_report_history_filename_uses_started_at_date():
    started = datetime(2026, 8, 20, 23, 0, 0, tzinfo=UTC)
    assert report_history_filename(started) == "nightly_2026-08-20.json"


def test_saved_reports_are_never_pruned(tmp_path: Path):
    """N-11: sinir yok, otomatik silme YOK -- iki farkli gunun raporu birlikte durur."""
    r1 = _make_report(
        started_at=datetime(2026, 8, 18, 23, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 19, 0, 0, 0, tzinfo=UTC),
    )
    r2 = _make_report(
        started_at=datetime(2026, 8, 19, 23, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, 0, 0, 0, tzinfo=UTC),
    )
    save_nightly_report(r1, reports_dir=tmp_path)
    save_nightly_report(r2, reports_dir=tmp_path)
    assert (tmp_path / "nightly_2026-08-18.json").exists()
    assert (tmp_path / "nightly_2026-08-19.json").exists()


# ---------------------------------------------------------------------------
# N-12/N-13 (I-24, I-25, K-64): durum turetme.
# ---------------------------------------------------------------------------


def test_deleted_source_yields_kaynak_yok_regardless_of_version():
    status = derive_derivative_status(
        scan_status="DELETED", derivative_exists=True, recorded_version="9.9.9", current_version="1.0.0"
    )
    assert status == "kaynak_yok"


def test_modified_source_yields_kaynak_degisti():
    status = derive_derivative_status(
        scan_status="MODIFIED", derivative_exists=True, recorded_version="1.0.0", current_version="1.0.0"
    )
    assert status == "kaynak_degisti"


def test_new_source_yields_kaynak_degisti():
    status = derive_derivative_status(
        scan_status="NEW", derivative_exists=False, recorded_version=None, current_version="1.0.0"
    )
    assert status == "kaynak_degisti"


def test_missing_derivative_yields_uretilmemis():
    status = derive_derivative_status(
        scan_status="UNCHANGED", derivative_exists=False, recorded_version=None, current_version="1.0.0"
    )
    assert status == "uretilmemis"


def test_stage_version_bump_without_touching_source_yields_asama_degisti():
    """N-13 dogrulamasi: surumu artir, hicbir dosyaya dokunma -> bayat, 'asama_degisti'."""
    status = derive_derivative_status(
        scan_status="UNCHANGED", derivative_exists=True, recorded_version="1.0.0", current_version="1.0.1"
    )
    assert status == "asama_degisti"


def test_matching_version_and_unchanged_source_is_taze():
    status = derive_derivative_status(
        scan_status="UNCHANGED", derivative_exists=True, recorded_version="1.0.1", current_version="1.0.1"
    )
    assert status == "taze"


def test_reprocessing_one_document_leaves_others_taze():
    """I-25: tek dokuman yeniden islenir, digerleri taze kalir."""
    reprocessed = derive_derivative_status(
        scan_status="MODIFIED", derivative_exists=True, recorded_version="1.0.0", current_version="1.0.0"
    )
    untouched = derive_derivative_status(
        scan_status="UNCHANGED", derivative_exists=True, recorded_version="1.0.0", current_version="1.0.0"
    )
    assert reprocessed == "kaynak_degisti"
    assert untouched == "taze"


def test_derivative_status_can_be_attached_to_report():
    report = _make_report(derivative_status={"a.pdf": {"parse": "taze", "chunk": "asama_degisti"}})
    d = report.to_json_dict()
    assert d["derivative_status"]["a.pdf"]["chunk"] == "asama_degisti"


def test_default_derivative_status_is_empty_dict():
    report = _make_report()
    assert report.derivative_status == {}


def test_current_pipeline_versions_reads_from_config():
    from medrag.pipeline.nightly_report import current_pipeline_versions

    versions = current_pipeline_versions()
    assert set(versions) == {"parser", "chunker", "facts", "vectorize"}
    assert all(isinstance(v, str) and v for v in versions.values())


def test_outcome_olculmemis_alanlarla_cokmez():
    """`outcome()` yalniz `failures` DOLUYKEN ilerleme kontrolune giriyor --
    yani rapora en cok ihtiyac duyulan anda. N-21 bazi alanlari Optional
    yapinca (`features_added` HER kosuda None: load_to_db gece kosusuna
    bagli degil) bu karsilastirma `None > 0` ile TypeError atiyordu; hata
    da rapor yazma yolunun ICINDE oldugu icin hem rapor hem teshis
    kaybolurdu. "Olculmedi" ilerleme kaniti sayilmaz, 0 gibi ele alinir."""
    rapor = _make_report()
    rapor.failures.append(FailureEntry(file="-", stage="scan", reason="RuntimeError",
                                        error_text="ornek"))
    rapor.parse.processed_count = 0
    rapor.chunk.produced_count = 0
    rapor.vectorize.points_written = 0
    rapor.product_features.features_added = None

    assert rapor.outcome() == "failed"

    rapor.parse.processed_count = 5
    assert rapor.outcome() == "partial"


# --- K-101: uretimdeki extractor_version bicimi semver gibi karsilastiriliyordu

def test_prompt_hash_only_version_is_not_silently_behind():
    """OLCULEN BUG (2026-08-27 gece kosusu): `spec_value.extractor_version`a
    `"chunk_v1:<sha12>"` yaziliyor, `staleness_audit` olcut (c) bunu
    `facts_version` ile semver olarak karsilastiriyordu. `_version_tuple`
    nokta icermeyen, sayisal olmayan tek parcayi `(0,)` gorup `(0,) < (1,0,0)`
    donuyor -> HER satir, HER gece "bayat". Kosuda olculen zarar: bayatlik
    kapisi 222 urun secti (olcut (c) tek basina 217), artimli kosu FIILEN
    yoktu."""
    assert not nightly_report._recorded_version_is_behind("chunk_v1:a1b2c3d4e5f6", "1.0.0")


def test_build_metadata_is_ignored_in_comparison():
    """K-101 sonrasi uretim bicimi: `"<facts_version>+<PROMPT_VERSION>"`.
    `+` sonrasi siralamaya GIRMEZ (semver kurali) -- prompt hash'i degisince
    surum "geride" sayilmamali, yalniz `facts_version` konusur."""
    assert not nightly_report._recorded_version_is_behind("1.0.0+chunk_v1:aaaa", "1.0.0")
    assert not nightly_report._recorded_version_is_behind("1.0.0+chunk_v1:bbbb", "1.0.0")
    assert nightly_report._recorded_version_is_behind("1.0.0+chunk_v1:aaaa", "1.1.0")
    assert not nightly_report._recorded_version_is_behind("1.1.0+chunk_v1:aaaa", "1.0.0")


def test_semver_comparison_still_works():
    """Fix, olcut (c)'nin ASIL isini bozmamali: gercek bir surum gerideyse
    HALA bayat sayilmali, `recorded is None` HALA True."""
    assert nightly_report._recorded_version_is_behind("0.9.0", "1.0.0")
    assert nightly_report._recorded_version_is_behind(None, "1.0.0")
    assert not nightly_report._recorded_version_is_behind("1.0.0", "1.0.0")
