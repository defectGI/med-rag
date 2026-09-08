"""Output layout (`chunker/layout.py`) tests.

The layout is a CONTRACT: the chunker writes, the pipeline registry
records, `run_chunk_test` validates, the consumer reads. Names are
locked here — if a file name silently changes, the consumer side can't
find it anymore.
"""

from __future__ import annotations

from pathlib import Path

from medrag.pipeline.chunker.layout import DEFAULT_OUTPUT_ROOT, OutputLayout


def test_iki_sabit_dosya():
    out = OutputLayout("/kok")
    assert out.all_chunks_file == Path("/kok/all_chunks.json")
    assert out.all_combined_file == Path("/kok/all_combined.json")


def test_dosyalar_kokte_viz_ayri_klasorde():
    out = OutputLayout("/kok")
    assert out.all_chunks_file.parent == out.root
    assert out.all_combined_file.parent == out.root
    assert out.viz == Path("/kok/viz")


def test_ensure_viz_klasorunu_acar_ve_idempotenttir(tmp_path):
    out = OutputLayout(tmp_path / "storage")
    assert not out.viz.exists()
    out.ensure()
    assert out.viz.is_dir()
    out.ensure()  # second call must not raise
    assert out.viz.is_dir()


def test_ensure_mevcut_icerigi_silmez(tmp_path):
    """"Empty? fill; populated? leave alone": ensure() never cleans up."""
    out = OutputLayout(tmp_path / "storage").ensure()
    (out.viz / "d1.tree.html").write_text("<html></html>", encoding="utf-8")
    out.ensure()
    assert (out.viz / "d1.tree.html").exists()


def test_default_kok():
    assert DEFAULT_OUTPUT_ROOT == "./storage"