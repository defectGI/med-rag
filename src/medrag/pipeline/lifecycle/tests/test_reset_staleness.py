"""_reset_staleness: forget sonrası vectorize'in "değişmemiş" deyip yeniden
gönmeyen oggetti için imza temizliği (notun cevaba girmemesi bug'ı)."""

from medrag.pipeline.lifecycle import runner
from medrag.pipeline.vectorize.layout import OutputLayout
from medrag.pipeline.vectorize.state import load_state, save_state

_IMZA = {"embedding": "model-x", "sig": 123}


def _write_state(root, state):
    out = OutputLayout(str(root))
    out.ensure()
    save_state(state, out.state_file)


def test_reset_staleness_removes_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORIZE_OUTPUT_DIR", str(tmp_path / "storage"))
    _write_state(tmp_path / "storage", {"docA": _IMZA, "docB": _IMZA})

    runner._reset_staleness("docA")

    state = load_state(tmp_path / "storage" / "state.json")
    assert "docA" not in state, "docA imzası silinmeli (yeniden gömülecek)"
    assert "docB" in state, "diğer dokümanlara dokunulmamalı"


def test_reset_staleness_noop_on_unknown_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORIZE_OUTPUT_DIR", str(tmp_path / "storage"))
    _write_state(tmp_path / "storage", {"docB": _IMZA})

    runner._reset_staleness("yok")

    assert load_state(tmp_path / "storage" / "state.json") == {"docB": _IMZA}


def test_reset_staleness_noop_on_missing_state(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORIZE_OUTPUT_DIR", str(tmp_path / "storage"))
    runner._reset_staleness("docA")  # state.json yok -> no-op, hata yok
    runner._reset_staleness("docA")  # tekrar çağrı da güvenli
