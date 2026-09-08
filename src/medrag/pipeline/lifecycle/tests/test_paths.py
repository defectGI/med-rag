"""paths.py: zorunlu env'ler, notlar/kuyruk/durum varsayılanları."""

import pytest

from medrag.pipeline.lifecycle import paths


@pytest.fixture()
def clear_env(monkeypatch):
    for v in ("BELGELER_DIR", "DOCUMENT_NODES_PATH", "PARSED_OUTPUT_DIR",
              "ISLER_DIR", "DURUM_DIR"):
        monkeypatch.delenv(v, raising=False)


def test_required_envs_raise(clear_env):
    for fn in (paths.belgeler_dir, paths.registry_path, paths.parsed_output_dir):
        with pytest.raises(paths.LifecycleConfigError):
            fn()


def test_notlar_dir_is_under_belgeler(monkeypatch, tmp_path):
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path / "BELGELER"))
    assert paths.notlar_dir() == (tmp_path / "BELGELER" / "notlar")


def test_isler_durum_defaults_are_siblings(monkeypatch, tmp_path):
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path / "korpus" / "BELGELER"))
    assert paths.isler_dir() == tmp_path / "korpus" / "isler"
    assert paths.durum_dir() == tmp_path / "korpus" / "durum"


def test_isler_durum_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path / "BELGELER"))
    monkeypatch.setenv("ISLER_DIR", str(tmp_path / "özel" / "isler"))
    monkeypatch.setenv("DURUM_DIR", str(tmp_path / "özel" / "durum"))
    assert paths.isler_dir() == (tmp_path / "özel" / "isler").resolve()
    assert paths.durum_dir() == (tmp_path / "özel" / "durum").resolve()
