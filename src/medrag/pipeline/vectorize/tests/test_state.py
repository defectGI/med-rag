from medrag.pipeline.vectorize.state import is_unchanged, load_state, save_state


def test_load_state_dosya_yoksa_bos_sozluk(tmp_path):
    assert load_state(tmp_path / "yok.json") == {}


def test_save_sonra_load_ayni_veriyi_dondurur(tmp_path):
    path = tmp_path / "state.json"
    save_state({"DOC1": {"a": 1}}, path)
    assert load_state(path) == {"DOC1": {"a": 1}}


def test_bozuk_dosya_bos_sozluk_donduru(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{bozuk", encoding="utf-8")
    assert load_state(path) == {}


def test_is_unchanged():
    state = {"DOC1": {"a": 1}}
    assert is_unchanged(state, "DOC1", {"a": 1}) is True
    assert is_unchanged(state, "DOC1", {"a": 2}) is False
    assert is_unchanged(state, "DOC2", {"a": 1}) is False
