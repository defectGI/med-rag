"""N-04 (yariden kalma dayanikliligi): vectorize asamasi.

OLCULEN BUG (2026-08-20): `calistir()` her kapsami basariyla islerken
`state[kapsam.scope_id] = imza`yi yalniz BELLEKTE tutuyordu -- `save_state`
(zaten write-then-replace, atomic) yalniz dongu TAMAMEN BITINCE bir kez
cagriliyordu. Bir kosu 50 kapsamin 49'unu Qdrant'a basariyla yazip TAM 50.de
kesilirse `state.json` hala BOSTU: bir sonraki kosu o 49 kapsami da (yanlis
degil ama GEREKSIZ) yeniden embed ediyordu -- N-04'un "kaldigi yerden devam"
kabul kriterini karsilamiyordu.

Duzeltme: `save_state` artik HER basarili kapsamdan SONRA cagriliyor (aşama-
sonu commit noktasi, chunk/parse ile ayni disiplin).

Fake embedder + fake Qdrant store kullanilir (kok AGENTS.md: bu makinede
gercek LLM/GPU/servis cagrisi yok); "kill" gercek zamanlama yerine, ucuncu
kapsamin islenmesi sirasinda `KeyboardInterrupt` firlatan bir enjeksiyonla
simule edilir (gorev notunun onerdigi yontem).
"""

from __future__ import annotations

import json

import pytest

import medrag.pipeline.vectorize.cli as cli_mod


class _SahteEmbedder:
    name = "fake:test"

    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class _SahteStore:
    def __init__(self):
        self.ensure_calls = []
        self.replace_calls = []

    def ensure_collection(self, vector_size):
        self.ensure_calls.append(vector_size)

    def replace_scope(self, scope_id, points):
        self.replace_calls.append(scope_id)


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def sahte_store(monkeypatch):
    store = _SahteStore()
    monkeypatch.setattr(cli_mod.QdrantVectorStore, "from_env",
                        classmethod(lambda cls, **kw: store))
    return store


@pytest.fixture
def sahte_embedder(monkeypatch):
    embedder = _SahteEmbedder()
    monkeypatch.setattr(cli_mod, "embedder_from_env", lambda **kw: embedder)
    return embedder


def _write_chunks(path, doc_id):
    node = {"node_id": f"{doc_id}::c0", "doc_id": doc_id, "tree_level": 0,
            "text": "merhaba dunya", "heading_path": ["Giris"]}
    path.write_text(json.dumps({
        "doc_id": doc_id,
        "provenance": {"chunker_version": "1.0", "generated_at": "t",
                      "source": {"raw_sha256": "sha256:abc"}},
        "nodes": [node],
    }), encoding="utf-8")


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORIZE_INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("VECTORIZE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("QDRANT_URL", "http://x:6333")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")
    monkeypatch.setenv("VECTORIZE_PROGRESS", "off")


def test_kill_mid_scope_loop_checkpoints_every_completed_scope(
        monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    for doc_id in ("DOC1", "DOC2", "DOC3"):
        _write_chunks(tmp_path / "in" / f"{doc_id}.chunks.json", doc_id)
    _env(monkeypatch, tmp_path)

    # DOC3 islenirken kosu "oldurulur" -- gercek bir SIGKILL/Ctrl+C'nin
    # per-kapsam `except Exception`i ATLAYACAGI bir sinyal (KeyboardInterrupt,
    # Exception'dan turemez) enjekte edilir.
    orijinal_embed_scope = cli_mod._embed_scope

    def _oldur_ucuncude(kapsam, *, cfg, embedder, store):
        if kapsam.scope_id == "DOC3":
            raise KeyboardInterrupt("simulated kill mid-scope")
        return orijinal_embed_scope(kapsam, cfg=cfg, embedder=embedder, store=store)

    with monkeypatch.context() as mp:
        mp.setattr(cli_mod, "_embed_scope", _oldur_ucuncude)
        with pytest.raises(KeyboardInterrupt):
            cli_mod.main([])

    state_path = tmp_path / "out" / "state.json"
    assert state_path.is_file(), "DOC1/DOC2 basarili oldugu icin state.json ZATEN yazilmis olmali"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state) == {"DOC1", "DOC2"}, (
        "kesinti ANINDAKI kapsam (DOC3) commit edilmemis, ONCEKI ikisi (asama-sonu "
        "commit noktasi sayesinde) KAYBOLMAMIS olmali")
    assert sorted(sahte_store.replace_calls) == ["DOC1", "DOC2"]

    # --- bir sonraki kosu: kaldigi yerden devam -----------------------------
    sahte_store.replace_calls.clear()
    assert cli_mod.main([]) == 0
    assert sahte_store.replace_calls == ["DOC3"], (
        "DOC1/DOC2 degismedigi icin ATLANMALI, yalniz kesintide yarim kalan DOC3 islenmeli")
    state_final = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state_final) == {"DOC1", "DOC2", "DOC3"}
