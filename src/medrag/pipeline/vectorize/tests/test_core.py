from medrag.pipeline.vectorize.core import ChunkSet


def _raw_set(**over):
    base = {
        "doc_id": "DOC1",
        "provenance": {
            "chunker_version": "1.4.0",
            "generated_at": "2026-07-22T00:00:00+03:00",
            "source": {"raw_sha256": "sha256:abc"},
            "raptor": {"mode": "real", "model": "ollama:qwen2.5:32b",
                      "prompt_version": "2"},
        },
        "nodes": [
            {"node_id": "DOC1::c0", "doc_id": "DOC1", "tree_level": 0,
             "text": "merhaba"},
        ],
    }
    base.update(over)
    return base


def test_bilinmeyen_alan_tolere_edilir():
    # chunker'ın tam şeması çok daha fazla alan taşır (flex_*, split_*, ...);
    # extra="ignore" olmasaydı bu doğrulama patlardı.
    raw = _raw_set()
    raw["nodes"][0]["flex_applied"] = False
    raw["nodes"][0]["cross_refs"] = []
    raw["gelecek_bir_alan"] = "henüz bilmediğimiz bir şey"
    cs = ChunkSet.model_validate(raw)
    assert cs.doc_id == "DOC1"
    assert cs.nodes[0].text == "merhaba"


def test_signature_generated_at_disinda_kalir():
    cs_a = ChunkSet.model_validate(_raw_set())
    cs_b = ChunkSet.model_validate(
        _raw_set(provenance={**_raw_set()["provenance"],
                            "generated_at": "2026-07-23T00:00:00+03:00"}))
    assert cs_a.signature(embedding_model="m") == cs_b.signature(embedding_model="m")


def test_signature_kaynak_hash_degisince_farklilasir():
    cs_a = ChunkSet.model_validate(_raw_set())
    raw_b = _raw_set()
    raw_b["provenance"]["source"]["raw_sha256"] = "sha256:def"
    cs_b = ChunkSet.model_validate(raw_b)
    assert cs_a.signature(embedding_model="m") != cs_b.signature(embedding_model="m")


def test_signature_farkli_embedding_modeli_farklilasir():
    cs = ChunkSet.model_validate(_raw_set())
    assert cs.signature(embedding_model="a") != cs.signature(embedding_model="b")
