from medrag.pipeline.vectorize.config import load_config


def test_default_config_yuklenir():
    cfg = load_config()
    assert cfg.embedding.batch_size >= 1
    assert cfg.qdrant.collection_name
    assert cfg.staleness.skip_unchanged is True


def test_embed_text_heading_prefix_acikken():
    cfg = load_config()
    assert cfg.embedding.prefix_heading_path is True
    metin = cfg.embed_text(heading_path=["A", "B"], text="gövde")
    assert metin == "A > B\n\ngövde"


def test_embed_text_heading_yoksa_dokunmaz():
    cfg = load_config()
    assert cfg.embed_text(heading_path=[], text="gövde") == "gövde"
