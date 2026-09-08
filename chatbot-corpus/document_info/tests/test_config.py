"""document_info config + classification tests (offline, no model/network).

Run: `python chatbot-corpus/document_info/tests/test_config.py`
(or pytest). document_info/ is added to sys.path so `config` and
`classify_documents` can be imported as sibling modules.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_DOC_INFO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DOC_INFO_DIR))

from config import DocumentInfoConfig, load_config  # noqa: E402


def test_defaults_load():
    cfg = load_config()
    # doc_type order is preserved (first match wins — order matters).
    assert [d.type for d in cfg.doc_types][:2] == ["CE_DECLARATION", "TECHNICAL_DRAWING"]
    assert ".TEKNIK DOSYALAR" in cfg.scan.exclude_dir_set()
    assert ".png" in cfg.scan.image_ext_set()
    assert cfg.scan.stp_ext_tuple() == (".stp", ".step")
    assert cfg.scan.junk_prefix_tuple() == ("~$", ".")
    # regex compiles and matches as expected.
    assert cfg.codes.compiled().findall("PATH/PN1141/x") == ["PN1141"]


def test_override_merges(tmp_path):
    ov = tmp_path / "ov.toml"
    ov.write_text('[scan]\njunk_names = ["desktop.ini"]\n', encoding="utf-8")
    cfg = load_config(ov)
    assert cfg.scan.junk_name_set() == {"desktop.ini"}      # overridden (list overwritten)
    assert ".png" in cfg.scan.image_ext_set()               # untouched default
    assert [d.type for d in cfg.doc_types][0] == "CE_DECLARATION"  # untouched


def test_rejects_unknown_key(tmp_path):
    from pydantic import ValidationError
    ov = tmp_path / "bad.toml"
    ov.write_text("[scan]\nbogus = 1\n", encoding="utf-8")
    try:
        load_config(ov)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for unknown key")


def test_rejects_bad_regex(tmp_path):
    from pydantic import ValidationError
    ov = tmp_path / "bad.toml"
    ov.write_text('[codes]\npattern = "DE[unterminated"\n', encoding="utf-8")
    try:
        load_config(ov)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for uncompilable regex")


def test_missing_section_fails():
    from pydantic import ValidationError
    try:
        DocumentInfoConfig.model_validate({"scan": {}})
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for missing sections")


def test_classify_helpers_use_injected_config():
    """is_junk / classify_doc_type behavior is preserved by the config."""
    import classify_documents as cd
    cfg = load_config()
    assert cd.is_junk("~$temp.docx", cfg.scan) is True
    assert cd.is_junk("Thumbs.db", cfg.scan) is True
    assert cd.is_junk("PN1127 REVISION_HISTORY.pdf", cfg.scan) is True
    assert cd.is_junk("PN1127 Datasheet.pdf", cfg.scan) is False
    assert cd.classify_doc_type("PN1127 DATASHEET.pdf", cfg) == "DATASHEET"
    assert cd.classify_doc_type("part.STP", cfg) == "STP"
    assert cd.classify_doc_type("front.PNG", cfg) == "PRODUCT_IMAGE"
    assert cd.classify_doc_type("mystery.xyz", cfg) is None


# -- rescan merge -------------------------------------------------------------


def _kayit(doc_id, rel_path, content_hash, **degisiklik):
    """A previously parsed record carried over from an earlier scan."""
    rec = {
        "identity": {"doc_id": doc_id, "file_name": rel_path.split("/")[-1],
                     "extension": ".pdf"},
        "location": {"rel_path": rel_path, "file_path": "C:/x/" + rel_path},
        "scan": {"content_hash": content_hash, "size_bytes": 10,
                 "last_modified_time": "2026-07-17T10:00:00+03:00",
                 "scan_status": "NEW"},
        "doc_type": "DATASHEET",
        "is_active": False,          # editorial decision: should not be shown
        "links": {"owner_ids": ["n_1"], "link_count": 1,
                  "resolution_method": "code_match"},
        "parse": {"parser": "run_parse_pipeline.py", "parser_version": "1.4.0",
                  "status": "SUCCESS", "parsed_from_hash": content_hash,
                  "parsed_json_path": "out/a.json",
                  "last_parsed": "2026-07-17T10:05:00+03:00",
                  "error": None, "stages": None},
        "chunk": {"status": "SUCCESS", "chunked_at": "2026-07-17T10:09:00+03:00",
                  "chunked_from_hash": content_hash,
                  "chunks_path": "out/a.chunks.json", "chunker_version": "1.0.0"},
    }
    rec.update(degisiklik)
    return rec


def _onceki(tmp_path, *kayitlar):
    import json
    yol = tmp_path / "document_nodes.json"
    yol.write_text(json.dumps({"documents": list(kayitlar)}), encoding="utf-8")
    return str(yol)


def test_load_previous_dosya_yoksa_bos(tmp_path):
    import classify_documents as cd
    assert cd.load_previous(str(tmp_path / "yok.json")) == ({}, {})


def test_load_previous_bozuk_dosya_ilk_tarama_gibi(tmp_path):
    import classify_documents as cd
    yol = tmp_path / "bozuk.json"
    yol.write_text("{bozuk", encoding="utf-8")
    assert cd.load_previous(str(yol)) == ({}, {})


def test_merge_kimligi_ve_pipeline_durumunu_korur(tmp_path):
    """Core of the merge contract: re-scanning does not regenerate doc_id,
    and an unchanged document is not re-parsed."""
    import classify_documents as cd
    onceki = _kayit("KALICI-ID", "a/DE1.pdf", "sha256:aaa")
    yeni = {
        "identity": {"doc_id": "TAZE-UUID", "file_name": "DE1.pdf",
                     "extension": ".pdf"},
        "location": {"rel_path": "a/DE1.pdf", "file_path": "C:/x/a/DE1.pdf"},
        "scan": {"content_hash": "sha256:aaa", "size_bytes": 10,
                 "last_modified_time": "2026-07-18T10:00:00+03:00",
                 "scan_status": "UNCHANGED"},
        "doc_type": "DATASHEET", "is_active": True,
        "links": {"owner_ids": ["n_1"], "link_count": 1,
                  "resolution_method": "code_match"},
        **cd.blank_pipeline_state(),
    }
    cd.merge_record(yeni, onceki)
    assert yeni["identity"]["doc_id"] == "KALICI-ID"     # identity preserved
    assert yeni["parse"]["status"] == "SUCCESS"          # no re-parse
    assert yeni["chunk"]["chunks_path"] == "out/a.chunks.json"
    assert yeni["is_active"] is False                    # editorial decision preserved
    # Truth of location/hash is the disk -- the scan's value stays.
    assert yeni["scan"]["last_modified_time"] == "2026-07-18T10:00:00+03:00"


def test_scan_status_turetme():
    import classify_documents as cd
    onceki = _kayit("id1", "a/DE1.pdf", "sha256:aaa")
    by_path = {"a/DE1.pdf": onceki}
    # no match at all
    assert cd.scan_status_for(None, by_path, "sha256:zzz", "b/YENI.pdf") == "NEW"
    # same path, same content
    assert cd.scan_status_for(onceki, by_path, "sha256:aaa", "a/DE1.pdf") == "UNCHANGED"
    # same path, content changed
    assert cd.scan_status_for(onceki, by_path, "sha256:bbb", "a/DE1.pdf") == "MODIFIED"
    # same content, different path (matched via hash)
    assert cd.scan_status_for(onceki, by_path, "sha256:aaa", "b/DE1.pdf") == "MOVED"


def test_blank_pipeline_state_vector_blogu_icermez():
    """The unowned `vector` block was removed from the schema, in favour of `chunk`."""
    import classify_documents as cd
    durum = cd.blank_pipeline_state()
    assert "vector" not in durum
    assert durum["chunk"]["status"] == "PENDING"
    assert durum["parse"]["status"] == "PENDING"


def test_blank_pipeline_state_facts_blogu_pending_baslar():
    """The facts (pass 1) block starts as PENDING and is filled by run_facts_pipeline.py."""
    import classify_documents as cd
    durum = cd.blank_pipeline_state()
    assert durum["facts"] == {
        "status": "PENDING",
        "extracted_at": None,
        "extracted_from_hash": None,
        "extractor_version": None,
        "prompt_version": None,
    }


def test_merge_facts_blogunu_da_korur():
    """Re-scanning must also preserve the facts pipeline's state -- the
    same protection already given to the chunk block."""
    import classify_documents as cd
    onceki = _kayit("KALICI-ID", "a/DE1.pdf", "sha256:aaa")
    onceki["facts"] = {
        "status": "SUCCESS", "extracted_at": "2026-07-29T10:00:00+03:00",
        "extracted_from_hash": "sha256:aaa", "extractor_version": "fake",
        "prompt_version": "1",
    }
    yeni = {
        "identity": {"doc_id": "TAZE-UUID", "file_name": "DE1.pdf", "extension": ".pdf"},
        "location": {"rel_path": "a/DE1.pdf", "file_path": "C:/x/a/DE1.pdf"},
        "scan": {"content_hash": "sha256:aaa", "size_bytes": 10,
                 "last_modified_time": "2026-07-18T10:00:00+03:00",
                 "scan_status": "UNCHANGED"},
        "doc_type": "DATASHEET", "is_active": True,
        "links": {"owner_ids": ["n_1"], "link_count": 1,
                  "resolution_method": "code_match"},
        **cd.blank_pipeline_state(),
    }
    cd.merge_record(yeni, onceki)
    assert yeni["facts"]["status"] == "SUCCESS"
    assert yeni["facts"]["extractor_version"] == "fake"


# -- owner_ids leaf-product fan-out (deterministic) ---------------------------


def _mini_product_nodes():
    """Small synthetic tree: one subfamily (PN1288) with 2 leaf products
    under it + one direct-leaf product (PN1232). Used by the fan-out and
    _expanded tests."""
    return [
        {"node_id": "n_fam", "type": "family", "parent_id": None,
         "source_ref": "9", "family": "Nav", "subfamily": None,
         "subfamily_2": None, "product": None,
         "category": {"reference_code": None}},
        {"node_id": "n_sub", "type": "subfamily", "parent_id": "n_fam",
         "source_ref": "9.1", "family": "Nav", "subfamily": "GPS Family",
         "subfamily_2": None, "product": None,
         "category": {"reference_code": "PN1288"}},
        {"node_id": "n_p1", "type": "product", "parent_id": "n_sub",
         "source_ref": "9.1.1", "family": "Nav", "subfamily": "GPS Family",
         "subfamily_2": None,
         "product": {"product_code": "PN1295", "variant_base": "PN1288",
                     "display_name": "A"}, "category": None},
        {"node_id": "n_p2", "type": "product", "parent_id": "n_sub",
         "source_ref": "9.1.2", "family": "Nav", "subfamily": "GPS Family",
         "subfamily_2": None,
         "product": {"product_code": "PN1302", "variant_base": "PN1288",
                     "display_name": "B"}, "category": None},
        {"node_id": "n_p3", "type": "product", "parent_id": "n_fam",
         "source_ref": "9.2", "family": "Nav", "subfamily": None,
         "subfamily_2": None,
         "product": {"product_code": "PN1232", "variant_base": None,
                     "display_name": "C"}, "category": None},
    ]


def _resolver(nodes):
    """Build load_whitelist_and_categories without going through the file."""
    import classify_documents as cd
    code_to_node = {}
    for n in nodes:
        if n["product"]:
            for code in (n["product"]["product_code"], n["product"]["variant_base"]):
                if code:
                    code_to_node.setdefault(code, n["node_id"])
        if n["category"] and n["category"]["reference_code"]:
            code_to_node.setdefault(n["category"]["reference_code"], n["node_id"])
    norm = lambda t: __import__("re").sub(r"[^A-Z0-9]+", "", t.upper()) if t else None
    cat = {}
    for n in nodes:
        if n["type"] in ("family", "subfamily", "subfamily_2"):
            own = n[n["type"]]
            if own:
                cat.setdefault(norm(own), n["node_id"])
    expand = cd.build_subtree_expander(nodes)
    import config
    pattern = config.load_config().codes.compiled()
    return cd, code_to_node, cat, norm, pattern, expand


def test_expander_leaf_product_kendini_dondurur():
    import classify_documents as cd
    expand = cd.build_subtree_expander(_mini_product_nodes())
    assert expand("n_p1") == ["n_p1"]          # already a leaf product
    assert expand("n_p3") == ["n_p3"]


def test_expander_kategoriyi_tum_alt_agaca_yayar():
    import classify_documents as cd
    expand = cd.build_subtree_expander(_mini_product_nodes())
    # subfamily -> itself + all leaf products beneath, source_ref sorted
    assert expand("n_sub") == ["n_sub", "n_p1", "n_p2"]
    # family -> itself + intermediate subfamily node INCLUDED + all leaf product descendants
    assert expand("n_fam") == ["n_fam", "n_sub", "n_p1", "n_p2", "n_p3"]


def test_resolve_owner_ids_kod_eslesmesi_leaf():
    cd, code_to_node, cat, norm, pattern, expand = _resolver(_mini_product_nodes())
    owners, matched, method = cd.resolve_owner_ids(
        "x/PN1232/datasheet.pdf", code_to_node, cat, norm, pattern, expand)
    assert owners == ["n_p3"]
    assert matched == "n_p3"
    assert method == "code_match"           # no expansion -> no suffix


def test_resolve_owner_ids_kategori_kodu_fanout_expanded():
    cd, code_to_node, cat, norm, pattern, expand = _resolver(_mini_product_nodes())
    # PN1288 is a subfamily reference_code -> itself + all leaf products must be reached
    owners, matched, method = cd.resolve_owner_ids(
        "x/PN1288/brochure.pdf", code_to_node, cat, norm, pattern, expand)
    assert owners == ["n_sub", "n_p1", "n_p2"]  # category INCLUDED, full subtree
    assert matched == "n_sub"                   # raw matched category preserved
    assert method == "code_match_expanded"       # _expanded is consistent IN CODE


def test_resolve_owner_ids_eslesme_yok():
    cd, code_to_node, cat, norm, pattern, expand = _resolver(_mini_product_nodes())
    owners, matched, method = cd.resolve_owner_ids(
        "x/PN1001/foo.pdf", code_to_node, cat, norm, pattern, expand)
    assert owners == [] and matched is None and method is None


def _run_all():
    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if fn.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f"  ok  {name}")
    print(f"{passed} tests passed")


if __name__ == "__main__":
    _run_all()
