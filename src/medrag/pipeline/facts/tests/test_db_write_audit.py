"""O-09 (I-15, K-17): `specs.db`ye deger yazan TEK akisin `load_to_db.py`
oldugunu -- `build_facts_db.py` HARIC, KARAR-016'nin ongordugu ikinci asama --
IMPORT/AST tabanli kilitler. Grep DEGIL: `db_write_audit.find_sql_write_files`
her dosyayi `ast.parse` eder, boylece "INSERT" gecen bir YORUM satirini
yanlislikla yazici saymaz (bkz. asagidaki `test_comment_mentioning_insert_is_not_a_false_positive`).

Bu testler `load_to_db.py`/`build_facts_db.py`nin KENDILERINI import ETMEZ
(ikisi de su an K-71 nedeniyle import-time'da patliyor) -- yalniz kaynak
metinlerini AST olarak okur, o yuzden BLOKEDEN bagimsiz calisirlar."""
from __future__ import annotations

import textwrap

from medrag.pipeline.facts import db_write_audit


def test_only_the_decreed_writers_touch_specs_db_in_the_real_tree():
    """Gercek `src/medrag/pipeline/facts/` agacinda deger-yazan SQL iceren
    dosyalarin kumesi TAM OLARAK {load_to_db.py, build_facts_db.py,
    forget_source.py} -- ne eksik (kontrat kirilmis olabilir), ne fazla
    (dorduncu bir yazici sizmis olabilir). Ucuncusu (`forget_source.py`)
    K-78/O-11'in BILINCLI silme istisnasi -- bkz. `db_write_audit`
    modul docstring'i."""
    writers = db_write_audit.find_sql_write_files()
    writer_basenames = {path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path in writers}

    assert writer_basenames == db_write_audit.ALLOWED_WRITER_FILENAMES, (
        f"beklenen yazici kumesi {db_write_audit.ALLOWED_WRITER_FILENAMES!r}, "
        f"bulunan {writer_basenames!r} -- ayrinti: {writers!r}"
    )


def test_load_to_db_and_build_facts_db_write_different_column_sets():
    """I-15/K-17'nin ayrimini somutlastirir: iki yazicinin gordugu INSERT/
    UPDATE hedef sutunlari FARKLI kumeler -- ayni satira YARISAN iki yazici
    DEGIL, KARAR-016'nin ongordugu iki AYRI asama."""
    writers = db_write_audit.find_sql_write_files()
    load_to_db_snippets = next(v for k, v in writers.items() if k.endswith("load_to_db.py"))
    build_facts_db_snippets = next(v for k, v in writers.items() if k.endswith("build_facts_db.py"))

    # load_to_db.py: deger sutunlarina yazar (num_value/text_value/... status).
    assert any("status=" in s for s in load_to_db_snippets)
    assert any("evidence" in s.lower() or "extractor" in s.lower() for s in load_to_db_snippets)
    # build_facts_db.py: iskelet/bootstrap satirlarini yazar (raw_text/bool_value
    # sabit degerlerle "present" isaretlenir, gercek deger sutunlarina DEGIL).
    assert any("spec_value" in s.lower() for s in build_facts_db_snippets)


def test_a_third_module_writing_to_spec_value_is_detected(tmp_path):
    """Sema yerine AKIS kasten bozulur: `facts/` agacina UCUNCU, sahte bir
    'yazici' modul eklenir -- kilit BUNU YAKALAMALI (bu grubun 'yazma
    reddedilir' kanitinin akis-tarafi karsiligi, O-08'in DB-tarafi kanitina
    ek olarak)."""
    fake_root = tmp_path / "facts"
    fake_root.mkdir()
    (fake_root / "load_to_db.py").write_text(
        "def f(cur):\n    cur.execute(\"UPDATE spec_value SET status='present' WHERE 1=0\")\n",
        encoding="utf-8",
    )
    (fake_root / "build_facts_db.py").write_text(
        "def f(cur):\n    cur.executemany(\"INSERT INTO spec_value (product_code) VALUES (?)\", [])\n",
        encoding="utf-8",
    )
    (fake_root / "sneaky_third_writer.py").write_text(
        "def f(cur):\n    cur.execute(\"INSERT INTO spec_value (product_code) VALUES ('X')\")\n",
        encoding="utf-8",
    )

    writers = db_write_audit.find_sql_write_files(root=fake_root)
    writer_basenames = {path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path in writers}

    assert "sneaky_third_writer.py" in writer_basenames
    assert not writer_basenames.issubset(db_write_audit.ALLOWED_WRITER_FILENAMES)


def test_comment_mentioning_insert_is_not_a_false_positive(tmp_path):
    """Grep-tabanli bir kontrol burada YANLIS ALARM verirdi -- AST-tabanli
    kontrol vermemeli: `execute` cagrisinin ilk argumani bir SELECT, INSERT
    kelimesi yalniz bir YORUMDA geciyor."""
    fake_root = tmp_path / "facts"
    fake_root.mkdir()
    (fake_root / "innocent_reader.py").write_text(
        textwrap.dedent(
            '''
            def f(cur):
                # NOT: bu fonksiyon specs.db'ye INSERT YAPMAZ, yalniz okur.
                return cur.execute("SELECT * FROM spec_value").fetchall()
            '''
        ),
        encoding="utf-8",
    )

    writers = db_write_audit.find_sql_write_files(root=fake_root)
    assert writers == {}


def test_tests_directory_and_pycache_are_excluded(tmp_path):
    fake_root = tmp_path / "facts"
    (fake_root / "tests").mkdir(parents=True)
    (fake_root / "__pycache__").mkdir()
    (fake_root / "tests" / "test_something.py").write_text(
        "def f(cur):\n    cur.execute(\"INSERT INTO spec_value (product_code) VALUES ('X')\")\n",
        encoding="utf-8",
    )
    (fake_root / "__pycache__" / "junk.py").write_text(
        "def f(cur):\n    cur.execute(\"INSERT INTO spec_value (product_code) VALUES ('X')\")\n",
        encoding="utf-8",
    )

    writers = db_write_audit.find_sql_write_files(root=fake_root)
    assert writers == {}


def test_dynamic_sql_from_a_variable_is_not_seen_this_is_a_documented_limitation(tmp_path):
    """Belgelenmis sinirlama: SQL bir DEGISKENDEN geliyorsa gorulmez. Testin
    amaci bu davranisi SESSIZCE degil, ACIKCA kayda gecirmek -- mevcut kod
    tabaninda boyle deger-yazan tek cagri olmadigi ayrica bu dosyanin
    docstring'inde belirtiliyor."""
    fake_root = tmp_path / "facts"
    fake_root.mkdir()
    (fake_root / "sneaky_dynamic_writer.py").write_text(
        textwrap.dedent(
            """
            def f(cur):
                q = "INSERT INTO spec_value (product_code) VALUES ('X')"
                cur.execute(q)
            """
        ),
        encoding="utf-8",
    )

    writers = db_write_audit.find_sql_write_files(root=fake_root)
    assert writers == {}  # bilinen sinirlama -- bkz. modul docstring'i
