"""Tokenization (`chunker.tokenization`) tests — offline.

Per the offline rule, real tiktoken is NEVER TOUCHED (it downloads the
encoding file from the network on first use): tiktoken-bearing paths are
tested via a stub module placed into `sys.modules`. So tests pass even in
an environment without tiktoken installed; what they actually verify is
HOW the backend calls tiktoken.
"""

import sys
from types import SimpleNamespace

import pytest

from medrag.pipeline.chunker.tokenization import (
    ENV_VAR,
    FALLBACK_SPEC,
    FakeTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    get_tokenizer,
)


class StubEncoding:
    """Stub encoding that records encode calls and counts 1 char = 1 token."""

    def __init__(self, name):
        self.name = name
        self.encode_calls = []

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return list(range(len(text)))


def stub_tiktoken_kur(monkeypatch, gecerli=("cl100k_base", "o200k_base")):
    """Install a stub module where `import tiktoken` will find it; returns
    the encodings dict."""
    kurulanlar = {}

    def get_encoding(name):
        if name not in gecerli:
            raise ValueError(f"Unknown encoding {name}")
        return kurulanlar.setdefault(name, StubEncoding(name))

    monkeypatch.setitem(sys.modules, "tiktoken",
                        SimpleNamespace(get_encoding=get_encoding))
    return kurulanlar


# -- FakeTokenizer ---------------------------------------------------------------


def test_fake_bos_metin_sifir():
    assert FakeTokenizer().count("") == 0


def test_fake_salt_whitespace_sifir():
    assert FakeTokenizer().count("  \t \n ") == 0


def test_fake_kelime_sayar():
    assert FakeTokenizer().count("bir iki üç dört") == 4


def test_fake_tum_whitespace_ayrac():
    # Tabs, newlines, and consecutive spaces all count as a single separator
    # (str.split semantics).
    assert FakeTokenizer().count("a\tb\nc   d") == 4


def test_fake_deterministik():
    metin = "aynı metin aynı sayı"
    fake = FakeTokenizer()
    assert fake.count(metin) == fake.count(metin) == FakeTokenizer().count(metin)


def test_fake_protokole_uyar():
    assert isinstance(FakeTokenizer(), Tokenizer)
    assert FakeTokenizer().name == "fake"


# -- TiktokenTokenizer (via stub) -----------------------------------------------


def test_tiktoken_dogru_encoding_ile_kurulur(monkeypatch):
    kurulanlar = stub_tiktoken_kur(monkeypatch)
    tok = TiktokenTokenizer("cl100k_base")
    assert list(kurulanlar) == ["cl100k_base"]
    assert tok.name == "tiktoken:cl100k_base"
    assert isinstance(tok, Tokenizer)


def test_tiktoken_count_encode_uzunlugu(monkeypatch):
    stub_tiktoken_kur(monkeypatch)
    assert TiktokenTokenizer("cl100k_base").count("abcde") == 5  # stub: 1 char = 1 token


def test_tiktoken_ozel_token_duz_metin_sayilir(monkeypatch):
    # Special-token strings that appear in document text must not break
    # chunking: encode must be called with disallowed_special=() (default
    # raises ValueError).
    kurulanlar = stub_tiktoken_kur(monkeypatch)
    TiktokenTokenizer("cl100k_base").count("önce  sonra")
    (_, kwargs), = kurulanlar["cl100k_base"].encode_calls
    assert kwargs == {"disallowed_special": ()}


def test_tiktoken_gecersiz_encoding_kurulumda_patlar(monkeypatch):
    stub_tiktoken_kur(monkeypatch)
    with pytest.raises(ValueError, match="hokus_pokus"):
        TiktokenTokenizer("hokus_pokus")


def test_tiktoken_kurulu_degilse_anlasilir_hata(monkeypatch):
    # None in sys.modules: `import tiktoken` → ImportError (simulates not installed).
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError, match="tiktoken is not installed"):
        TiktokenTokenizer("cl100k_base")


# -- get_tokenizer (env-based selection) ----------------------------------------


def test_factory_fake_spec():
    assert isinstance(get_tokenizer("fake"), FakeTokenizer)


def test_factory_env_fake(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "fake")
    assert isinstance(get_tokenizer(), FakeTokenizer)


def test_factory_acik_spec_envi_ezer(monkeypatch):
    stub_tiktoken_kur(monkeypatch)
    monkeypatch.setenv(ENV_VAR, "fake")
    tok = get_tokenizer("o200k_base")
    assert tok.name == "tiktoken:o200k_base"


def test_factory_env_encoding_adi(monkeypatch):
    stub_tiktoken_kur(monkeypatch)
    monkeypatch.setenv(ENV_VAR, "o200k_base")
    assert get_tokenizer().name == "tiktoken:o200k_base"


def test_factory_env_yoksa_fallback(monkeypatch):
    stub_tiktoken_kur(monkeypatch)
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert get_tokenizer().name == f"tiktoken:{FALLBACK_SPEC}"


def test_factory_bos_env_fallback(monkeypatch):
    # If the .env has "TOKENIZER=" (empty), the fallback must still kick in.
    stub_tiktoken_kur(monkeypatch)
    monkeypatch.setenv(ENV_VAR, "")
    assert get_tokenizer().name == f"tiktoken:{FALLBACK_SPEC}"