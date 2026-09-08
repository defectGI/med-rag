# tokenization

Tokenizer abstraction. Chunk size is measured by **token** count, not
characters; `core/` counts only through this interface — so the
tokenizer (tiktoken, HuggingFace, a model-specific counter...) is
pluggable.

## Structure

- `base.py` — the `Tokenizer` protocol: `count(text) -> int` + diagnostic
  `name`. Deliberately narrow (no `encode`/`decode` exposed); being a
  `Protocol` means a new backend does not have to inherit from this
  package.
- `tiktoken_backend.py` — default implementation (`TiktokenTokenizer`).
  The tiktoken import is deferred to construction; special-token strings
  (`<|...|>`) are counted as plain text. The encoding file is
  downloaded and cached on first use (`TIKTOKEN_CACHE_DIR`).
- `fake.py` — deterministic test counter (`FakeTokenizer`,
  1 word = 1 token). Offline test/e2e only; not for production.
- `factory.py` — `get_tokenizer(spec=None)`: explicit `spec` > `TOKENIZER`
  env > `cl100k_base` fallback. `"fake"` selects the fake; any other
  value is a tiktoken encoding name. Loading the `.env` file is the entry
  point's job; the library only reads `os.environ`.

## Usage

```python
from chunker.tokenization import get_tokenizer

tok = get_tokenizer()              # from env (or cl100k_base)
tok = get_tokenizer("o200k_base")  # explicit selection
tok.count("hello world")         # -> int
```

The engine (`core/`) doesn't select its tokenizer; it receives it as a
parameter. The factory is only where entry points (CLI) build a concrete
tokenizer from env.

Tests are offline: real tiktoken is never touched; a `sys.modules` stub
verifies HOW the backend calls tiktoken
(`tests/test_tokenization.py`).