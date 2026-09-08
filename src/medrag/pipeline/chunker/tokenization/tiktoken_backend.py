"""Default tokenizer backend: tiktoken.

The tiktoken import is deferred to class construction (not module import):
importing the package does NOT require tiktoken — tests using `fake` and
tiktoken-less environments can import `chunker.tokenization` freely.
If tiktoken isn't installed, construction fails with an actionable error
(no silent fallback to `fake`; same "fail at the boundary" principle).

Note: tiktoken downloads and caches the encoding file on first use
(location can be selected via `TIKTOKEN_CACHE_DIR`). The test suite
therefore doesn't touch real tiktoken — see the offline rule in
`tests/test_tokenization.py`.
"""

from __future__ import annotations


class TiktokenTokenizer:
    """Counter that uses a tiktoken encoding.

    `encoding_name` is a tiktoken encoding name (e.g. ``cl100k_base``,
    ``o200k_base``); an invalid name is rejected by tiktoken at
    construction time.
    """

    def __init__(self, encoding_name: str) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "tiktoken is not installed; required for the default tokenizer. "
                "Install: pip install -r requirements.txt "
                "(alternative for tests: TOKENIZER=fake)") from exc
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def count(self, text: str) -> int:
        # disallowed_special=(): treat any special-token strings ("<|"..."|"<")
        # that may appear in document text as plain text — the default raises
        # ValueError, which would drop chunking for an entire document.
        return len(self._encoding.encode(text, disallowed_special=()))