"""tokenization — token-counting abstraction.

Chunk size is measured in tokens, not characters; `core/` counts only
through the `Tokenizer` interface. Selection rationale lives in the
module docstrings: interface in `base`, default backend in
`tiktoken_backend`, test fake in `fake`, env-based selection in `factory`.
"""

from medrag.pipeline.chunker.tokenization.base import Tokenizer
from medrag.pipeline.chunker.tokenization.factory import (
    ENV_VAR,
    FAKE_SPEC,
    FALLBACK_SPEC,
    get_tokenizer,
)
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer
from medrag.pipeline.chunker.tokenization.tiktoken_backend import TiktokenTokenizer

__all__ = [
    "ENV_VAR",
    "FAKE_SPEC",
    "FALLBACK_SPEC",
    "FakeTokenizer",
    "TiktokenTokenizer",
    "Tokenizer",
    "get_tokenizer",
]