"""Tokenizer selection: `TOKENIZER` env variable → concrete tokenizer.

Selection grammar (matches `.env.example` documentation exactly):

- ``fake``            → `FakeTokenizer` (test/offline; not for production)
- any other value     → tiktoken encoding name, used by `TiktokenTokenizer`
                        (e.g. ``cl100k_base``, ``o200k_base``)

When the env variable is absent, `FALLBACK_SPEC` is used — the code-side
counterpart of the README's "works without .env, tokenizer selection is
optional" promise. This selects a measurement tool, NOT a splitting
threshold — outside the no-hardcoded-threshold rule.

The engine receives its tokenizer as a parameter (dependency injection);
`get_tokenizer()` is only where entry points (CLI) build a concrete
tokenizer from env. Loading the `.env` file is the entry point's job —
library code only reads `os.environ`.
"""

from __future__ import annotations

import os

from medrag.pipeline.chunker.tokenization.base import Tokenizer
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer
from medrag.pipeline.chunker.tokenization.tiktoken_backend import TiktokenTokenizer

ENV_VAR = "TOKENIZER"
FALLBACK_SPEC = "cl100k_base"
FAKE_SPEC = "fake"


def get_tokenizer(spec: str | None = None) -> Tokenizer:
    """Build a tokenizer from `spec` (or `TOKENIZER` env if not given).

    Precedence: explicit `spec` argument > env variable > `FALLBACK_SPEC`.
    An invalid tiktoken encoding name fails HERE, at construction time —
    not at the first `count` call.
    """
    if spec is None:
        spec = os.environ.get(ENV_VAR) or FALLBACK_SPEC
    if spec == FAKE_SPEC:
        return FakeTokenizer()
    return TiktokenTokenizer(spec)