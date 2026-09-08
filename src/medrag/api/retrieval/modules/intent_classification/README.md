# intent_classification

Status: **usable (v1).**

Classifies an incoming query into exactly one of the 9 v1 intents and returns
**only a label** (`core.IntentResult`). It does **not** route, and it does not
know what any intent's retrieval strategy is — that mapping is the consuming
project's job (ARCHITECTURE.md #7). Per-intent strategy drafts live separately
in the `strategies/` directory.

Extra: `pip install -e ".[intent_classification]"`

## Design

- **v1 backend:** LLM prompt based. The prompt lists the 9 labels (from
  `core.IntentLabel`) with short definitions + non-golden few-shots, and asks
  for a single label token. Output is parsed defensively (casing/quotes/extra
  text tolerated); an unparseable answer falls back to `out_of_scope`.
- **Provider-agnostic:** the classifier takes a `langchain-core`
  `BaseChatModel`. Tests inject a fake model (no network/GPU); production wires
  a real OpenAI-compatible endpoint (e.g. Ollama `/v1`) via
  `build_default_chat_model()` reading `.env`.
- **Swappable:** future backends (embedding-similarity, small model) implement
  the same `IntentClassifier` protocol; consuming code is unchanged.
- **Confidence:** derived from token logprobs when the endpoint returns them,
  else `None` — never fatal (works against endpoints without logprobs support).

## Usage

```python
from retrieval.modules.intent_classification import (
    LLMIntentClassifier, build_default_chat_model,
)

# On the inference machine (reads .env: LLM_BASE_URL/API_KEY/MODEL):
clf = LLMIntentClassifier(build_default_chat_model())
result = await clf.classify("de1000 fiyati ne kadar")
# result.label -> IntentLabel.PRODUCT_FACT ; result.confidence -> float | None
```

In tests, pass a `BaseChatModel` fake instead of `build_default_chat_model()`.

## Eval

`scripts/eval_intent.py` runs the classifier over a golden set and prints
accuracy / macro-F1 / micro-F1 / per-class. It calls the real endpoint, so run
it **on the inference machine**. The same pass is available as an
`@pytest.mark.integration` test (skipped by default; GPU rule,
ARCHITECTURE.md #10).
