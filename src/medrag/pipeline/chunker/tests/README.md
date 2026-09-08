# tests

Test package. Principles (same as the first_parse parser repo):

- **Fully offline** — no network, no real LLM/model calls, no `.env`
  required. The LLM client and tokenizer are stubbed in tests.
- Each test builds its own small input: adapter tests use hand-written
  minimal IR JSONs; core tests work directly with the inner model.
- File names mirror the modules they cover: `test_core_*.py`,
  `test_adapter_first_parse.py`, `test_tokenization.py`, ...

Run with: `pytest`