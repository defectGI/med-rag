"""N-04 (resilience to kill-and-resume): facts phase.

`run_full.run_products()` ALREADY carries product-level checkpointing
(`results/<CODE>.json`, rewritten after EVERY chunk, with the
`partial=true` flag -- see `run_one_product`/`_is_partial` docstrings,
2026-08-06 fix). This file's job is to PROVE that mechanism ACTUALLY
"resumes from where it stopped" with a destructive test: monkeypatch
`_call_chunk` (a fake version that does NOT make a real LLM/Ollama
call) to throw a `KeyboardInterrupt` mid-way through the third product
and "kill" the run, then resume with `--skip-existing` equivalent
(`skip_existing=True`).

KNOWN BLOCK (O-04, same family as test_chunk_full_run_load.py):
`run_full.py` reads `prompts/chunk_single_system.md` at module level;
this file was NOT part of O-02's moved set, so this module currently
throws `FileNotFoundError` at import. When the prompt files are moved
into the package, the skip lifts itself (the tests below start running
that day) -- N-04's "code, not test setup" cost is already written, only
the import block is waiting to be lifted.
"""

from __future__ import annotations

import json

import pytest

try:
    from medrag.pipeline.facts import run_full

    _RUN_FULL_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 -- see module-level note
    run_full = None
    _RUN_FULL_ERROR = exc

_needs_run_full = pytest.mark.skipif(
    run_full is None,
    reason=f"medrag.pipeline.facts.run_full import blocked (prompts/chunk_single_system.md not moved into package): {_RUN_FULL_ERROR}",
)


class _Manifest:
    def __init__(self, code):
        self.product_code = code
        self.family, self.subfamily, self.display_name = "F", None, code
        self.doc_ids = [f"{code}-doc"]
        self.docs_missing_text = []
        self.docs_excluded_fanout = []
        self.anchors_not_found = []

        class _Chunk:
            def __init__(self):
                self.chunk_sha256 = f"sha256:{code}" + "0" * 58
                self.text = f"{code} body text"
                self.anchors_applied = []

        self.chunks = [_Chunk()]


@_needs_run_full
def test_kill_mid_run_then_resume_with_skip_existing(tmp_path, monkeypatch):
    codes = ["DE1", "DE2", "DE3", "DE4"]
    results_dir = tmp_path / "results"

    call_log: list[str] = []

    def _fake_call_chunk(provider, model, base_url, api_key, manifest, chunk, label, system):
        call_log.append(manifest.product_code)
        if manifest.product_code == "DE3":
            raise KeyboardInterrupt("simulated kill mid-product")
        facts = [{"key": "supply_voltage", "kind": "single", "num_value": 24,
                 "source_chunk_id": chunk.chunk_sha256}]
        return facts, [{"batch_label": label, "n_chunks": 1, "n_facts": 1,
                        "chunk_sha256": chunk.chunk_sha256, "usage": {}}]

    monkeypatch.setattr(run_full, "_call_chunk", _fake_call_chunk)

    def _manifest_for(con, code, **kw):
        return _Manifest(code)

    monkeypatch.setattr(run_full, "build_product_manifest", _manifest_for)

    con = object()  # not queried in this test (manifest is faked)

    # `run_products` CATCHS KeyboardInterrupt ITSELF (run_full.py:514) --
    # does NOT re-raise, returns with a clean `summary["interrupted"]=True`.
    # When this test was first written, the import block (K-71) meant
    # it never ran; the expectation `pytest.raises(KeyboardInterrupt)`
    # turned out to be wrong once it actually did run.
    first_summary = run_full.run_products(
        codes, con=con, provider="ollama", model="m", base_url="http://x",
        api_key=None, all_chunks_index={}, atoms_bridge_index={},
        exclude_doc_types=frozenset(), owner_counts={},
        chunk_ownership_index={}, chunk_anchor_index={},
        results_dir=results_dir, system="sys",
    )
    assert first_summary["interrupted"] is True

    # Products before the kill (DE1, DE2) are COMPLETE and valid --
    # checkpoints are not half-written, the `partial` flag is cleared.
    for code in ("DE1", "DE2"):
        data = json.loads((results_dir / f"{code}.json").read_text(encoding="utf-8"))
        assert run_full._is_partial(results_dir / f"{code}.json") is False
        assert data["facts"]

    # If DE3's file exists (chunk-level checkpoint may have written
    # `partial=True` at the first chunk), `_is_partial` must mark it as
    # needing RE-PROCESSING. DE4 was never reached.
    de3_path = results_dir / "DE3.json"
    if de3_path.is_file():
        assert run_full._is_partial(de3_path) is True
    assert not (results_dir / "DE4.json").is_file()

    # --- next run: --skip-existing equivalent, resume ---
    call_log.clear()

    def _call_chunk_ok(provider, model, base_url, api_key, manifest, chunk, label, system):
        call_log.append(manifest.product_code)
        facts = [{"key": "supply_voltage", "kind": "single", "num_value": 24,
                 "source_chunk_id": chunk.chunk_sha256}]
        return facts, [{"batch_label": label, "n_chunks": 1, "n_facts": 1,
                        "chunk_sha256": chunk.chunk_sha256, "usage": {}}]

    monkeypatch.setattr(run_full, "_call_chunk", _call_chunk_ok)
    summary = run_full.run_products(
        codes, con=con, provider="ollama", model="m", base_url="http://x",
        api_key=None, all_chunks_index={}, atoms_bridge_index={},
        exclude_doc_types=frozenset(), owner_counts={},
        chunk_ownership_index={}, chunk_anchor_index={},
        results_dir=results_dir, system="sys", skip_existing=True,
    )

    assert set(call_log) == {"DE3", "DE4"}, (
        "DE1/DE2 are already COMPLETE (partial=false), so skip_existing=True "
        "must NOT RE-PROCESS them -- only those half-done or never-started "
        "at kill time (DE3, DE4) should continue"
    )
    assert summary["interrupted"] is False
    for code in codes:
        assert run_full._is_partial(results_dir / f"{code}.json") is False