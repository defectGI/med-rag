import os
from pathlib import Path

import pytest

from medrag.api import factory
from medrag.api.flows import (
    AggregationFlow,
    ComparisonFlow,
    DefaultTopNFlow,
    DocDownloadFlow,
    NoRetrievalFlow,
    RecommendationFlow,
    SqliteDocumentLookup,
    SqlTopNFlow,
)
from medrag.api.retrieval.modules.db_query import DbQueryError
from medrag.api.sql_citation import add_citation_columns


class _FakeRetriever:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def retrieve(self, query, k=10):
        return []


def test_build_top_n_retriever_from_env_passes_chatbot_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://gpu-host:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setenv("QDRANT_COLLECTION", "medrag_chunks")

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return _FakeRetriever(**kwargs)

    monkeypatch.setattr(factory, "build_top_n_retriever", fake_build)

    factory.build_top_n_retriever_from_env()

    assert captured == {
        "qdrant_url": "http://gpu-host:6333",
        "qdrant_api_key": "secret",
        "collection_name": "medrag_chunks",
    }


def test_build_db_query_retriever_from_env_missing_falls_back_to_bundled(monkeypatch, tmp_path):
    # Without env, falls back to the committed facts/db/specs.db.
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    monkeypatch.delenv("SCHEMA_PATH", raising=False)
    monkeypatch.delenv("PROMPT_DIR", raising=False)
    bundled = tmp_path / "specs.db"
    bundled.write_bytes(b"")
    monkeypatch.setattr(factory, "_BUNDLED_DB_PATH", bundled)
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return _FakeRetriever(**kwargs)

    monkeypatch.setattr(factory, "build_db_query_retriever", fake_build)
    factory.build_db_query_retriever_from_env()

    assert captured["db_path"] == str(bundled)


def test_build_db_query_retriever_from_env_missing_and_no_bundled_raises(monkeypatch, tmp_path):
    # Neither env nor the committed DB -> it still raises.
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    monkeypatch.setattr(factory, "_BUNDLED_DB_PATH", tmp_path / "yok.db")
    with pytest.raises(DbQueryError):
        factory.build_db_query_retriever_from_env()


def test_build_db_query_retriever_from_env_passes_path_and_bundled_config(monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "specs.db")
    monkeypatch.delenv("SCHEMA_PATH", raising=False)
    monkeypatch.delenv("PROMPT_DIR", raising=False)
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return _FakeRetriever(**kwargs)

    monkeypatch.setattr(factory, "build_db_query_retriever", fake_build)
    factory.build_db_query_retriever_from_env()

    assert captured["db_path"] == "specs.db"
    assert captured["config_path"] == str(factory._BUNDLED_CONFIG_PATH)
    # If SCHEMA_PATH/PROMPT_DIR are not set by the user, it falls back to the
    # committed files. SCHEMA_PATH now points at the TRIMMED projection -- NOT
    # the full `schema.yaml`.
    assert os.environ["SCHEMA_PATH"] == str(factory._BUNDLED_MODEL_SCHEMA_PATH)
    assert os.environ["SCHEMA_PATH"] != str(factory._BUNDLED_SCHEMA_PATH)
    assert os.environ["PROMPT_DIR"] == str(factory._BUNDLED_PROMPT_DIR)
    # Citation columns are added by CODE, not by the PROMPT:
    # `sql_citation.add_citation_columns` must be passed as a rewriter to the retriever.
    assert add_citation_columns in captured["rewriters"]


def test_build_db_query_retriever_from_env_respects_user_schema_path_override(monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "specs.db")
    monkeypatch.setenv("SCHEMA_PATH", "/ozel/schema.yaml")
    monkeypatch.setenv("PROMPT_DIR", "/ozel/prompts")
    monkeypatch.setattr(factory, "build_db_query_retriever", lambda **kwargs: _FakeRetriever(**kwargs))

    factory.build_db_query_retriever_from_env()

    # setdefault does NOT overwrite the user's own value.
    assert os.environ["SCHEMA_PATH"] == "/ozel/schema.yaml"
    assert os.environ["PROMPT_DIR"] == "/ozel/prompts"


# --- resolve_db_path_from_env / build_document_lookup_from_env -----------------


def test_resolve_db_path_from_env_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "/ozel/specs.db")
    assert factory.resolve_db_path_from_env() == "/ozel/specs.db"


def test_resolve_db_path_from_env_falls_back_to_bundled(monkeypatch, tmp_path):
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    bundled = tmp_path / "specs.db"
    bundled.write_bytes(b"")
    monkeypatch.setattr(factory, "_BUNDLED_DB_PATH", bundled)
    assert factory.resolve_db_path_from_env() == str(bundled)


def test_resolve_db_path_from_env_raises_when_neither_available(monkeypatch, tmp_path):
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    monkeypatch.setattr(factory, "_BUNDLED_DB_PATH", tmp_path / "yok.db")
    with pytest.raises(DbQueryError):
        factory.resolve_db_path_from_env()


def test_build_document_lookup_from_env_uses_resolved_path(monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "/ozel/specs.db")
    lookup = factory.build_document_lookup_from_env()
    assert isinstance(lookup, SqliteDocumentLookup)
    assert lookup._db_path == "/ozel/specs.db"


def test_build_db_query_retriever_from_env_and_document_lookup_share_same_path(monkeypatch):
    # There must be NO two separate resolution paths -- both call the same
    # function (resolve_db_path_from_env), so they cannot drift apart.
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "/ozel/specs.db")
    monkeypatch.setattr(factory, "build_db_query_retriever", lambda **kw: _FakeRetriever(**kw))
    captured = {}
    real_build = factory.build_db_query_retriever

    def spy(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(factory, "build_db_query_retriever", spy)
    factory.build_db_query_retriever_from_env()
    lookup = factory.build_document_lookup_from_env()

    assert captured["db_path"] == "/ozel/specs.db"
    assert lookup._db_path == "/ozel/specs.db"


def test_document_lookup_from_env_or_none_uses_resolved_path(monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "/ozel/specs.db")
    lookup = factory._document_lookup_from_env_or_none()
    assert isinstance(lookup, SqliteDocumentLookup)
    assert lookup._db_path == "/ozel/specs.db"


def test_document_lookup_from_env_or_none_degrades_gracefully_when_db_missing(monkeypatch, tmp_path):
    # The injectable instance for `Orchestrator._document_hint` -- when the DB
    # is not configured at all / the committed file is also missing, it must
    # not take down `create_app()` ENTIRELY (unlike
    # `build_document_lookup_from_env` this returns `None`; the hint feature
    # silently turns off).
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    monkeypatch.setattr(factory, "_BUNDLED_DB_PATH", tmp_path / "yok.db")
    assert factory._document_lookup_from_env_or_none() is None


def test_build_flows_from_env_wires_all_three(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "build_top_n_retriever_from_env", lambda: _FakeRetriever())
    monkeypatch.setattr(factory, "build_db_query_retriever_from_env", lambda **kw: _FakeRetriever())
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", str(tmp_path / "specs.db"))

    flows = factory.build_flows_from_env()

    assert set(flows) == {
        "default_topn", "sql_topn", "aggregation", "comparison", "recommendation",
        "doc_download", "no_retrieval",
    }
    assert isinstance(flows["default_topn"], DefaultTopNFlow)
    assert isinstance(flows["sql_topn"], SqlTopNFlow)
    assert isinstance(flows["aggregation"], AggregationFlow)
    assert isinstance(flows["comparison"], ComparisonFlow)
    assert isinstance(flows["recommendation"], RecommendationFlow)
    assert isinstance(flows["doc_download"], DocDownloadFlow)
    assert isinstance(flows["no_retrieval"], NoRetrievalFlow)
    # doc_download uses its own DocumentLookup -- it does NOT SHARE the
    # retriever with sql_topn (unlike comparison/aggregation/recommendation).
    assert isinstance(flows["doc_download"]._lookup, SqliteDocumentLookup)
    # comparison/aggregation/recommendation share the SAME SqlTopNFlow instance
    # as sql_topn (see the factory.py docstring) -- this verifies no separate
    # retriever is built.
    assert flows["comparison"]._pass is flows["sql_topn"]
    assert flows["aggregation"]._pass is flows["sql_topn"]
    assert flows["recommendation"]._pass is flows["sql_topn"]
    # comparison is now built with a REAL (deterministic) splitter -- unlike
    # aggregation (splitter=None) -- so N-product comparison + clarification
    # flows are active in production from the start.
    from medrag.api.flows.comparison import DeterministicComparisonSplitter

    assert isinstance(flows["comparison"]._splitter, DeterministicComparisonSplitter)
    assert flows["aggregation"]._splitter is None


def test_build_router_from_env_uses_flows(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "build_top_n_retriever_from_env", lambda: _FakeRetriever())
    monkeypatch.setattr(factory, "build_db_query_retriever_from_env", lambda **kw: _FakeRetriever())
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", str(tmp_path / "specs.db"))

    router = factory.build_router_from_env()

    from medrag.api.retrieval.core import IntentLabel

    # med-rag tıbbi routing: out_of_scope dışında her intent tek vektör yoluna
    # (default_topn) düşer.
    assert isinstance(router.flow_for(IntentLabel.MEDICAL_FACT), DefaultTopNFlow)
    assert isinstance(router.flow_for(IntentLabel.CLINICAL_DECISION), DefaultTopNFlow)
    assert isinstance(router.flow_for(IntentLabel.COMPARISON), DefaultTopNFlow)
    assert isinstance(router.flow_for(IntentLabel.INTERACTION), DefaultTopNFlow)
    assert isinstance(router.flow_for(IntentLabel.LIBRARY), DefaultTopNFlow)
    assert isinstance(router.flow_for(IntentLabel.OUT_OF_SCOPE), NoRetrievalFlow)


# --- GPU gate: the SQL retriever is serialized --------------------------------


def test_db_query_retriever_wrapped_in_serialized(monkeypatch):
    from medrag.api.gpu_gate import SerializedRetriever

    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", "specs.db")
    monkeypatch.setattr(factory, "build_db_query_retriever", lambda **kw: _FakeRetriever(**kw))

    retriever = factory.build_db_query_retriever_from_env(sql_concurrency=2)
    assert isinstance(retriever, SerializedRetriever)
    assert retriever._sem._value == 2  # the semaphore counter came from config


def test_serialized_retriever_passes_through_kwargs():
    # Optional hooks like on_sql/on_stage must pass through the wrapper VERBATIM.
    import asyncio

    from medrag.api.gpu_gate import SerializedRetriever

    seen = {}

    class _HookRetriever:
        async def retrieve(self, query, k=10, **kwargs):
            seen.update(kwargs)
            return ["ok"]

    wrapped = SerializedRetriever(_HookRetriever(), concurrency=1)
    out = asyncio.run(wrapped.retrieve("q", k=5, on_sql=lambda s: None, on_stage=lambda a, b: None))
    assert out == ["ok"]
    assert set(seen) == {"on_sql", "on_stage"}


# --- fusion: intent-label injection into the reconciler ------------------------


def test_reconciler_gets_intent_labels_when_fuse_on(monkeypatch):
    from medrag.api.config import load_config

    monkeypatch.setenv("LLM_BASE_URL", "http://llm/v1")
    monkeypatch.setenv("LLM_MODEL", "small")
    cfg = load_config()
    assert cfg.session.fuse_intent is True  # enabled by default
    rec = factory.build_reconciler_from_env(cfg)
    assert rec is not None
    assert rec._intent_labels is not None and len(rec._intent_labels) >= 6


def test_reconciler_no_labels_when_fuse_off(monkeypatch, tmp_path):
    from medrag.api.config import load_config

    override = tmp_path / "off.toml"
    override.write_text("[session]\nfuse_intent = false\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm/v1")
    monkeypatch.setenv("LLM_MODEL", "small")
    cfg = load_config(override=override)
    rec = factory.build_reconciler_from_env(cfg)
    assert rec is not None
    assert rec._intent_labels is None


# --- bundled-path gate ---------------------------------------------------------
# Every test above monkeypatches the `_BUNDLED_*` constants to tmp_path --
# so they stay green even if the real path is wrong. These are the only gates
# that catch the migration's actual failure mode (the file not being there).


def test_bundled_paths_point_at_real_committed_files():
    """The committed fallback files REALLY exist and live under facts/db/."""
    repo_root = Path(factory.__file__).resolve().parents[3]
    facts_db = repo_root / "facts" / "db"

    assert factory._BUNDLED_DB_PATH == facts_db / "specs.db"
    assert factory._BUNDLED_SCHEMA_PATH == facts_db / "schema.yaml"
    assert factory._BUNDLED_DB_PATH.is_file(), "facts/db/specs.db missing"
    assert factory._BUNDLED_SCHEMA_PATH.is_file(), "facts/db/schema.yaml missing"

    # config.toml + prompts/ STAYED in this component (runtime wiring)
    assert factory._BUNDLED_CONFIG_PATH == repo_root / "chatbot" / "text2sql" / "config.toml"
    assert factory._BUNDLED_CONFIG_PATH.is_file()
    assert factory._BUNDLED_PROMPT_DIR.is_dir()


def test_build_intent_chat_model_default_uses_retrieval_factory(monkeypatch):
    """When `INTENT_PROVIDER` is empty, retrieval's `build_default_chat_model`
    is used -- behavior must remain unchanged (the single pre-feature path)."""
    import medrag.api.retrieval.modules.intent_classification as intent_mod

    sentinel = object()
    monkeypatch.setattr(intent_mod, "build_default_chat_model", lambda: sentinel)
    assert factory._build_intent_chat_model({}) is sentinel


def test_build_intent_chat_model_ollama_builds_native_model(monkeypatch):
    """`INTENT_PROVIDER=ollama` reads retrieval's own LLM_* values
    (base_url/model) and passes them to `NativeOllamaChatModel`; when num_ctx
    is empty it falls back to the 16384 default
    (answering_model._DEFAULT_OLLAMA_NUM_CTX)."""
    env = {
        "INTENT_PROVIDER": "ollama",
        "LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_MODEL": "gemma4:31b",
    }
    model = factory._build_intent_chat_model(env)
    assert model.ollama_base_url == "http://localhost:11434/v1"
    assert model.ollama_model == "gemma4:31b"
    assert model.num_ctx == 16384


def test_build_intent_chat_model_ollama_num_ctx_override():
    env = {
        "INTENT_PROVIDER": "ollama",
        "LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_MODEL": "gemma4:31b",
        "INTENT_NUM_CTX": "131072",
    }
    assert factory._build_intent_chat_model(env).num_ctx == 131072


def test_build_intent_chat_model_ollama_without_llm_endpoint_raises():
    from medrag.api.answering_model import ProviderError

    with pytest.raises(ProviderError):
        factory._build_intent_chat_model({"INTENT_PROVIDER": "ollama"})


def test_chatbot_never_imports_facts_package():
    """Boundary: the API never imports `facts` CODE, it only reads paths."""
    pkg_dir = Path(factory.__file__).resolve().parent
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import facts", "from facts ")):
                offenders.append(f"{py.name}:{lineno}")
    assert not offenders, f"the API imports the facts package: {offenders}"
