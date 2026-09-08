"""SqlGenerator / cleaning tests. No engine, no network."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from medrag.api.retrieval.modules.db_query import (
    SqlEngine,
    SqlGenerator,
    Text2SqlGenerator,
    clean_sql,
    sqlite_ilike_to_like,
)


@dataclass
class _FakeResult:
    sql: str


class _FakeEngine:
    """Stand-in for text2sql.Text2SQL: returns a canned SQL string."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def run(self, request: str) -> _FakeResult:
        return _FakeResult(sql=self._sql)


@dataclass
class _Usage:
    total_tokens: int = 10


@dataclass
class _StageMeta:
    latency_seconds: float = 0.1
    usage: _Usage = field(default_factory=_Usage)
    attempts: int = 1


@dataclass
class _RunMeta:
    linking: _StageMeta = field(default_factory=_StageMeta)
    generation: _StageMeta = field(default_factory=_StageMeta)


@dataclass
class _LinkedColumn:
    table: str
    column: str


@dataclass
class _LinkedSchema:
    tables: list = field(default_factory=lambda: ["product"])
    columns: list = field(default_factory=lambda: [_LinkedColumn("product", "model")])
    rationale: str = "user asked about a specific model"


@dataclass
class _RichFakeResult:
    """Mirrors text2sql.Text2SQLResult's shape (linked_schema/metadata/
    explanation) -- the real object `_emit_stage_details` reads via getattr."""

    sql: str
    linked_schema: _LinkedSchema = field(default_factory=_LinkedSchema)
    metadata: _RunMeta = field(default_factory=_RunMeta)
    explanation: str = "matched product.model"


class _RichFakeEngine:
    def __init__(self, sql: str) -> None:
        self._sql = sql

    def run(self, request: str) -> _RichFakeResult:
        return _RichFakeResult(sql=self._sql)


class _FailingEngine:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def run(self, request: str) -> _FakeResult:
        raise self._exc


class _StageRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, stage: str, detail: dict) -> None:
        self.events.append((stage, detail))


def test_clean_sql_strips_fences_and_prefix_and_semicolon():
    raw = "```sql\nSELECT 1;\n```"
    assert clean_sql(raw) == "SELECT 1"


def test_clean_sql_plain_passthrough():
    assert clean_sql("  SELECT a FROM t  ") == "SELECT a FROM t"


def test_sqlite_ilike_rewrite():
    assert sqlite_ilike_to_like("SELECT * FROM t WHERE n ILIKE '%x%'") == (
        "SELECT * FROM t WHERE n LIKE '%x%'"
    )


def test_generator_satisfies_protocol():
    gen = Text2SqlGenerator(_FakeEngine("SELECT 1"))
    assert isinstance(gen, SqlGenerator)


def test_fake_engine_satisfies_sql_engine_protocol():
    # The injected engine is typed to the SqlEngine protocol, not the concrete
    # text2sql class — a fake with run(request) -> .sql is a valid substitute.
    assert isinstance(_FakeEngine("SELECT 1"), SqlEngine)


def test_generator_cleans_and_rewrites():
    engine = _FakeEngine("```sql\nSELECT * FROM t WHERE n ILIKE 'a';\n```")
    gen = Text2SqlGenerator(engine, rewriters=[sqlite_ilike_to_like])
    assert gen.generate("q") == "SELECT * FROM t WHERE n LIKE 'a'"


# --- on_stage: observability hook ----------------------------------------------


def test_generate_without_on_stage_unaffected():
    # Minimal SqlResult (only .sql) -- on_stage not passed, nothing to skip.
    gen = Text2SqlGenerator(_FakeEngine("SELECT 1"))
    assert gen.generate("q") == "SELECT 1"


def test_on_stage_skipped_gracefully_for_minimal_result():
    # Fake result only has `.sql` (no linked_schema/metadata) -- on_stage
    # given but nothing fires; must not raise (getattr-based, not required).
    gen = Text2SqlGenerator(_FakeEngine("SELECT 1"))
    recorder = _StageRecorder()
    assert gen.generate("q", on_stage=recorder) == "SELECT 1"
    assert recorder.events == []


def test_on_stage_emits_linking_and_generation_detail():
    gen = Text2SqlGenerator(_RichFakeEngine("SELECT model FROM product"))
    recorder = _StageRecorder()
    sql = gen.generate("de9001 nedir", on_stage=recorder)

    assert sql == "SELECT model FROM product"
    steps = [name for name, _ in recorder.events]
    assert steps == ["linking_done", "generation_done"]

    linking_detail = recorder.events[0][1]
    assert linking_detail["tables"] == ["product"]
    assert linking_detail["columns"] == ["product.model"]
    assert linking_detail["rationale"] == "user asked about a specific model"
    assert linking_detail["tokens"] == 10

    generation_detail = recorder.events[1][1]
    assert generation_detail["explanation"] == "matched product.model"


def test_on_stage_receives_error_detail_and_error_still_raises():
    class _FakeErrorWithRawText(Exception):
        def __init__(self, message: str, raw_text: str) -> None:
            super().__init__(message)
            self.raw_text = raw_text

    exc = _FakeErrorWithRawText("could not parse", raw_text="oops not json")
    gen = Text2SqlGenerator(_FailingEngine(exc))
    recorder = _StageRecorder()

    with pytest.raises(_FakeErrorWithRawText):
        gen.generate("q", on_stage=recorder)

    assert recorder.events == [
        ("sql_error", {
            "error_type": "_FakeErrorWithRawText",
            "message": "could not parse",
            "raw_text": "oops not json",
        })
    ]


def test_on_stage_error_detail_defaults_raw_text_when_absent():
    # A plain exception (no raw_text attribute, e.g. a transport error) must
    # not crash the observability path -- getattr defaults to "".
    gen = Text2SqlGenerator(_FailingEngine(RuntimeError("boom")))
    recorder = _StageRecorder()

    with pytest.raises(RuntimeError):
        gen.generate("q", on_stage=recorder)

    assert recorder.events == [
        ("sql_error", {"error_type": "RuntimeError", "message": "boom", "raw_text": ""})
    ]
