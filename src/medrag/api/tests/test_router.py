import pytest

from medrag.api.config import Routing
from medrag.api.flows.base import FlowContext
from medrag.api.retrieval.core import IntentLabel
from medrag.api.router import Router


class _FakeFlow:
    def __init__(self, name: str):
        self.name = name

    async def run(self, query: str) -> FlowContext:
        return FlowContext(results=[])


def _routing(**intents: str) -> Routing:
    return Routing(default_flow="default_topn", intents=intents)


def test_routes_listed_intent_to_its_flow():
    flows = {"default_topn": _FakeFlow("default"), "sql_topn": _FakeFlow("sql")}
    router = Router(flows, _routing(medical_fact="sql_topn"))
    assert router.flow_for(IntentLabel.MEDICAL_FACT) is flows["sql_topn"]


def test_unlisted_intent_falls_back_to_default():
    flows = {"default_topn": _FakeFlow("default"), "sql_topn": _FakeFlow("sql")}
    router = Router(flows, _routing(medical_fact="sql_topn"))
    assert router.flow_for(IntentLabel.CLINICAL_DECISION) is flows["default_topn"]


def test_accepts_plain_string_label():
    flows = {"default_topn": _FakeFlow("default")}
    router = Router(flows, _routing())
    assert router.flow_for("medical_fact") is flows["default_topn"]


def test_unknown_default_flow_raises_at_construction():
    flows = {"sql_topn": _FakeFlow("sql")}
    with pytest.raises(KeyError):
        Router(flows, _routing())


def test_unknown_intent_flow_raises_at_construction():
    flows = {"default_topn": _FakeFlow("default")}
    with pytest.raises(KeyError):
        Router(flows, _routing(medical_fact="sql_topn"))


# --- flow_by_name (pinned-flow mechanism) ------------------------------------


def test_flow_by_name_returns_registered_flow():
    flows = {"default_topn": _FakeFlow("default"), "comparison": _FakeFlow("cmp")}
    router = Router(flows, _routing())
    assert router.flow_by_name("comparison") is flows["comparison"]


def test_flow_by_name_unknown_raises_keyerror():
    flows = {"default_topn": _FakeFlow("default")}
    router = Router(flows, _routing())
    with pytest.raises(KeyError):
        router.flow_by_name("no_such_flow")


# --- `on_trace` -- matched intent vs. silent default_flow fallback -----------


def test_flow_for_without_on_trace_behaves_unchanged():
    flows = {"default_topn": _FakeFlow("default"), "sql_topn": _FakeFlow("sql")}
    router = Router(flows, _routing(medical_fact="sql_topn"))
    assert router.flow_for(IntentLabel.MEDICAL_FACT) is flows["sql_topn"]
    assert router.flow_for(IntentLabel.CLINICAL_DECISION) is flows["default_topn"]


def test_flow_for_emits_routed_when_intent_matched():
    flows = {"default_topn": _FakeFlow("default"), "sql_topn": _FakeFlow("sql")}
    router = Router(flows, _routing(medical_fact="sql_topn"))
    events: list[tuple[str, object]] = []

    result = router.flow_for(IntentLabel.MEDICAL_FACT, lambda step, data: events.append((step, data)))

    assert result is flows["sql_topn"]
    assert events == [("routed", "sql_topn")]


def test_flow_for_emits_routed_default_fallback_when_unlisted():
    flows = {"default_topn": _FakeFlow("default"), "sql_topn": _FakeFlow("sql")}
    router = Router(flows, _routing(medical_fact="sql_topn"))
    events: list[tuple[str, object]] = []

    result = router.flow_for(IntentLabel.CLINICAL_DECISION, lambda step, data: events.append((step, data)))

    assert result is flows["default_topn"]
    assert events == [("routed_default_fallback", "default_topn")]
