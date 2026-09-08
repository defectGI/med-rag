import asyncio

from medrag.api.config import Routing
from medrag.api.flows.base import FlowContext
from medrag.api.flows.doc_download import DocumentRow
from medrag.api.memory import ConversationMemory, Message
from medrag.api.orchestrator import Orchestrator, load_strategy
from medrag.api.reconciler import ReconcileResult
from medrag.api.retrieval.core import IntentLabel, IntentResult, RetrievalResult
from medrag.api.router import Router
from medrag.api.session_state import PinnedFlow, SessionState, SessionStateStore


class _FakeClassifier:
    def __init__(self, label: IntentLabel, *, raw_text: str | None = None):
        self._label = label
        self._raw_text = raw_text
        self.seen: list[str] = []

    async def classify(self, query: str) -> IntentResult:
        self.seen.append(query)
        return IntentResult(label=self._label, raw_text=self._raw_text)


class _FakeReconciler:
    """Returns a fixed ReconcileResult; records which (state, history, query)
    it was called with."""

    def __init__(self, result: ReconcileResult):
        self._result = result
        self.calls: list[tuple] = []

    async def reconcile(self, state, history, query):
        self.calls.append((state, list(history), query))
        return self._result


class _FakeFlow:
    def __init__(self, results: list[RetrievalResult] | None = None):
        self.calls: list[str] = []
        self.on_trace_seen = []
        self._results = results if results is not None else []

    async def run(self, query: str, on_trace=None) -> FlowContext:
        self.calls.append(query)
        self.on_trace_seen.append(on_trace)
        if on_trace:
            on_trace("flow_ran", query)
        return FlowContext(results=list(self._results))


class _FakePinnableFlow:
    """Fake flow implementing the `PinnableFlow` protocol -- `check_pin`
    returns a fixed value (or raises one if given) and records its calls."""

    def __init__(self, results: list[RetrievalResult] | None = None,
                 *, holds: bool = True, raises: Exception | None = None):
        self.calls: list[str] = []
        self.check_pin_calls: list[tuple] = []
        self._results = results if results is not None else []
        self._holds = holds
        self._raises = raises

    async def run(self, query: str, on_trace=None) -> FlowContext:
        self.calls.append(query)
        if on_trace:
            on_trace("flow_ran", query)
        return FlowContext(results=list(self._results))

    async def check_pin(self, query: str, pinned) -> bool:
        self.check_pin_calls.append((query, pinned))
        if self._raises is not None:
            raise self._raises
        return self._holds


class _FakeDocumentLookup:
    """Mimics `DocumentLookup` -- filters `model` from a fixed `DocumentRow`
    list and records its calls (so cases where `_document_hint` never calls
    it can be tested too)."""

    def __init__(self, rows: list[DocumentRow] | None = None):
        self._rows = rows if rows is not None else []
        self.calls: list[list[str]] = []

    async def find(self, models: list[str], doc_type: str | None = None) -> list[DocumentRow]:
        self.calls.append(list(models))
        upper = {m.upper() for m in models}
        return [r for r in self._rows if r.model.upper() in upper]


class _FakeAnsweringModel:
    def __init__(self):
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []

    async def answer(self, query, context, strategy_prompt, history, **kwargs):
        self.calls.append((query, context, strategy_prompt, list(history)))
        self.kwargs.append(kwargs)
        return f"cevap: {query}"


def test_load_strategy_reads_existing_file():
    text = load_strategy("product_fact")
    assert "product_fact" in text


def test_load_strategy_missing_file_returns_empty():
    assert load_strategy("no_such_intent") == ""


def test_load_strategy_doc_question_has_real_guidance():
    """doc_question.md (med-rag, C1): zorunlu kanıt/atıf kuralını taşıyan
    gerçek rehber olmalı."""
    text = load_strategy("doc_question")
    assert "doc_question" in text
    assert "to be filled in when this intent's strategy is addressed" not in text
    assert "MUST carry an inline citation" in text
    assert "Conflicting sources" in text


def test_handle_wires_classify_route_flow_answer_together():
    flow = _FakeFlow()
    router = Router(
        {"default_topn": flow, "no_retrieval": flow},
        Routing(default_flow="default_topn", intents={"out_of_scope": "no_retrieval"}),
    )
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(classifier, router, model, ConversationMemory())

    answer = asyncio.run(orchestrator.handle("session-1", "hava durumu nasil"))

    assert answer == "cevap: hava durumu nasil"
    assert flow.calls == ["hava durumu nasil"]
    query, _context, strategy_prompt, history = model.calls[0]
    assert query == "hava durumu nasil"
    assert "out_of_scope" in strategy_prompt
    assert history == []  # first turn, memory empty


def test_handle_defaults_to_web_channel():
    flow = _FakeFlow()
    router = Router(
        {"default_topn": flow, "no_retrieval": flow},
        Routing(default_flow="default_topn", intents={"out_of_scope": "no_retrieval"}),
    )
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(classifier, router, model, ConversationMemory())

    asyncio.run(orchestrator.handle("session-1", "hava durumu nasil"))

    assert model.kwargs[0]["channel"] == "web"


def test_handle_passes_configured_whatsapp_channel_to_answering_model():
    # `Orchestrator(channel=...)` -- the instance built by wa_bot.py must
    # pass this channel to `answer()` on every call so the WhatsApp-specific
    # FORMATTING instruction reaches the model (see
    # answering_model.py::_formatting_for_channel).
    flow = _FakeFlow()
    router = Router(
        {"default_topn": flow, "no_retrieval": flow},
        Routing(default_flow="default_topn", intents={"out_of_scope": "no_retrieval"}),
    )
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(
        classifier, router, model, ConversationMemory(), channel="whatsapp"
    )

    asyncio.run(orchestrator.handle("session-1", "hava durumu nasil"))

    assert model.kwargs[0]["channel"] == "whatsapp"


def test_handle_passes_prior_turns_as_history_and_appends_new_ones():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    memory = ConversationMemory()
    orchestrator = Orchestrator(classifier, router, model, memory)

    asyncio.run(orchestrator.handle("s1", "ilk soru"))
    asyncio.run(orchestrator.handle("s1", "ikinci soru"))

    # the history passed to the model on the second call contains the first
    # turn's user+assistant messages.
    _, _, _, history_on_second_call = model.calls[1]
    assert history_on_second_call == [
        Message(role="user", content="ilk soru"),
        Message(role="assistant", content="cevap: ilk soru"),
    ]

    # at session end four messages accumulated in memory (two turns).
    assert memory.get("s1") == [
        Message(role="user", content="ilk soru"),
        Message(role="assistant", content="cevap: ilk soru"),
        Message(role="user", content="ikinci soru"),
        Message(role="assistant", content="cevap: ikinci soru"),
    ]


def test_different_sessions_do_not_share_history():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    memory = ConversationMemory()
    orchestrator = Orchestrator(classifier, router, model, memory)

    asyncio.run(orchestrator.handle("session-a", "soru a"))
    asyncio.run(orchestrator.handle("session-b", "soru b"))

    _, _, _, history_b = model.calls[1]
    assert history_b == []


# --- L0-L4: the reconciler path ------------------------------------------------


def _orchestrator_with_reconciler(reconciler, classifier=None):
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = classifier or _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    orch = Orchestrator(
        classifier, router, model, ConversationMemory(),
        reconciler=reconciler, session_state=store, hedge_threshold=0.5,
    )
    return orch, flow, classifier, model, store


def test_reconciler_resolved_query_flows_to_classify_and_flow():
    new_state = SessionState(task="karşılaştırma", focus="de1100 gücü")
    result = ReconcileResult(resolved_query="de1100'in gücü", new_state=new_state,
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, flow, classifier, model, _ = _orchestrator_with_reconciler(rec)

    asyncio.run(orch.handle("s1", "peki ya diğeri"))

    # classify + flow see the RESOLVED query, not the raw one.
    assert classifier.seen == ["de1100'in gücü"]
    assert flow.calls == ["de1100'in gücü"]
    # the goal reached the answering model as a frame.
    assert "karşılaştırma" in model.kwargs[0]["session_goal"]
    assert model.kwargs[0]["hedge"] is False


def test_reconciler_low_confidence_triggers_hedge():
    result = ReconcileResult(resolved_query="soru", new_state=SessionState(task="t"),
                             transition="continue", confidence=0.2)
    rec = _FakeReconciler(result)
    orch, _, _, model, _ = _orchestrator_with_reconciler(rec)
    asyncio.run(orch.handle("s1", "soru"))
    assert model.kwargs[0]["hedge"] is True  # 0.2 < 0.5 threshold


def test_reconciler_new_state_is_persisted():
    new_state = SessionState(task="kalıcı görev", turn=1)
    result = ReconcileResult(resolved_query="q", new_state=new_state,
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, _, store = _orchestrator_with_reconciler(rec)
    asyncio.run(orch.handle("s1", "q"))
    assert store.get("s1").task == "kalıcı görev"


def test_reconciler_raw_query_stored_in_memory_not_resolved():
    result = ReconcileResult(resolved_query="ÇÖZÜMLENMİŞ", new_state=SessionState(),
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    memory = ConversationMemory()
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    orch = Orchestrator(_FakeClassifier(IntentLabel.PRODUCT_FACT), router,
                        _FakeAnsweringModel(), memory, reconciler=rec)
    asyncio.run(orch.handle("s1", "ham mesaj"))
    # memory must hold what the user ACTUALLY typed, not the resolved form.
    assert memory.get("s1")[0] == Message(role="user", content="ham mesaj")


def test_reconciler_emits_trace_events():
    result = ReconcileResult(resolved_query="çözümlenmiş", new_state=SessionState(task="t"),
                             transition="evolve", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, _, _ = _orchestrator_with_reconciler(rec)
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "ham", lambda s, d: events.append((s, d))))
    steps = [s for s, _ in events]
    assert "reconcile" in steps
    assert "resolved_query" in steps  # raw != resolved
    assert "goal" in steps
    assert ("reconcile", "evolve") in events


def test_reconciler_raw_text_emits_reconcile_raw_trace():
    # If the reconciler returned something but it couldn't be parsed, that
    # raw text must not vanish silently -- it goes to the trace as
    # "reconcile_raw".
    result = ReconcileResult(resolved_query="soru", new_state=SessionState(),
                             transition="continue", confidence=None,
                             raw_text="garbled reconciler output")
    rec = _FakeReconciler(result)
    orch, _, _, _, _ = _orchestrator_with_reconciler(rec)
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    assert ("reconcile_raw", "garbled reconciler output") in events


def test_reconciler_raw_text_absent_by_default_no_trace():
    result = ReconcileResult(resolved_query="soru", new_state=SessionState(),
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, _, _ = _orchestrator_with_reconciler(rec)
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    steps = [s for s, _ in events]
    assert "reconcile_raw" not in steps


def test_intent_raw_text_emits_intent_raw_trace():
    # The intent classifier returned something that matched no label (it
    # silently fell back to out_of_scope) -- the raw text must be visible.
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE, raw_text="bugun hava nasil")
    orch = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory())
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    assert ("intent_raw", "bugun hava nasil") in events


def test_intent_raw_text_absent_by_default_no_trace():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)  # raw_text=None
    orch = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory())
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    steps = [s for s, _ in events]
    assert "intent_raw" not in steps


# --- fusion: intent comes from the reconciler ----------------------------------


def test_fused_intent_skips_classifier():
    # the reconciler also returned an intent -> NO separate classify call.
    new_state = SessionState(task="t", focus="f")
    result = ReconcileResult(resolved_query="de1000 gücü", new_state=new_state,
                             transition="continue", confidence=0.9,
                             intent="product_fact")
    rec = _FakeReconciler(result)
    orch, flow, classifier, _, _ = _orchestrator_with_reconciler(rec)

    asyncio.run(orch.handle("s1", "de1000 gücü"))

    assert classifier.seen == []  # the classifier was never called
    assert flow.calls == ["de1000 gücü"]


def test_fused_intent_emits_fusion_marker_in_trace():
    result = ReconcileResult(resolved_query="q", new_state=SessionState(task="t"),
                             transition="continue", confidence=0.8,
                             intent="product_fact")
    rec = _FakeReconciler(result)
    orch, _, _, _, _ = _orchestrator_with_reconciler(rec)
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "q", lambda s, d: events.append((s, d))))
    intent_events = [d for s, d in events if s == "intent"]
    assert intent_events and "füzyon" in intent_events[0]


def test_fused_invalid_label_falls_back_to_classifier():
    # the reconciler gave no valid label (None) -> fall back to the old path, the classifier.
    result = ReconcileResult(resolved_query="q", new_state=SessionState(task="t"),
                             transition="continue", confidence=0.9, intent=None)
    rec = _FakeReconciler(result)
    orch, _, classifier, _, _ = _orchestrator_with_reconciler(rec)
    asyncio.run(orch.handle("s1", "q"))
    assert classifier.seen == ["q"]  # fallback: classify was called


def test_transition_note_reaches_answering_model():
    # the switch/digress transition must reach the answering model inside
    # session_goal -- previously it stayed only in trace + internal state.
    new_state = SessionState(task="yeni görev", focus="fiyat")
    result = ReconcileResult(resolved_query="q", new_state=new_state,
                             transition="switch", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, model, _ = _orchestrator_with_reconciler(rec)
    asyncio.run(orch.handle("s1", "q"))
    goal = model.kwargs[0]["session_goal"]
    assert "yeni görev" in goal
    assert "NOTE:" in goal  # transition note appended


def test_without_reconciler_uses_raw_query():
    # Without a Reconciler (optional) the old behavior: raw query goes straight through.
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    orch = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory())
    asyncio.run(orch.handle("s1", "ham sorgu"))
    assert classifier.seen == ["ham sorgu"]
    assert flow.calls == ["ham sorgu"]


# --- last_results ----------------------------------------------------------------


def test_last_results_persisted_without_reconciler():
    # Even without a Reconciler the flow result must be stored for the next
    # turn -- the "not just this flow; a general part of the architecture"
    # requirement.
    hits = [RetrievalResult(id="row_1", score=1.0, metadata={"model": "PN1197"})]
    flow = _FakeFlow(results=hits)
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    store = SessionStateStore()
    orch = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory(),
                        session_state=store)
    asyncio.run(orch.handle("s1", "de2200 nedir"))
    assert store.get("s1").last_results == hits


def test_last_results_persisted_with_reconciler_alongside_other_state():
    # WITH a Reconciler, last_results must be written onto the same
    # SessionState without disturbing the reconciler's other fields (task etc.).
    hits = [RetrievalResult(id="row_1", score=1.0, metadata={"model": "PN1197"})]
    new_state = SessionState(task="kalıcı görev", turn=1)
    result = ReconcileResult(resolved_query="q", new_state=new_state,
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, flow, _, _, store = _orchestrator_with_reconciler(rec)
    flow._results = hits
    asyncio.run(orch.handle("s1", "q"))
    persisted = store.get("s1")
    assert persisted.last_results == hits
    assert persisted.task == "kalıcı görev"


def test_last_results_overwritten_not_accumulated_across_turns():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    store = SessionStateStore()
    orch = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory(),
                        session_state=store)

    flow._results = [RetrievalResult(id="row_1", score=1.0)]
    asyncio.run(orch.handle("s1", "birinci soru"))
    assert store.get("s1").last_results == [RetrievalResult(id="row_1", score=1.0)]

    flow._results = [RetrievalResult(id="row_2", score=1.0)]
    asyncio.run(orch.handle("s1", "ikinci soru"))
    # the second turn's results REPLACED the first, no accumulation.
    assert store.get("s1").last_results == [RetrievalResult(id="row_2", score=1.0)]


# --- followups -------------------------------------------------------------------


def test_followups_reach_answering_model_from_previous_turn(monkeypatch):
    # No Reconciler -- previously state was never read in this case (only a
    # fresh get() after L4), so followups could never reach the answering model.
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    store = SessionStateStore()
    store.set("s1", SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
    ]))
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)
    asyncio.run(orch.handle("s1", "ikinci soru"))
    assert "PN1309: operating_temperature" in model.kwargs[0]["followups"]


def test_followups_empty_on_first_turn():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    store = SessionStateStore()
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)
    asyncio.run(orch.handle("s1", "ilk soru"))
    assert model.kwargs[0]["followups"] == ""


def test_followups_survive_alongside_reconciler():
    hits_prev_turn = [RetrievalResult(id="r1", score=1.0, metadata={
        "product_code": "PN1309", "key": "input_voltage", "status": "present",
    })]
    new_state = SessionState(task="kalıcı görev", turn=1)
    result = ReconcileResult(resolved_query="q", new_state=new_state,
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, model, store = _orchestrator_with_reconciler(rec)
    store.set("s1", SessionState(last_results=hits_prev_turn))
    asyncio.run(orch.handle("s1", "q"))
    assert "PN1309: input_voltage" in model.kwargs[0]["followups"]


# --- _document_hint: proactive file suggestion while a product is discussed -----


def test_document_hint_suggests_real_file_for_single_product_this_turn():
    """`product_fact` (the sql_topn core) returned a single product THIS turn
    -- the previous turn is empty and `document_lookup` knows a real active
    file for that product -- the suggestion must be derived from THIS turn's
    results and reach the model."""
    flow = _FakeFlow([RetrievalResult(id="r1", score=1.0, metadata={
        "product_code": "PN1309", "key": "weight", "status": "present",
    })])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    lookup = _FakeDocumentLookup([
        DocumentRow(doc_id="d1", model="PN1309", doc_type="DATASHEET", file_name="PN5106_Datasheet.pdf"),
    ])
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), document_lookup=lookup)

    asyncio.run(orch.handle("s1", "PN1309 agirligi kac kg"))

    assert lookup.calls == [["PN1309"]]
    assert "PN5106_Datasheet.pdf" in model.kwargs[0]["followups"]
    assert "DATASHEET" in model.kwargs[0]["followups"]


def test_document_hint_skipped_when_multiple_products_in_results():
    """Multiple product codes -- it cannot know WHICH one to suggest for, so
    it doesn't guess (silently skipped on multi-product comparison/aggregation
    results; `document_lookup` is never called)."""
    flow = _FakeFlow([
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "weight"}),
        RetrievalResult(id="r2", score=1.0, metadata={"product_code": "PN1316", "key": "weight"}),
    ])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    lookup = _FakeDocumentLookup([
        DocumentRow(doc_id="d1", model="PN1309", doc_type="DATASHEET", file_name="a.pdf"),
    ])
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), document_lookup=lookup)

    asyncio.run(orch.handle("s1", "PN1309 ve PN1316 agirligi"))

    assert lookup.calls == []
    assert model.kwargs[0]["followups"] == ""


def test_document_hint_empty_when_lookup_has_no_file():
    """The product is single and clear, but the `document` table has no real
    active file for it -- no invention, returns empty."""
    flow = _FakeFlow([RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "weight"})])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    lookup = _FakeDocumentLookup([])
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), document_lookup=lookup)

    asyncio.run(orch.handle("s1", "PN1309 agirligi"))

    assert model.kwargs[0]["followups"] == ""


def test_document_hint_disabled_when_no_lookup_injected():
    """If `document_lookup` is not injected (default `None`, legacy
    wiring/tests) zero behavior change -- returns empty."""
    flow = _FakeFlow([RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "weight"})])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory())

    asyncio.run(orch.handle("s1", "PN1309 agirligi"))

    assert model.kwargs[0]["followups"] == ""


def test_document_hint_yields_priority_to_previous_turn_followup():
    """If the previous turn already produced a spec-key follow-up
    (`available_followups`), `_document_hint` is NEVER CALLED -- the "single,
    natural suggestion" principle (see session_state.py::followups_prompt)
    is also upheld against THIS turn's file hint; two suggestions never pile
    up."""
    flow = _FakeFlow([RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "weight"})])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    lookup = _FakeDocumentLookup([
        DocumentRow(doc_id="d1", model="PN1309", doc_type="DATASHEET", file_name="a.pdf"),
    ])
    store = SessionStateStore()
    store.set("s1", SessionState(last_results=[RetrievalResult(id="r0", score=1.0, metadata={
        "product_code": "PN1309", "key": "operating_temperature", "status": "present",
    })]))
    model = _FakeAnsweringModel()
    orch = Orchestrator(
        classifier, router, model, ConversationMemory(),
        session_state=store, document_lookup=lookup,
    )

    asyncio.run(orch.handle("s1", "agirligi da ne"))

    assert lookup.calls == []
    assert "PN1309: operating_temperature" in model.kwargs[0]["followups"]
    assert "a.pdf" not in model.kwargs[0]["followups"]


def test_document_hint_ignores_doc_download_own_results():
    """doc_download's own rows carry the `model` key, NOT `product_code`
    (see flows/doc_download.py::_to_result) -- so `_document_hint` finds no
    candidates on doc_download turns (no clash with its own follow-up) and
    `document_lookup` is not called."""
    flow = _FakeFlow([RetrievalResult(id="doc:d1:PN1309", score=1.0, metadata={
        "doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
        "file_name": "a.pdf", "requested": True,
    })])
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.DOC_DOWNLOAD)
    lookup = _FakeDocumentLookup([
        DocumentRow(doc_id="d2", model="PN1309", doc_type="BROCHURE", file_name="b.pdf"),
    ])
    model = _FakeAnsweringModel()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), document_lookup=lookup)

    asyncio.run(orch.handle("s1", "de9001 datasheet"))

    assert lookup.calls == []
    assert model.kwargs[0]["followups"] == ""


# --- trace (live background panel) -----------------------------------------------


def test_handle_without_on_trace_still_works():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(classifier, router, model, ConversationMemory())

    answer = asyncio.run(orchestrator.handle("s1", "soru"))
    assert answer == "cevap: soru"
    assert flow.on_trace_seen == [None]


def test_handle_emits_intent_strategy_answering_and_forwards_to_flow():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(classifier, router, model, ConversationMemory())

    events: list[tuple] = []
    asyncio.run(orchestrator.handle("s1", "soru", lambda step, data: events.append((step, data))))

    # "timing" (per-stage duration) is sprinkled in between -- verify the
    # real flow order by filtering those out. With `intents={}` empty,
    # `out_of_scope` always falls to `default_flow`, and
    # "routed_default_fallback" is now a distinguishable event.
    steps = [name for name, _ in events if name != "timing"]
    assert steps == ["intent", "routed_default_fallback", "flow_ran", "strategy", "answering"]
    non_timing = [(s, d) for s, d in events if s != "timing"]
    assert non_timing[0] == ("intent", "out_of_scope")
    assert non_timing[1] == ("routed_default_fallback", "default_topn")
    assert non_timing[2] == ("flow_ran", "soru")  # on_trace really forwarded to the flow
    assert non_timing[3] == ("strategy", "out_of_scope")


def test_handle_emits_timing_for_each_stage():
    # The user wanted to see every stage's duration -- even without a
    # reconciler, timing events must arrive for intent/retrieval/answering/total.
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    orchestrator = Orchestrator(classifier, router, _FakeAnsweringModel(), ConversationMemory())

    events: list[tuple] = []
    asyncio.run(orchestrator.handle("s1", "soru", lambda step, data: events.append((step, data))))

    timings = {d["stage"]: d["seconds"] for s, d in events if s == "timing"}
    assert set(timings) == {"intent", "retrieval", "answering", "total"}
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())


def test_handle_emits_reconcile_timing_when_reconciler_present():
    result = ReconcileResult(resolved_query="soru", new_state=SessionState(),
                             transition="continue", confidence=0.9)
    rec = _FakeReconciler(result)
    orch, _, _, _, _ = _orchestrator_with_reconciler(rec)
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda step, data: events.append((step, data))))
    timings = {d["stage"] for s, d in events if s == "timing"}
    assert "reconcile" in timings


# --- pinned session --------------------------------------------------------------


def _orchestrator_with_pinned_flow(pinned_flow, *, reconciler=None, classifier=None):
    """`pinned_flow` is registered under `flow_by_name("comparison")`;
    `default_topn` is a different fake flow that the full-pipeline path falls
    to (deliberately different objects so pinned/unpinned paths can be
    verified not to cross)."""
    default_flow = _FakeFlow()
    router = Router(
        {"default_topn": default_flow, "comparison": pinned_flow},
        Routing(default_flow="default_topn", intents={}),
    )
    classifier = classifier or _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    orch = Orchestrator(classifier, router, model, ConversationMemory(),
                        reconciler=reconciler, session_state=store)
    return orch, default_flow, classifier, model, store


def _pinned_state(**kw) -> SessionState:
    return SessionState(
        task="karşılaştırma", focus="de1000 vs de1100",
        pinned=PinnedFlow(flow="comparison", intent="product_fact",
                          expected={"sides": ["de1000", "de1100"]}),
        **kw,
    )


def test_pinned_session_skips_reconcile_and_classify_when_pin_holds():
    pinned_flow = _FakePinnableFlow(holds=True)
    reconciler = _FakeReconciler(ReconcileResult(
        resolved_query="asla ulaşmamalı", new_state=SessionState(),
        transition="continue", confidence=0.9,
    ))
    orch, default_flow, classifier, _, store = _orchestrator_with_pinned_flow(
        pinned_flow, reconciler=reconciler,
    )
    store.set("s1", _pinned_state())

    answer = asyncio.run(orch.handle("s1", "gücü ne kadar"))

    assert classifier.seen == []  # full classify never called
    assert reconciler.calls == []  # reconcile never called
    assert default_flow.calls == []  # never fell to the full-pipeline flow
    assert pinned_flow.calls == ["gücü ne kadar"]  # the raw query went directly
    assert answer == "cevap: gücü ne kadar"
    # check_pin saw the pinned state (including expected).
    (_, seen_pin), = pinned_flow.check_pin_calls
    assert seen_pin.expected == {"sides": ["de1000", "de1100"]}


def test_pinned_turn_session_goal_carries_continuation_note():
    # On the pinned path the reconciler never runs -> there is no real
    # rec.transition, but this turn is structurally the continuation of the
    # earlier clarification -- a fixed note must reach the answering model
    # (previously it never did).
    pinned_flow = _FakePinnableFlow(holds=True)
    orch, _, _, model, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())

    asyncio.run(orch.handle("s1", "gücü ne kadar"))

    goal = model.kwargs[0]["session_goal"]
    assert "karşılaştırma" in goal  # the goal frame is still present
    assert "continues the earlier clarification" in goal


def test_pinned_session_persists_pin_and_advances_turn_when_held():
    pinned_flow = _FakePinnableFlow(holds=True, results=[
        RetrievalResult(id="r1", score=1.0, metadata={"model": "PN1015"}),
    ])
    orch, _, _, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state(turn=2))

    asyncio.run(orch.handle("s1", "soru"))

    persisted = store.get("s1")
    assert persisted.pinned is not None
    assert persisted.pinned.flow == "comparison"
    assert persisted.turn == 3  # advanced
    assert persisted.last_results == [
        RetrievalResult(id="r1", score=1.0, metadata={"model": "PN1015"}),
    ]


def test_pinned_session_uses_pin_intent_as_strategy_key():
    pinned_flow = _FakePinnableFlow(holds=True)
    orch, _, _, model, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())
    asyncio.run(orch.handle("s1", "soru"))
    _, _, strategy_prompt, _ = model.calls[0]
    assert "product_fact" in strategy_prompt


def test_pinned_session_falls_back_to_full_pipeline_when_pin_breaks():
    # Tests verifying that the fallback really works when pinning is
    # mis-triggered -- this is exactly that scenario.
    pinned_flow = _FakePinnableFlow(holds=False)
    orch, default_flow, classifier, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())

    answer = asyncio.run(orch.handle("s1", "tamamen alakasız bir soru"))

    assert classifier.seen == ["tamamen alakasız bir soru"]  # fell to the full pipeline
    assert default_flow.calls == ["tamamen alakasız bir soru"]
    assert pinned_flow.calls == []  # the pinned flow's run was never called
    assert answer == "cevap: tamamen alakasız bir soru"
    # Escape hatch: reset so the pin does not re-trigger.
    assert store.get("s1").pinned is None


def test_pinned_session_falls_back_when_flow_does_not_implement_pinnable():
    # No REAL flow implements PinnableFlow at this STAGE -- if a pin is set
    # anyway (manually, as in tests), the SAFE default must take over:
    # cannot be checked -> considered broken -> fall to the full pipeline.
    plain_flow = _FakeFlow()  # no check_pin
    orch, default_flow, classifier, _, store = _orchestrator_with_pinned_flow(plain_flow)
    store.set("s1", _pinned_state())

    asyncio.run(orch.handle("s1", "soru"))

    assert classifier.seen == ["soru"]
    assert default_flow.calls == ["soru"]
    assert plain_flow.calls == []
    assert store.get("s1").pinned is None


def test_pinned_session_falls_back_when_check_pin_raises():
    # Even if check_pin itself blows up (network error etc.), the user must
    # not get stuck -- the same safety net as the Reconciler's "no error ever
    # drops a turn" principle.
    pinned_flow = _FakePinnableFlow(raises=RuntimeError("boom"))
    orch, default_flow, classifier, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())

    answer = asyncio.run(orch.handle("s1", "soru"))

    assert classifier.seen == ["soru"]
    assert default_flow.calls == ["soru"]
    assert answer == "cevap: soru"
    assert store.get("s1").pinned is None


def test_pin_check_emits_held_trace_event():
    pinned_flow = _FakePinnableFlow(holds=True)
    orch, _, _, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    assert ("pin_check", "held") in events
    assert not any(s == "pin_fallback" for s, _ in events)


def test_pin_check_emits_broken_and_fallback_trace_events():
    pinned_flow = _FakePinnableFlow(holds=False)
    orch, _, _, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    assert ("pin_check", "broken") in events
    assert ("pin_fallback", "comparison") in events


def test_pinned_session_emits_pin_check_timing():
    pinned_flow = _FakePinnableFlow(holds=True)
    orch, _, _, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())
    events: list[tuple] = []
    asyncio.run(orch.handle("s1", "soru", lambda s, d: events.append((s, d))))
    timings = {d["stage"] for s, d in events if s == "timing"}
    assert "pin_check" in timings


def test_pinned_session_records_real_query_in_memory():
    pinned_flow = _FakePinnableFlow(holds=True)
    memory = ConversationMemory()
    router = Router({"default_topn": _FakeFlow(), "comparison": pinned_flow},
                    Routing(default_flow="default_topn", intents={}))
    store = SessionStateStore()
    store.set("s1", _pinned_state())
    orch = Orchestrator(_FakeClassifier(IntentLabel.PRODUCT_FACT), router,
                        _FakeAnsweringModel(), memory, session_state=store)
    asyncio.run(orch.handle("s1", "gerçek kullanıcı mesajı"))
    assert memory.get("s1")[0] == Message(role="user", content="gerçek kullanıcı mesajı")


def test_unpinned_session_behaves_exactly_as_before():
    # Regression: a session with pinned=None (current/old) must see zero
    # behavior change -- the mechanism is fully "opt-in".
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    orchestrator = Orchestrator(classifier, router, model, ConversationMemory())
    answer = asyncio.run(orchestrator.handle("s1", "soru"))
    assert answer == "cevap: soru"
    assert classifier.seen == ["soru"]


# --- ComparisonFlow end to end: SelfPinningFlow --------------------------------
# NOT fake flows -- the real `ComparisonFlow` + `SqlTopNFlow` +
# `DeterministicComparisonSplitter`; only the bottom SQL retriever is faked
# (no LLM/GPU calls on this machine, but this splitter is deterministic/regex
# anyway and can be tested with the real objects).


class _FakeSpecRowsRetriever:
    def __init__(self, rows_by_query: dict[str, list[RetrievalResult]]):
        self._rows_by_query = rows_by_query
        self.calls: list[str] = []

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append(query)
        return list(self._rows_by_query.get(query, []))


def _comparison_router(rows_by_query: dict[str, list[RetrievalResult]]) -> Router:
    from medrag.api.flows.comparison import (
        ComparisonFlow,
        DeterministicComparisonSplitter,
    )
    from medrag.api.flows.sql_topn import SqlTopNFlow

    sql = _FakeSpecRowsRetriever(rows_by_query)
    comparison_flow = ComparisonFlow(
        SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter()
    )
    return Router(
        {"default_topn": _FakeFlow(), "comparison": comparison_flow},
        Routing(default_flow="default_topn", intents={"comparison": "comparison"}),
    )


def test_comparison_ambiguous_query_pins_session_via_full_pipeline():
    router = _comparison_router({})
    classifier = _FakeClassifier(IntentLabel.COMPARISON)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    answer = asyncio.run(orch.handle("s1", "hangisi daha iyi"))

    assert answer == "cevap: hangisi daha iyi"
    pinned = store.get("s1").pinned
    assert pinned is not None
    assert pinned.flow == "comparison"
    assert pinned.intent == "comparison"


def test_comparison_pinned_session_resolves_and_clears_pin_via_cheap_path():
    router = _comparison_router({
        "PN1015": [RetrievalResult(id="r1", score=1.0,
                                   metadata={"key": "weight", "product_code": "PN1015",
                                             "status": "present", "raw_text": "2 kg"})],
        "PN1036": [RetrievalResult(id="r2", score=1.0,
                                   metadata={"key": "weight", "product_code": "PN1036",
                                             "status": "present", "raw_text": "3 kg"})],
    })
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)  # must not fire
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        pinned=PinnedFlow(flow="comparison", intent="comparison", expected={}),
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    asyncio.run(orch.handle("s1", "de1000 ve de1100'ün ağırlığı"))

    assert classifier.seen == []  # while pinned, full classify never called
    persisted = store.get("s1")
    assert persisted.pinned is None  # clarification resolved, pin cleared
    assert len(persisted.last_results) >= 2


def test_comparison_pinned_session_stays_pinned_while_still_ambiguous():
    router = _comparison_router({})
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        pinned=PinnedFlow(flow="comparison", intent="comparison",
                          expected={"raw_query": "hangisi daha iyi"}),
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    # It contains a single product code (an answer attempt -- the check_pin
    # fallback counts it as "relevant") but still fewer than 2 -- the
    # ambiguity is not resolved.
    asyncio.run(orch.handle("s1", "de1000, ne dersin"))

    persisted = store.get("s1")
    assert persisted.pinned is not None  # still ambiguous -- pin kept
    assert persisted.pinned.flow == "comparison"


def test_comparison_pinned_session_drops_pin_on_irrelevant_message():
    # A message with neither a product code nor a cancel word now DROPS the
    # pin (old behavior: "True until proven otherwise", the pin held forever).
    # The escape hatch kicks in; the user doesn't get stuck.
    router = _comparison_router({})
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        pinned=PinnedFlow(flow="comparison", intent="comparison",
                          expected={"raw_query": "hangisi daha iyi"}),
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    asyncio.run(orch.handle("s1", "bilmiyorum işte"))

    persisted = store.get("s1")
    assert persisted.pinned is None


# --- PinAwareRunFlow: while pinned, run_pinned() INSTEAD OF run() --------------


class _FakePinAwareFlow:
    """Implements both `PinnableFlow` and `PinAwareRunFlow` -- keeps a
    separate call list so the test can tell if plain `run()` was called."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.run_calls: list[str] = []
        self.run_pinned_calls: list[tuple] = []
        self._results = results if results is not None else []

    async def run(self, query: str, on_trace=None) -> FlowContext:
        self.run_calls.append(query)
        return FlowContext(results=list(self._results))

    async def run_pinned(self, query: str, pinned, on_trace=None) -> FlowContext:
        self.run_pinned_calls.append((query, pinned))
        return FlowContext(results=list(self._results))

    async def check_pin(self, query: str, pinned) -> bool:
        return True


def test_pinned_session_calls_run_pinned_instead_of_run_when_flow_implements_it():
    flow = _FakePinAwareFlow()
    orch, _, _, _, store = _orchestrator_with_pinned_flow(flow)
    store.set("s1", _pinned_state())

    asyncio.run(orch.handle("s1", "500 dolar civarı"))

    assert flow.run_calls == []  # plain run() never called
    query, seen_pin = flow.run_pinned_calls[0]
    assert query == "500 dolar civarı"
    assert seen_pin.flow == "comparison"


def test_pinned_session_still_calls_plain_run_when_flow_lacks_pin_aware_run():
    # `_FakePinnableFlow` implements only `run()` (NOT PinAwareRunFlow) --
    # zero behavior change for this flow (regression protection).
    pinned_flow = _FakePinnableFlow(holds=True)
    orch, _, _, _, store = _orchestrator_with_pinned_flow(pinned_flow)
    store.set("s1", _pinned_state())

    asyncio.run(orch.handle("s1", "soru"))

    assert pinned_flow.calls == ["soru"]


# --- simple flow-to-flow handoff field -------------------------------------------


class _FakeHandoffSourceFlow:
    """Mimics RecommendationFlow's first turn -- produces results AND offers a
    handoff (`suggested_next_flow`/`suggested_next_candidates`)."""

    def __init__(self, candidates: list[str]):
        self._candidates = candidates
        self.calls: list[str] = []

    async def run(self, query: str, on_trace=None) -> FlowContext:
        self.calls.append(query)
        return FlowContext(
            results=[RetrievalResult(id="r1", score=1.0, metadata={"model": self._candidates[0]})],
            suggested_next_flow="comparison", suggested_next_candidates=list(self._candidates),
        )


class _FakeHandoffTargetFlow:
    """Handoff target (e.g. `ComparisonFlow`) -- records which
    (pre-populated) query it was called with."""

    def __init__(self):
        self.calls: list[str] = []

    async def run(self, query: str, on_trace=None) -> FlowContext:
        self.calls.append(query)
        return FlowContext(results=[RetrievalResult(id="r2", score=1.0)])


def _handoff_router(source_flow, target_flow):
    return Router(
        {"default_topn": source_flow, "comparison": target_flow},
        Routing(default_flow="default_topn", intents={}),
    )


def test_recommendation_run_sets_suggested_handoff_on_session_state():
    source = _FakeHandoffSourceFlow(["PN1015", "PN1036"])
    target = _FakeHandoffTargetFlow()
    router = _handoff_router(source, target)
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)  # any label; should just fall to default_topn
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    asyncio.run(orch.handle("s1", "DAQ sistemi öner"))

    persisted = store.get("s1")
    assert persisted.suggested_next_flow == "comparison"
    assert persisted.suggested_next_candidates == ["PN1015", "PN1036"]
    assert target.calls == []  # not accepted yet, target flow NOT called


def test_user_accepts_handoff_routes_directly_to_target_flow_with_candidates():
    source = _FakeHandoffSourceFlow(["PN1015", "PN1036"])
    target = _FakeHandoffTargetFlow()
    router = _handoff_router(source, target)
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        suggested_next_flow="comparison", suggested_next_candidates=["PN1015", "PN1036"],
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    answer = asyncio.run(orch.handle("s1", "evet"))

    assert classifier.seen == []  # full classify skipped -- the cheap handoff path
    assert target.calls == ["PN1015 PN1036 evet"]  # candidates were PRE-populated
    assert answer == "cevap: evet"
    # The offer is one-shot -- reset after firing.
    persisted = store.get("s1")
    assert persisted.suggested_next_flow is None
    assert persisted.suggested_next_candidates == []


def test_user_declines_handoff_falls_back_to_normal_full_pipeline():
    # Deliberately a plain `_FakeFlow` (offering no handoff) instead of
    # `_FakeHandoffSourceFlow` as the full-pipeline flow -- the goal is to
    # verify "reject -> full pipeline ran + offer reset" in isolation, not
    # whether a NEW offer formed on this turn.
    source = _FakeFlow()
    target = _FakeHandoffTargetFlow()
    router = _handoff_router(source, target)
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        suggested_next_flow="comparison", suggested_next_candidates=["PN1015", "PN1036"],
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(), session_state=store)

    asyncio.run(orch.handle("s1", "hayır teşekkürler başka bir şey soracağım"))

    assert target.calls == []  # handoff never fired
    assert classifier.seen == ["hayır teşekkürler başka bir şey soracağım"]  # full pipeline ran
    persisted = store.get("s1")
    assert persisted.suggested_next_flow is None  # the offer was reset anyway (one-shot)


class _FakeContinuationChecker:
    def __init__(self, result: bool):
        self._result = result
        self.calls: list[tuple[str, str]] = []

    async def check(self, query: str, context: str) -> bool:
        self.calls.append((query, context))
        return self._result


class _RaisingContinuationChecker:
    async def check(self, query: str, context: str) -> bool:
        raise RuntimeError("endpoint down")


def test_handoff_accept_delegates_to_continuation_checker_when_injected():
    # "tabii ama bekle" is NOT in _AFFIRMATIVE_WORDS -- under the old
    # behavior it always counted as a rejection. Once a checker is injected,
    # the decision is delegated to it.
    source = _FakeHandoffSourceFlow(["PN1015", "PN1036"])
    target = _FakeHandoffTargetFlow()
    router = _handoff_router(source, target)
    classifier = _FakeClassifier(IntentLabel.PRODUCT_FACT)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        suggested_next_flow="comparison", suggested_next_candidates=["PN1015", "PN1036"],
    ))
    checker = _FakeContinuationChecker(True)
    orch = Orchestrator(classifier, router, model, ConversationMemory(),
                         session_state=store, continuation_checker=checker)

    asyncio.run(orch.handle("s1", "tabii ama bekle"))

    assert target.calls == ["PN1015 PN1036 tabii ama bekle"]
    assert len(checker.calls) == 1
    query, context = checker.calls[0]
    assert query == "tabii ama bekle"
    assert "PN1015" in context and "PN1036" in context


def test_handoff_accept_checker_error_defaults_to_declined():
    source = _FakeHandoffSourceFlow(["PN1015", "PN1036"])
    target = _FakeHandoffTargetFlow()
    router = _handoff_router(source, target)
    classifier = _FakeClassifier(IntentLabel.OUT_OF_SCOPE)
    model = _FakeAnsweringModel()
    store = SessionStateStore()
    store.set("s1", SessionState(
        suggested_next_flow="comparison", suggested_next_candidates=["PN1015", "PN1036"],
    ))
    orch = Orchestrator(classifier, router, model, ConversationMemory(),
                         session_state=store, continuation_checker=_RaisingContinuationChecker())

    asyncio.run(orch.handle("s1", "evet"))

    assert target.calls == []  # checker blew up -> safe default: treat as declined
    assert classifier.seen == ["evet"]  # fell to the normal full pipeline, user not stuck


def test_handoff_suggestion_persists_through_pinned_answer_path():
    # The `_answer_pinned` path must also carry `context.suggested_next_flow`
    # into the next turn -- not exclusive to the full pipeline.
    class _FakePinnableWithHandoff:
        async def run(self, query, on_trace=None):
            return FlowContext(
                results=[], suggested_next_flow="comparison",
                suggested_next_candidates=["PN1309"],
            )

        async def check_pin(self, query, pinned):
            return True

    flow = _FakePinnableWithHandoff()
    orch, _, _, _, store = _orchestrator_with_pinned_flow(flow)
    store.set("s1", _pinned_state())

    asyncio.run(orch.handle("s1", "soru"))

    persisted = store.get("s1")
    assert persisted.suggested_next_flow == "comparison"
    assert persisted.suggested_next_candidates == ["PN1309"]


# --- preamble --------------------------------------------------------------
# Scope: in ALL THREE Orchestrator branches, the bubble must start at the
# RIGHT moment with the right `known_flow`. The pool/language/watchdog logic
# itself is locked in `tests/test_preamble.py` -- only the WIRING is tested
# here.


def _preamble_events(trace: list[tuple[str, object]]) -> list[dict]:
    from medrag.api.preamble import PREAMBLE_STEP

    return [data for step, data in trace if step == PREAMBLE_STEP]


def _collector() -> tuple[list[tuple[str, object]], object]:
    seen: list[tuple[str, object]] = []

    def on_trace(step, data):
        seen.append((step, data))

    return seen, on_trace


def _settings(**kw):
    from medrag.api.preamble import PreambleSettings

    kw.setdefault("watchdog_seconds", 0.0)  # watchdog off in these tests
    kw.setdefault("fast_flows", frozenset({"no_retrieval", "doc_download", "default_topn"}))
    return PreambleSettings(**kw)


def test_preamble_verilmezse_hic_yayinlanmaz():
    """The default (`preamble=None`) behavior must be byte-for-byte the
    pre-feature one -- no new event may leak."""
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "PN1099 ağırlığı?", on_trace))

    assert _preamble_events(seen) == []


def test_preamble_tam_pipelinede_jenerik_veya_urun_kodu_havuzundan_basar():
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "PN1099 ağırlığı?", on_trace))

    events = _preamble_events(seen)
    assert len(events) == 1
    assert events[0]["pool"] == "product_code"
    assert events[0]["stage"] == "early"
    # The bubble must have gone out BEFORE retrieval (that is the whole point).
    steps = [step for step, _ in seen]
    assert steps.index("preamble") < steps.index("flow_ran")


def test_preamble_metni_konusma_hafizasina_GIRMEZ():
    """If it had entered, the reconciler on the next turn would mistake it
    for a real assistant reply (see orchestrator.py: only REAL
    questions/answers go into memory)."""
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    memory = ConversationMemory()
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=memory, preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "PN1099 ağırlığı?", on_trace))

    preamble_text = _preamble_events(seen)[0]["text"]
    saklanan = [m.content for m in memory.get("s1")]
    assert preamble_text not in saklanan
    assert saklanan == ["PN1099 ağırlığı?", "cevap: PN1099 ağırlığı?"]


def test_preamble_pinli_yavas_flowda_o_flowun_havuzundan_basar():
    flow = _FakePinnableFlow()
    router = Router({"comparison": flow}, Routing(default_flow="comparison", intents={}))
    store = SessionStateStore()
    store.set("s1", SessionState(pinned=PinnedFlow(flow="comparison", intent="comparison")))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        session_state=store, preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "peki fiyatları?", on_trace))

    events = _preamble_events(seen)
    assert len(events) == 1
    assert events[0]["pool"] == "flow:comparison"
    # It must have been printed AFTER the pin check -- saying "comparing"
    # for a pin that might break would be an unfulfilled promise.
    steps = [step for step, _ in seen]
    assert steps.index("pin_check") < steps.index("preamble")


def test_preamble_pinli_hizli_flowda_hic_basilmaz():
    flow = _FakePinnableFlow()
    router = Router({"doc_download": flow}, Routing(default_flow="doc_download", intents={}))
    store = SessionStateStore()
    store.set("s1", SessionState(pinned=PinnedFlow(flow="doc_download", intent="doc_download")))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        session_state=store, preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "kullanım kılavuzu", on_trace))

    assert _preamble_events(seen) == []


def test_preamble_pin_kirilirsa_SUSAR():
    """On the escape-hatch path the flow is no longer KNOWN. The bubble must
    NOT say "comparing"; with no evidence it prints nothing at all (silence
    is the default). If the turn drags on, the watchdog catches it anyway."""
    flow = _FakePinnableFlow(holds=False)
    router = Router({"comparison": flow}, Routing(default_flow="comparison", intents={}))
    store = SessionStateStore()
    store.set("s1", SessionState(pinned=PinnedFlow(flow="comparison", intent="comparison")))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        session_state=store, preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "bambaşka bir konu", on_trace))

    assert _preamble_events(seen) == []


def test_preamble_selamlasmada_hic_basmaz():
    """The original complaint: even saying hello triggered "working on your
    request". End to end: full pipeline + signal-free message -> no
    bubble at all."""
    flow = _FakeFlow()
    router = Router({"default_topn": flow}, Routing(default_flow="default_topn", intents={}))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.OUT_OF_SCOPE), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "selam", on_trace))

    assert _preamble_events(seen) == []


def test_preamble_handoff_yolunda_hedef_flowun_havuzundan_basar():
    target = _FakeFlow()
    fallback = _FakeFlow()
    router = Router(
        {"comparison": target, "default_topn": fallback},
        Routing(default_flow="default_topn", intents={}),
    )
    store = SessionStateStore()
    store.set("s1", SessionState(
        suggested_next_flow="comparison", suggested_next_candidates=["DE1", "DE2"],
    ))
    orch = Orchestrator(
        classifier=_FakeClassifier(IntentLabel.PRODUCT_FACT), router=router,
        answering_model=_FakeAnsweringModel(), memory=ConversationMemory(),
        session_state=store, preamble=_settings(),
    )
    seen, on_trace = _collector()

    asyncio.run(orch.handle("s1", "evet", on_trace))

    assert target.calls  # the handoff really ran
    events = _preamble_events(seen)
    assert len(events) == 1
    assert events[0]["pool"] == "flow:comparison"
