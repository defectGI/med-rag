"""Pre-info ("preamble") bubbles: so the user isn't staring at an empty screen
while waiting for a long turn.

Measured from `timing` trace events in `logs/conversations/` (10 turns):
`sql` (text2sql linking+generation) averages 18.5s, a full turn averages
14.6s, worst 32.6s. The webapp's live trace panel makes this wait visible,
but on WhatsApp the user sees nothing -- this module fills that gap.

## Why NO LLM

The preamble is DETERMINISTIC, templated and chosen from a pool. The single
GPU runs serially (see `gpu_gate.py`); an extra model call just for the
preamble would delay the very answer we're trying to speed up. The cheap
signals already in hand (pinned flow, handoff offer, product-code regex) are
enough to produce a well-informed sentence.

## Pools, language, repetition

Every pool carries TR and EN variants; the language is detected
deterministically from the user's MESSAGE (Turkish-specific letters + two
small stop-word sets), default `tr` (the product is Turkish). Each pool has
multiple sentences and one is chosen at random -- a single fixed sentence
felt robotic after a few turns. Which pool the chosen sentence came from is
carried in the emitted event (`pool`) and recorded in the conversation log --
so pool frequencies can be measured later.

## When to STAY SILENT (the most important rule)

**Silence is the default; the early bubble is an EVIDENCE-backed exception.**
The first version printed from the generic pool even with no signal and looked
absurd on "hi"/"thanks" turns (a user even getting "working on your request"
when greeting -- printing should be limited to few scenarios).

The early bubble prints in exactly two cases: (1) a pinned/handoff turn -- a
flow WILL run -- and that flow is not in `fast_flows`; (2) a product code
(`PN1099`) occurs in the message. Never otherwise.

This is safe because the cost of being wrong is bounded: if a signal-less turn
unexpectedly runs long, the watchdog engages at second 12 anyway. The two
mechanisms deliberately complement each other -- **the early bubble demands
CERTAINTY, the watchdog demands TIME**; no turn stays infinitely silent.

The `fast_flows` silencing does NOT cover the WATCHDOG (see
`watchdog_preamble`): that list is a GUESS, and the watchdog is the safety net
that catches when the guess fails.

## Known limit: the watchdog prints ONCE

`comparison`/`aggregation` call `SqlTopNFlow.run_pass` N times and the SQL
retriever is SERIALIZED with `sql_concurrency = 1` (`gpu_gate.py`) -- so N
passes are N × ~18.5s; `asyncio.gather` doesn't parallelize it. A two-product
comparison can take ~37s, longer still with an angle breakdown. Today the
watchdog prints only ONCE (at second 12); afterwards the user falls back into a
long silence. A repeating watchdog was deliberately NOT done (message-spam
risk on WhatsApp) -- left as an open decision.

## Limits (non-negotiable)

- The preamble does NOT enter `ConversationMemory`. If it did, `reconciler`
  would mistake it for a real assistant answer next turn (see the
  orchestrator.py module docstring: only REAL question/answers enter memory).
- The preamble NEVER goes to the answering model -- it exits via the `on_trace`
  channel, i.e. it's on the same side of the "flow internal steps don't leak
  to the model" rule.
- A bubble text must NEVER promise work that won't happen: pool selection
  binds only to signals we TRULY have (pinned flow / product code); with an
  unknown flow name it falls back to the generic pool.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field

logger = logging.getLogger("medrag.api.preamble")

#: `on_trace` step name -- the webapp turns it into a chat bubble, `wa_bot`
#: into a separate WhatsApp message; both listen to the same event.
PREAMBLE_STEP = "preamble"

#: The `stage` field of the emitted event.
STAGE_EARLY = "early"
STAGE_WATCHDOG = "watchdog"

# Same shape as `flows/doc_download.py::_PRODUCT_CODE_RE` (that file notes it
# also matches `flows/comparison.py`) -- used here ONLY for the "did the user
# write a product code" signal, never for retrieval decisions.
_PRODUCT_CODE_RE = re.compile(r"\bDE[\s-]?\d{3,7}\b", re.IGNORECASE)

# --- language detection -------------------------------------------------

_TURKISH_CHARS = set("ğüşıöçĞÜŞİÖÇ")

# Short, high-frequency stop words. The goal is not language classification
# but a cheap answer to "is this message English" -- if it lands on the wrong
# side, only the preamble's language is affected, never the real answer.
_TR_WORDS = frozenset(
    ["ne", "nedir", "nasil", "nasıl", "hangi", "kac", "kaç", "icin", "için", "var", "yok", "mi", "mı", "mu", "mü", "bana", "lutfen", "lütfen", "bir", "bu", "su", "şu", "ile", "ve", "veya", "degil", "değil", "olan", "olur", "gibi", "daha", "en", "urun", "ürün", "fiyat", "teklif", "belge", "dosya", "ozellik", "özellik", "karsilastir", "karşılaştır", "goster", "göster", "listele", "istiyorum", "lazim", "lazım", "merhaba", "selam", "tesekkur", "teşekkür", "evet", "hayir", "hayır", "tamam", "tabii", "tabi", "peki", "hem", "de", "ama", "sonra", "simdi", "şimdi"]
)
_EN_WORDS = frozenset(
    ["what", "which", "how", "many", "much", "is", "are", "the", "a", "an", "do", "does", "can", "could", "would", "you", "i", "me", "my", "we", "our", "for", "with", "and", "or", "not", "of", "to", "in", "on", "it", "this", "that", "please", "need", "want", "show", "list", "send", "give", "tell", "about", "between", "compare", "price", "quote", "document", "file", "spec", "specs", "feature", "features", "hello", "hi", "thanks", "thank", "yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "sounds", "good", "got", "right", "correct", "also", "then", "but", "more", "another", "other", "one", "both"]
)

_WORD_RE = re.compile(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]+")

LANG_TR = "tr"
LANG_EN = "en"


def detect_language(text: str) -> str:
    """Detects the message's language as `"tr"`/`"en"` -- NO LLM, pure code.

    If any Turkish-specific letter (ğ/ş/ı/ç ...) occurs, immediately `tr`.
    Otherwise two small stop-word sets are counted; `en` only when English is
    CLEARLY ahead, and `tr` in every other case including ties/ambiguity (the
    product is Turkish, that's the default). Deliberately asymmetric: wrongly
    speaking English is more jarring than wrongly speaking Turkish."""
    if any(ch in _TURKISH_CHARS for ch in text):
        return LANG_TR
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return LANG_TR
    tr_hits = sum(1 for w in words if w in _TR_WORDS)
    en_hits = sum(1 for w in words if w in _EN_WORDS)
    return LANG_EN if en_hits > tr_hits else LANG_TR


# --- pools ---------------------------------------------------------------

# `{code}` is used only in the `product_code` pool and applied with
# `str.replace` (NOT `str.format` -- an unescaped brace somewhere in the text
# would blow up at runtime; this is a user-visible path, it must be silent and
# resilient).
_POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": {
        LANG_TR: (
            "Talebinizi aldım, ilgili kaynakları inceliyorum…",
            "Sorgunuz için kayıtlarımızı tarıyorum…",
            "İlgili bilgileri derliyorum…",
            "Talebiniz üzerinde çalışıyorum…",
            "Kaynaklarımızı kontrol ediyorum…",
            "Konuyla ilgili verileri topluyorum…",
        ),
        LANG_EN: (
            "Received — I'm reviewing the relevant sources…",
            "Searching our records for your request…",
            "Compiling the relevant information…",
            "Working on your request…",
            "Checking our sources…",
            "Gathering the data on this…",
        ),
    },
    "product_code": {
        LANG_TR: (
            "{code} kaydını inceliyorum…",
            "{code} için kayıtlarımızı tarıyorum…",
            "{code} verilerini derliyorum…",
            "{code} ile ilgili bilgileri kontrol ediyorum…",
        ),
        LANG_EN: (
            "Reviewing the record for {code}…",
            "Searching our records for {code}…",
            "Compiling the data on {code}…",
            "Checking the information for {code}…",
        ),
    },
    "flow:comparison": {
        LANG_TR: (
            "Karşılaştırma için her iki ürünün verilerini derliyorum…",
            "Karşılaştırma tablosu için teknik verileri topluyorum…",
            "Ürünler arasındaki farkları inceliyorum…",
        ),
        LANG_EN: (
            "Compiling the data for both products…",
            "Gathering the technical figures for the comparison…",
            "Reviewing the differences between the products…",
        ),
    },
    "flow:recommendation": {
        LANG_TR: (
            "Belirttiğiniz kriterlere uygun ürünleri tarıyorum…",
            "İhtiyacınıza uygun seçenekleri değerlendiriyorum…",
            "Uygun ürünleri belirlemek için kayıtlarımızı inceliyorum…",
        ),
        LANG_EN: (
            "Searching for products that match your criteria…",
            "Evaluating the options that fit your requirements…",
            "Reviewing our range to identify suitable products…",
        ),
    },
    "flow:sql_topn": {
        LANG_TR: (
            "Teknik özellik kayıtlarını sorguluyorum…",
            "Ürün veri tabanında ilgili değerleri arıyorum…",
            "Teknik verileri derliyorum…",
        ),
        LANG_EN: (
            "Querying the technical specification records…",
            "Looking up the relevant values in the product database…",
            "Compiling the technical data…",
        ),
    },
    "flow:aggregation": {
        LANG_TR: (
            "İlgili kategorileri sırayla tarıyorum…",
            "Sonuçları kategori bazında derleyip birleştiriyorum…",
        ),
        LANG_EN: (
            "Scanning the relevant categories in turn…",
            "Compiling and consolidating the results by category…",
        ),
    },
    # Watchdog pools -- printed only when the turn REALLY runs long. Tone
    # rule: not an apology or a request, a STATUS UPDATE (user feedback:
    # "please be patient a little longer... that's a terrible message, can't it
    # be more professional"). Phrases like "patience", "don't go anywhere",
    # "almost there" DO NOT enter these pools.
    "slow:sql": {
        LANG_TR: (
            "Sorgu devam ediyor, kısa süre içinde tamamlanacak…",
            "Veri taraması sürüyor, birkaç saniye daha gerekiyor…",
            "Teknik veriler üzerinde çalışmaya devam ediyorum…",
        ),
        LANG_EN: (
            "The query is still running and should complete shortly…",
            "The data search is ongoing — a few more seconds…",
            "Still processing the technical data…",
        ),
    },
    "slow:generic": {
        LANG_TR: (
            "İşlem beklenenden uzun sürüyor, çalışmaya devam ediyorum…",
            "Talebiniz üzerinde çalışmaya devam ediyorum…",
            "Derleme sürüyor, kısa süre içinde tamamlanacak…",
        ),
        LANG_EN: (
            "This is taking longer than usual — still working on it…",
            "Still working on your request…",
            "Compilation is in progress and should finish shortly…",
        ),
    },
}

#: Which pool a flow name falls into during the watchdog stage. Every flow not
#: listed falls to `slow:generic` (adding a new flow requires no line here --
#: it silently uses the generic pool and never produces a wrong promise).
_SLOW_POOL_BY_FLOW = {
    "sql_topn": "slow:sql",
    "comparison": "slow:sql",
    "aggregation": "slow:sql",
}

#: Which pool to use in the early stage when the flow name IS KNOWN (pinned/
#: handoff). No match -> the generic pool.
_EARLY_POOL_BY_FLOW = {
    "comparison": "flow:comparison",
    "recommendation": "flow:recommendation",
    "sql_topn": "flow:sql_topn",
    "aggregation": "flow:aggregation",
}


@dataclass(frozen=True)
class PreambleSettings:
    """Runtime settings derived from `config/default.toml [preamble]`.

    Given OPTIONALLY to the `Orchestrator`; if omitted (`None`) no preamble is
    produced at all -- same pattern as the `reconciler`/`session_state`
    optionality in this component, zero backward behavior change."""

    watchdog_seconds: float
    #: At t≈0, if we KNOW the flow is in this list, NO early bubble is
    #: printed. Does NOT affect the watchdog (see the `watchdog_preamble`
    #: docstring): this list is a guess, the watchdog is the safety net that
    #: catches when the guess fails.
    fast_flows: frozenset[str] = field(default_factory=frozenset)


def _pick(pool: str, lang: str, rng: random.Random) -> str:
    """A random sentence from the pool. Unknown pool/language falls back to
    generic TR -- this path is user-visible and must not raise `KeyError`."""
    by_lang = _POOLS.get(pool) or _POOLS["generic"]
    options = by_lang.get(lang) or by_lang[LANG_TR]
    return rng.choice(options)


def _event(pool: str, lang: str, stage: str, rng: random.Random, code: str | None = None) -> dict:
    text = _pick(pool, lang, rng)
    if code:
        text = text.replace("{code}", code)
    return {"text": text, "pool": pool, "lang": lang, "stage": stage}


def early_preamble(
    query: str,
    *,
    known_flow: str | None,
    settings: PreambleSettings,
    rng: random.Random | None = None,
) -> dict | None:
    """The bubble to print at t≈0; `None` when nothing should print.

    **Rule: silence is the default, speaking is EVIDENCE-backed.** The earlier
    version printed from the generic pool even with no signal -- on turns like
    "hi"/"thanks" that was both wrong and annoying. Now the early bubble prints
    only when there is evidence IN HAND that the turn will really be long:

    1. `known_flow` filled (pinned/handoff turn -- a flow WILL run) and not
       in `fast_flows`, OR
    2. a product code occurs in the message (`PN1099` -- nearly always goes to
       a data/spec query, i.e. the SQL chain).

    None of these -> `None`: the turn starts silently. This is SAFE because if
    we're wrong (a signal-less turn runs long), `watchdog_preamble` already
    engages at second 12 -- worst case a slightly late bubble, never infinite
    silence. The two mechanisms deliberately pair this way: the early bubble
    demands CERTAINTY, the watchdog demands TIME."""
    rng = rng or random
    lang = detect_language(query)
    if known_flow is not None:
        if known_flow in settings.fast_flows:
            return None
        # Pinned/handoff: a flow WILL run. Not in the map (a new flow) ->
        # fall back to generic -- we KNOW retrieval is running, we just have no
        # sentence specific to it.
        pool = _EARLY_POOL_BY_FLOW.get(known_flow, "generic")
        return _event(pool, lang, STAGE_EARLY, rng)
    match = _PRODUCT_CODE_RE.search(query)
    if match:
        # Show the user-typed code NORMALIZED, not AS-TYPED ("de 1405" ->
        # "PN1099") -- if the bubble text references a product code, it should
        # look like a real one.
        code = re.sub(r"[\s-]", "", match.group(0)).upper()
        return _event("product_code", lang, STAGE_EARLY, rng, code=code)
    return None


def watchdog_preamble(
    *,
    flow: str | None,
    lang: str,
    settings: PreambleSettings,
    rng: random.Random | None = None,
) -> dict:
    """The second bubble, printed when the threshold is crossed.

    `flow` is the flow name learned so far (from the router's
    `routed`/`routed_default_fallback` trace event on the full pipeline; known
    from the start on the pinned path).

    **`fast_flows` is NOT APPLIED HERE** -- that list is a GUESS and it
    silences the early bubble; the watchdog exists precisely to catch the guess
    being WRONG. If `doc_download` unexpectedly takes 30 seconds, the user
    should see something at second 12 -- for the first time; staying silent
    "because it was supposed to be fast" would be silence at exactly the
    longest point. Hence this function NEVER returns `None`."""
    rng = rng or random
    pool = _SLOW_POOL_BY_FLOW.get(flow or "", "slow:generic")
    return _event(pool, lang, STAGE_WATCHDOG, rng)


#: The routing events `Router.flow_for` publishes (see router.py) -- the
#: watchdog learns which flow was entered by the time from THESE events, without
#: chaining a new parameter.
_ROUTE_STEPS = ("routed", "routed_default_fallback")


class PreambleTicker:
    """One turn's preamble lifecycle: early bubble + watchdog bubble.

    `Orchestrator.handle` constructs it at turn start but DELIBERATELY delays
    `start()` -- which flow will run is only HONESTLY known after the pin check
    (~0.5s) or the handoff approval passes. That way the "comparing products"
    bubble prints only if a comparison will really happen; if the pin breaks,
    the full-pipeline path starts with `known_flow=None` and falls to the
    generic pool.

    If `settings` or `on_trace` is missing it is FULLY passive (`trace`
    returns the raw `on_trace` unchanged) -- same pattern as the
    `reconciler`/`session_state` optionality in `Orchestrator`.
    """

    def __init__(
        self,
        query: str,
        on_trace,
        settings: PreambleSettings | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._query = query
        self._on_trace = on_trace
        self._settings = settings
        self._rng = rng
        self.active = settings is not None and on_trace is not None
        self._lang = detect_language(query) if self.active else LANG_TR
        self._flow: str | None = None
        self._task: asyncio.Task | None = None

    @property
    def trace(self):
        """The `on_trace` passed down (to the flows). When passive it IS the
        raw callback (zero wrapping); when active it's a wrapper that listens
        to routing events and forwards them verbatim."""
        return self._sniff if self.active else self._on_trace

    def _sniff(self, step: str, data) -> None:
        if step in _ROUTE_STEPS and isinstance(data, str):
            self._flow = data
        self._on_trace(step, data)

    def _emit(self, event: dict | None) -> None:
        """Publishing NEVER breaks the turn -- the preamble is an observational
        addition; an error on the `on_trace` side is logged and the flow
        continues."""
        if event is None:
            return
        try:
            self._on_trace(PREAMBLE_STEP, event)
        except Exception:
            logger.exception("preamble yayınlanamadı")

    def start(self, *, known_flow: str | None = None) -> None:
        if not self.active:
            return
        assert self._settings is not None
        self._flow = known_flow
        self._emit(
            early_preamble(
                self._query, known_flow=known_flow, settings=self._settings, rng=self._rng
            )
        )
        if self._settings.watchdog_seconds > 0:
            self._task = asyncio.create_task(self._watchdog())

    async def _watchdog(self) -> None:
        assert self._settings is not None
        try:
            await asyncio.sleep(self._settings.watchdog_seconds)
        except asyncio.CancelledError:
            return
        self._emit(
            watchdog_preamble(
                flow=self._flow, lang=self._lang, settings=self._settings, rng=self._rng
            )
        )

    def stop(self) -> None:
        """Called when the turn ends (always from a `finally`) -- a watchdog
        that hasn't reached its threshold is cancelled, so short turns never
        print the second bubble at all."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None


def pool_names() -> list[str]:
    """Defined pool names -- for tests/analysis scripts to validate the `pool`
    field."""
    return sorted(_POOLS)


__all__ = [
    "LANG_EN",
    "LANG_TR",
    "PREAMBLE_STEP",
    "STAGE_EARLY",
    "STAGE_WATCHDOG",
    "PreambleSettings",
    "PreambleTicker",
    "detect_language",
    "early_preamble",
    "pool_names",
    "watchdog_preamble",
]
