"""`medrag.api.preamble` -- preamble bubbles.

No test touches a real LLM/network: the whole module is deterministic pure
code (that is the feature's core decision -- an extra model call would delay
the real answer; see the module docstring).
"""

from __future__ import annotations

import asyncio
import random

from medrag.api.preamble import (
    LANG_EN,
    LANG_TR,
    PREAMBLE_STEP,
    STAGE_EARLY,
    STAGE_WATCHDOG,
    PreambleSettings,
    PreambleTicker,
    detect_language,
    early_preamble,
    pool_names,
    watchdog_preamble,
)

_SETTINGS = PreambleSettings(
    watchdog_seconds=12.0,
    fast_flows=frozenset({"no_retrieval", "doc_download", "default_topn"}),
)


# --- language detection --------------------------------------------------------


def test_turkce_ozel_harf_dogrudan_tr():
    assert detect_language("PN1099 için ağırlık kaç?") == LANG_TR


def test_ingilizce_durak_kelimeleri_en():
    assert detect_language("what is the weight of PN1099") == LANG_EN


def test_kisa_onay_mesajlari_dogru_dile_gider():
    """Pinned/handoff turns consist exactly of these short affirmations --
    when `yes`/`ok` were missing from the word set, such messages counted as
    signal-free and fell back to `tr`."""
    assert detect_language("yes, stainless") == LANG_EN
    assert detect_language("ok sounds good") == LANG_EN
    assert detect_language("no, another one") == LANG_EN
    assert detect_language("evet olur") == LANG_TR
    assert detect_language("tamam peki") == LANG_TR


def test_kararsiz_mesaj_tr_varsayilanina_duser():
    # No stop word at all -> stay in Turkish, the product's language
    # (deliberate asymmetry: accidentally speaking English is more jarring).
    assert detect_language("PN1099") == LANG_TR
    assert detect_language("") == LANG_TR


# --- early bubble ---------------------------------------------------------------


def test_erken_balon_urun_kodunu_normalize_edip_metne_koyar():
    event = early_preamble("de 1405 belgesi", known_flow=None, settings=_SETTINGS)
    assert event is not None
    assert event["pool"] == "product_code"
    assert event["stage"] == STAGE_EARLY
    assert "PN1099" in event["text"]
    assert "{code}" not in event["text"]


def test_erken_balon_sinyal_yoksa_SUSAR():
    """Silence is the default (a greeting shouldn't trigger "working on your
    request"). No product code + unknown flow -> nothing is printed; if the
    turn drags on, the watchdog catches it."""
    assert early_preamble("bana bir şey öner", known_flow=None, settings=_SETTINGS) is None


def test_selamlasma_ve_tesekkur_turlarinda_hic_balon_yok():
    """The exact scenario the complaint came from."""
    for mesaj in ("selam", "merhaba", "teşekkürler", "günaydın", "hello",
                  "thanks", "iyi günler", "ok"):
        assert early_preamble(mesaj, known_flow=None, settings=_SETTINGS) is None


def test_erken_balon_hizli_flow_biliniyorsa_hic_basilmaz():
    for flow in ("no_retrieval", "doc_download", "default_topn"):
        assert early_preamble("selam", known_flow=flow, settings=_SETTINGS) is None


def test_erken_balon_yavas_flow_biliniyorsa_o_flowun_havuzu():
    event = early_preamble("evet", known_flow="comparison", settings=_SETTINGS)
    assert event is not None
    assert event["pool"] == "flow:comparison"


def test_erken_balon_bilinmeyen_flow_jenerige_duser():
    # Adding a new flow must NOT require adding a line to the pool map --
    # an unmatched name silently falls back to the generic pool, avoiding a
    # false promise.
    event = early_preamble("x", known_flow="brand_new_flow", settings=_SETTINGS)
    assert event is not None
    assert event["pool"] == "generic"


def test_ingilizce_mesaj_ingilizce_havuzdan_secer():
    event = early_preamble("what is the price of PN1099", known_flow=None, settings=_SETTINGS)
    assert event is not None
    assert event["lang"] == LANG_EN
    assert "PN1099" in event["text"]


def test_havuz_tek_cumleli_degil():
    """Decision: the deterministic message must not be a single fixed string;
    draw it from a pool."""
    seen = {
        early_preamble("PN1099 nedir", known_flow=None, settings=_SETTINGS)["text"]
        for _ in range(200)
    }
    assert len(seen) > 1


def test_her_havuz_iki_dili_de_tasir():
    for pool in pool_names():
        for lang in (LANG_TR, LANG_EN):
            texts = [
                _pick_via_public_api(pool, lang)
                for _ in range(1)
            ]
            assert texts and texts[0]


def _pick_via_public_api(pool: str, lang: str) -> str:
    from medrag.api.preamble import _POOLS  # in-test access: pool completeness check

    assert lang in _POOLS[pool], f"{lang} missing from pool {pool}"
    assert _POOLS[pool][lang], f"{pool}/{lang} is empty"
    return _POOLS[pool][lang][0]


# --- watchdog ------------------------------------------------------------


def test_watchdog_sql_flowlari_icin_ozel_havuz():
    for flow in ("sql_topn", "comparison", "aggregation"):
        event = watchdog_preamble(flow=flow, lang=LANG_TR, settings=_SETTINGS)
        assert event is not None
        assert event["pool"] == "slow:sql"
        assert event["stage"] == STAGE_WATCHDOG


def test_watchdog_bilinmeyen_flow_jenerik_yavas_havuz():
    event = watchdog_preamble(flow=None, lang=LANG_EN, settings=_SETTINGS)
    assert event is not None
    assert event["pool"] == "slow:generic"
    assert event["lang"] == LANG_EN


def test_watchdog_hizli_flowda_da_BASAR():
    """`fast_flows` only silences the EARLY bubble. That list is a GUESS; the
    watchdog's whole reason to exist is catching the guess being WRONG. If
    `doc_download` takes 30 seconds, the user must see something -- for the
    first time -- at second 12."""
    for flow in ("no_retrieval", "doc_download", "default_topn"):
        event = watchdog_preamble(flow=flow, lang=LANG_TR, settings=_SETTINGS)
        assert event is not None
        assert event["stage"] == STAGE_WATCHDOG
        assert event["pool"] == "slow:generic"


def test_hizli_flowda_erken_susar_ama_watchdog_yine_basar():
    """End to end: the early bubble is never printed, but if the turn drags
    on, the watchdog bubble becomes the FIRST message the user sees."""
    events, on_trace = _collect()
    settings = PreambleSettings(
        watchdog_seconds=0.01, fast_flows=frozenset({"doc_download"})
    )

    async def run():
        ticker = PreambleTicker("kullanım kılavuzu", on_trace, settings)
        ticker.start(known_flow="doc_download")
        await asyncio.sleep(0.05)
        ticker.stop()

    asyncio.run(run())
    preambles = [data for step, data in events if step == PREAMBLE_STEP]
    assert len(preambles) == 1
    assert preambles[0]["stage"] == STAGE_WATCHDOG


# --- ticker (lifecycle) ---------------------------------------------------------


def _collect() -> tuple[list[tuple[str, object]], object]:
    events: list[tuple[str, object]] = []

    def on_trace(step: str, data) -> None:
        events.append((step, data))

    return events, on_trace


def test_ticker_ayarsizken_tamamen_pasif():
    """`settings=None` -> `trace` is the RAW callback ITSELF (zero wrapping),
    no events are published -- byte-for-byte the pre-feature behavior."""
    events, on_trace = _collect()
    ticker = PreambleTicker("merhaba", on_trace, None)
    assert ticker.trace is on_trace
    ticker.start(known_flow=None)
    ticker.stop()
    assert events == []


def test_ticker_on_trace_yokken_de_pasif():
    ticker = PreambleTicker("merhaba", None, _SETTINGS)
    assert ticker.trace is None
    ticker.start(known_flow=None)  # must not raise
    ticker.stop()


def test_ticker_erken_balonu_hemen_yayinlar():
    events, on_trace = _collect()

    async def run():
        ticker = PreambleTicker("PN1099 ağırlığı?", on_trace, _SETTINGS)
        ticker.start(known_flow=None)
        ticker.stop()

    asyncio.run(run())
    assert [step for step, _ in events] == [PREAMBLE_STEP]
    assert events[0][1]["stage"] == STAGE_EARLY


def test_ticker_kisa_turda_watchdog_hic_basmaz():
    events, on_trace = _collect()

    async def run():
        ticker = PreambleTicker("PN1099 nedir", on_trace, _SETTINGS)
        ticker.start(known_flow=None)
        await asyncio.sleep(0)  # the turn ended immediately
        ticker.stop()
        await asyncio.sleep(0.02)  # give the cancelled watchdog a chance

    asyncio.run(run())
    stages = [data["stage"] for step, data in events if step == PREAMBLE_STEP]
    assert stages == [STAGE_EARLY]


def test_ticker_uzun_turda_watchdog_basar_ve_flowu_trace_ten_ogrenir():
    events, on_trace = _collect()
    settings = PreambleSettings(watchdog_seconds=0.01, fast_flows=frozenset())

    async def run():
        ticker = PreambleTicker("PN1099 nedir", on_trace, settings)
        ticker.start(known_flow=None)
        # The Router's real event -- the ticker must listen to this and learn the flow.
        ticker.trace("routed", "sql_topn")
        await asyncio.sleep(0.05)
        ticker.stop()

    asyncio.run(run())
    preambles = [data for step, data in events if step == PREAMBLE_STEP]
    assert len(preambles) == 2
    assert preambles[1]["stage"] == STAGE_WATCHDOG
    assert preambles[1]["pool"] == "slow:sql"
    # The wrapper must have forwarded the router event VERBATIM (observational, never swallows).
    assert ("routed", "sql_topn") in events


def test_ticker_trace_sarmalayicisi_her_olayi_aynen_iletir():
    events, on_trace = _collect()
    ticker = PreambleTicker("PN1099 nedir", on_trace, _SETTINGS)
    ticker.trace("intent", "product_fact")
    ticker.trace("timing", {"stage": "sql", "seconds": 18.5})
    assert events == [
        ("intent", "product_fact"),
        ("timing", {"stage": "sql", "seconds": 18.5}),
    ]


def test_ticker_on_trace_patlarsa_tur_dusmez():
    def kirik(step: str, data) -> None:
        raise RuntimeError("panel koptu")

    async def run():
        ticker = PreambleTicker("PN1099 nedir", kirik, _SETTINGS)
        ticker.start(known_flow=None)  # must be swallowed
        ticker.stop()

    asyncio.run(run())  # no exception should propagate


def test_watchdog_kapaliyken_gorev_hic_kurulmaz():
    events, on_trace = _collect()
    settings = PreambleSettings(watchdog_seconds=0.0, fast_flows=frozenset())

    async def run():
        ticker = PreambleTicker("PN1099 nedir", on_trace, settings)
        ticker.start(known_flow=None)
        await asyncio.sleep(0.03)
        ticker.stop()

    asyncio.run(run())
    stages = [data["stage"] for step, data in events if step == PREAMBLE_STEP]
    assert stages == [STAGE_EARLY]


def test_rng_enjekte_edilebilir_deterministik():
    a = early_preamble("PN1099", known_flow=None, settings=_SETTINGS, rng=random.Random(7))
    b = early_preamble("PN1099", known_flow=None, settings=_SETTINGS, rng=random.Random(7))
    assert a == b


def test_sinyalsiz_uzun_turda_YALNIZ_watchdog_basar():
    """The early bubble stays silent (no evidence) but if the turn drags on
    the watchdog still speaks -- the end-to-end form of the
    "evidence-request / time-request" pairing."""
    events, on_trace = _collect()
    settings = PreambleSettings(watchdog_seconds=0.01, fast_flows=frozenset())

    async def run():
        ticker = PreambleTicker("bana bir şey öner", on_trace, settings)
        ticker.start(known_flow=None)
        await asyncio.sleep(0.05)
        ticker.stop()

    asyncio.run(run())
    preambles = [d for step, d in events if step == PREAMBLE_STEP]
    assert len(preambles) == 1
    assert preambles[0]["stage"] == STAGE_WATCHDOG


def test_sinyalsiz_kisa_turda_HICBIR_sey_basilmaz():
    """The whole greeting turn: neither the early bubble nor the watchdog."""
    events, on_trace = _collect()

    async def run():
        ticker = PreambleTicker("selam", on_trace, _SETTINGS)
        ticker.start(known_flow=None)
        ticker.stop()
        await asyncio.sleep(0.02)

    asyncio.run(run())
    assert [d for step, d in events if step == PREAMBLE_STEP] == []
