"""`templates/index.html` -- silent-failure points of the palette switcher.

This template broke production twice, both times in testable ways:

  1. `body[data-view="widget"] .palette-fab { display: none }` fully hid the
     palette button in the widget preview -- silently violating the
     requirement "the switcher must also work on the widget screen"
     (no error, just a missing button).
  2. The `web` service has a healthcheck probing the `GET /` route: a broken
     Jinja template now means not just a 500, but an unhealthy container ->
     crash loop -> Coolify stopping the ENTIRE application after 10 restarts.

CSS/JS parity is also a silent trap: a palette in the list but missing from
CSS shows up as "I click and nothing happens"; a palette in CSS but not in
the list is unreachable. Neither raises an error.

The tests do NOT run a browser -- they measure the template text and the
Jinja render.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
_INDEX = _TEMPLATES / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _css_palette_blocks(html: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for m in re.finditer(r':root\[data-palette="([a-z0-9]+)"\]\s*\{(.*?)\}', html, re.DOTALL):
        out[m.group(1)] = set(re.findall(r"(--[a-z0-9-]+)\s*:", m.group(2)))
    return out


def _js_palette_ids(html: str) -> list[str]:
    blok = re.search(r"const PALETTES = \[(.*?)\];", html, re.DOTALL)
    assert blok, "PALETTES array not found -- it is the switcher's data source"
    return re.findall(r"id:\s*'([a-z0-9]+)'", blok.group(1))


def test_sablon_jinja_ile_render_olur():
    """The `web` healthcheck hits `GET /` -- a broken template means a crash loop."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)))
    cikti = env.get_template("index.html").render()
    assert len(cikti) > 5000, "render came out empty"
    assert "palette-toggle" in cikti


def test_css_ve_js_palet_kumeleri_ayni(html):
    css = set(_css_palette_blocks(html))
    js = set(_js_palette_ids(html))
    assert css == js, (
        f"in CSS but not JS: {sorted(css - js)} | "
        f"in JS but not CSS: {sorted(js - css)}"
    )


def test_js_listesinde_yinelenen_id_yok(html):
    ids = _js_palette_ids(html)
    assert len(ids) == len(set(ids)), f"duplicate palette id: {ids}"


def test_en_az_on_bes_palet_var(html):
    """Product requirement: ~15 candidate palettes, not a handful."""
    assert len(_js_palette_ids(html)) >= 15


def test_her_palet_ayni_degisken_kumesini_tanimlar(html):
    """A variable left out of one palette (e.g. `--accent`) is INHERITED from
    the previously applied palette when this one is selected -- wrong color,
    no error."""
    bloklar = _css_palette_blocks(html)
    assert bloklar, "no palette blocks found"
    ortak = set.intersection(*bloklar.values())
    hatali = {pid: sorted(v ^ ortak) for pid, v in bloklar.items() if v != ortak}
    assert not hatali, f"palettes with a differing variable set: {hatali}"


def test_varsayilan_palet_listede(html):
    m = re.search(r"<html[^>]*data-palette=\"([a-z0-9]+)\"", html)
    assert m, "no data-palette on <html> -- initial paint cannot be done"
    assert m.group(1) in _js_palette_ids(html)


def test_widget_gorunumunde_palet_anahtari_gizlenmiyor(html):
    """Regression lock: this rule was once `display: none`."""
    kurallar = re.findall(
        r'body\[data-view="widget"\][^{]*\.palette-(?:fab|toast)[^{]*\{([^}]*)\}', html)
    for govde in kurallar:
        assert "display: none" not in govde.replace(" ", " "), (
            f"widget view hides the palette switcher: {govde.strip()}")
    birlesik = re.findall(
        r'body\[data-view="widget"\]([^{]*)\{([^}]*display:\s*none[^}]*)\}', html)
    for secici, govde in birlesik:
        assert "palette" not in secici, (
            f"palette switcher hidden in widget: {secici.strip()} {{{govde.strip()}}}")


def test_palet_secimi_kalici(html):
    """The selection must survive a page refresh; `localStorage` access must
    also be inside try/catch (in a private tab access THROWS and the ENTIRE
    page fails to render)."""
    assert "localStorage.setItem('chat-palette'" in html
    assert "localStorage.getItem('chat-palette'" in html
    for cagri in ("localStorage.setItem('chat-palette'", "localStorage.getItem('chat-palette'"):
        i = html.index(cagri)
        pencere = html[max(0, i - 400):i + 200]
        assert "try {" in pencere, f"{cagri} is OUTSIDE try/catch"


def test_anahtar_dugmesinin_erisilebilir_adi_var(html):
    m = re.search(r'<button id="palette-toggle"(.*?)>', html, re.DOTALL)
    assert m, "no palette-toggle button"
    assert "aria-label" in m.group(1)
