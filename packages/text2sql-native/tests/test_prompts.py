"""Prompt template loading tests, including the packaged fallback."""

from __future__ import annotations

import pytest

from text2sql_native.errors import ConfigError
from text2sql_native.prompts import PromptTemplates

from .conftest import PROMPT_DIR


def test_load_from_explicit_directory():
    templates = PromptTemplates(PROMPT_DIR)
    tmpl = templates.load("linking.txt")
    assert tmpl.system
    assert tmpl.user


def test_packaged_fallback_when_no_directory():
    # No directory given -> use templates packaged inside text2sql_native.prompts.
    templates = PromptTemplates()
    for name in ("linking.txt", "generation.txt"):
        tmpl = templates.load(name)
        assert "SYSTEM" not in tmpl.system  # marker already stripped
        assert tmpl.system and tmpl.user


def test_render_substitutes_placeholders():
    tmpl = PromptTemplates().load("generation.txt")
    system, user = tmpl.render(
        linked_schema="X",
        request="Y",
        dialect="postgres",
        entities="a",
        filters="b",
        columns="c",
        joins="d",
        rationale="e",
    )
    assert "postgres" in system
    assert "Y" in user


def test_missing_placeholder_raises():
    tmpl = PromptTemplates(PROMPT_DIR).load("generation.txt")
    with pytest.raises(ConfigError):
        tmpl.render(request="only-one-arg")  # missing dialect/linked_schema/...


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        PromptTemplates(PROMPT_DIR).load("does_not_exist.txt")


def test_missing_packaged_template_raises():
    with pytest.raises(ConfigError):
        PromptTemplates().load("nope_not_packaged.txt")
