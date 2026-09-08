"""describe/ tests: the shared describe pass and its per-type strategies.

The point of the package is that a table and a block diagram are the same kind
of thing -- a describable block -- so these tests mostly assert that ONE pass
handles both, honors the same config, and differs only in strategy. No network:
VLM/LLM are fakes.
"""

from __future__ import annotations

import json

import pytest

from medrag.pipeline.parser.describe.core import describe_blocks
from medrag.pipeline.parser.llm import LLMError
from medrag.pipeline.parser.parsers.base import (
    HeadingBlock,
    ImageBlock,
    ParagraphBlock,
    ParsedDocument,
    TableBlock,
    TableData,
    text_cell,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-pixels"


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Keep every cache/blob write inside tmp_path -- a test that forgets this
    writes into the repo's real storage/ tree."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("PROGRESS_BAR", "0")
    # The shipped default enables visual_check; pin it OFF here so the baseline
    # tests exercise the no-check path deterministically. The verify+retry tests
    # opt back in with their own setenv (which wins, running after this).
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK", "0")
    return tmp_path


def _blob(tmp_path, sha: str = "crop1", data: bytes = PNG) -> str:
    root = tmp_path / "images"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{sha}.png").write_bytes(data)
    return sha


class FakeVLM:
    def __init__(self, reply: str | None = "A block diagram of the AAF unit.",
                 raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0
        self.systems: list[str] = []
        self.users: list[str] = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls += 1
        self.systems.append(system)
        self.users.append(user)
        if self.raises:
            raise LLMError("boom")
        return self.reply


class FakeLLM:
    def __init__(self, reply: str = "A spec table.") -> None:
        self.reply = reply
        self.calls = 0
        self.users: list[str] = []

    def complete(self, *, system, user, max_tokens=1024):
        self.calls += 1
        self.users.append(user)
        return self.reply


def _image(tmp_path, *, visual_type="block_diagram", **over) -> ImageBlock:
    kwargs = {"id": "b0", "image_index": 1, "visual_type": visual_type,
              "image_id": _blob(tmp_path), "mime": "image/png"}
    kwargs.update(over)
    return ImageBlock(**kwargs)


def _table(rows=(("pin", "volt"), ("p1", "33"))) -> TableBlock:
    return TableBlock(id="t0", table=TableData(
        n_rows=len(rows), n_cols=len(rows[0]),
        cells=[[text_cell(v) for v in r] for r in rows]))


def _doc(*blocks) -> ParsedDocument:
    doc = ParsedDocument(doc_id="d", source_path="x.docx", fmt="docx")
    doc.blocks = list(blocks)
    return doc


# --- the whole point: non-table visuals finally get described -----------------


def test_block_diagram_gets_a_description(_isolate_storage):
    """Before the describe pass a block_diagram carried a type label and
    nothing else -- a product's architecture diagram contributed zero text."""
    doc = _doc(_image(_isolate_storage))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description == "A block diagram of the AAF unit."
    assert doc.blocks[0].description_source == "vlm-vision"
    assert vlm.calls == 1


@pytest.mark.parametrize("vtype", ["chart", "block_diagram",
                                   "technical_drawing", "flowchart"])
def test_every_describable_visual_type_is_described(_isolate_storage, vtype):
    doc = _doc(_image(_isolate_storage, visual_type=vtype))
    describe_blocks(doc, stage="vlm", vlm=FakeVLM())
    assert doc.blocks[0].description is not None


@pytest.mark.parametrize("vtype", ["decorative", "product_photo", "unknown"])
def test_types_outside_the_describe_list_are_left_alone(_isolate_storage, vtype):
    """decorative has nothing to describe, product_photo's OCR usually
    suffices, unknown is uncertain -- none is in `[describe] types` by default."""
    doc = _doc(_image(_isolate_storage, visual_type=vtype))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description is None
    assert vlm.calls == 0


def test_describe_types_is_config_not_code(_isolate_storage, tmp_path, monkeypatch):
    """"unknown" (e.g. an oddly-drawn chart the classifier couldn't place) can
    be brought in by config alone -- no code change."""
    ov = tmp_path / "ov.toml"
    ov.write_text('[describe]\ntypes = ["unknown"]\n', encoding="utf-8")
    monkeypatch.setenv("PARSER_CONFIG", str(ov))
    doc = _doc(_image(_isolate_storage, visual_type="unknown"))

    describe_blocks(doc, stage="vlm", vlm=FakeVLM())

    assert doc.blocks[0].description is not None


def test_describe_enabled_false_stops_every_type(_isolate_storage, monkeypatch):
    """`DESCRIBE_ENABLED=0` stops description generation unconditionally -- not
    even a type listed in `types` gets described, and the VLM is NEVER called
    (cost decision: visual descriptions do not enter the chunk body anyway)."""
    monkeypatch.setenv("DESCRIBE_ENABLED", "0")
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram"))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description is None
    assert vlm.calls == 0


def test_describe_enabled_false_does_not_mark_excluded(_isolate_storage, monkeypatch):
    """A global shut-off must NOT mark `excluded_at_parse` -- that flag means
    "this type is deliberately left out" and draws a placeholder for the
    consumer. Temporarily switching it off for cost must not change what the
    block means, otherwise re-enabling it would leave a wrong 'left out' trace
    in the IR."""
    monkeypatch.setenv("DESCRIBE_ENABLED", "0")
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram"))

    describe_blocks(doc, stage="vlm", vlm=FakeVLM())

    assert doc.blocks[0].excluded_at_parse is not True


# --- context: the reason this is a separate pass ------------------------------


def test_context_reaches_a_visual_exactly_like_a_table(_isolate_storage, monkeypatch):
    """A classification call sees one isolated crop. This pass runs over the
    finished document, so a diagram gets the same surrounding text a table
    gets -- that is why it is a pass and not part of classification."""
    monkeypatch.setenv("DESCRIBE_CONTEXT", "1")
    doc = _doc(HeadingBlock(id="h0", text="AAF Unit", level=1),
               ParagraphBlock(id="p0", text="The block diagram below shows it."),
               _image(_isolate_storage),
               ParagraphBlock(id="p1", text="Signals flow left to right."))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    (user,) = vlm.users
    assert "Section: AAF Unit" in user
    assert "The block diagram below shows it." in user
    assert "Signals flow left to right." in user


def test_context_off_by_default(_isolate_storage):
    doc = _doc(ParagraphBlock(id="p0", text="Nearby text."),
               _image(_isolate_storage))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert "Nearby text." not in vlm.users[0]


def test_prompt_is_type_aware(_isolate_storage):
    doc = _doc(_image(_isolate_storage, visual_type="flowchart"))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert "flowchart" in vlm.systems[0]
    assert "decision points" in vlm.systems[0]


# --- confidence gate: a weak label is not asserted as fact --------------------


def test_low_confidence_type_falls_back_to_unknown_prompt(_isolate_storage, monkeypatch):
    """A classification below the threshold must not be handed to the describer
    as fact (asserting a wrong type is what makes the VLM confabulate). The
    prompt drops to the humble "unknown" strategy instead."""
    monkeypatch.setenv("DESCRIBE_MIN_VISUAL_CONFIDENCE", "0.5")
    doc = _doc(_image(_isolate_storage, visual_type="technical_drawing",
                      visual_type_confidence=0.2))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    system = vlm.systems[0]
    assert "technical drawing" not in system        # not asserted
    assert "genuinely cannot tell" in system         # unknown guidance ran
    # The IR still records the real classified type + confidence untouched.
    assert doc.blocks[0].visual_type == "technical_drawing"
    assert doc.blocks[0].visual_type_confidence == 0.2


def test_high_confidence_type_is_kept(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_MIN_VISUAL_CONFIDENCE", "0.5")
    doc = _doc(_image(_isolate_storage, visual_type="technical_drawing",
                      visual_type_confidence=0.9))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert "technical drawing" in vlm.systems[0]


def test_threshold_zero_disables_the_gate(_isolate_storage, monkeypatch):
    """Threshold 0.0 = gate off: even a near-zero confidence changes nothing."""
    monkeypatch.setenv("DESCRIBE_MIN_VISUAL_CONFIDENCE", "0")
    doc = _doc(_image(_isolate_storage, visual_type="technical_drawing",
                      visual_type_confidence=0.01))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert "technical drawing" in vlm.systems[0]


# --- size gate: an unreadably small crop is never described -------------------


def test_tiny_crop_is_not_described(_isolate_storage):
    """A crop below the min edge (default 48px) is left as type + placeholder --
    describing it can only invent content (the 47x40 logo case)."""
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram",
                      width=47, height=40))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description is None
    assert vlm.calls == 0
    assert doc.blocks[0].visual_type == "block_diagram"  # type is kept


def test_large_enough_crop_is_described(_isolate_storage):
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram",
                      width=200, height=120))
    describe_blocks(doc, stage="vlm", vlm=FakeVLM())
    assert doc.blocks[0].description is not None


def test_size_gate_is_configurable(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_MIN_VISUAL_EDGE_PX", "0")  # disabled
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram",
                      width=10, height=10))
    vlm = FakeVLM()
    describe_blocks(doc, stage="vlm", vlm=vlm)
    assert vlm.calls == 1  # nothing gated when threshold is 0


def test_unknown_dimensions_never_gate(_isolate_storage):
    """width/height None (dimensions unknown) must not be treated as small."""
    doc = _doc(_image(_isolate_storage, visual_type="block_diagram"))
    vlm = FakeVLM()
    describe_blocks(doc, stage="vlm", vlm=vlm)
    assert vlm.calls == 1


# --- verify+retry: an unfaithful description is flagged / retried -------------


class CheckingVLM:
    """A fake VLM that answers describe calls and verify calls differently
    (the verify system prompt is the one mentioning "faithful"). `verdicts` is
    the sequence of faithfulness booleans returned by successive verify calls;
    `descriptions` the sequence of descriptions returned by successive describe
    calls (the last one repeats if exhausted)."""

    def __init__(self, verdicts, descriptions=("desc A", "desc B", "desc C")):
        self.verdicts = list(verdicts)
        self.descriptions = list(descriptions)
        self.describe_calls = 0
        self.verify_calls = 0
        self.users: list[str] = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.users.append(user)
        if "faithful" in system:  # a verify call
            faithful = self.verdicts[self.verify_calls]
            self.verify_calls += 1
            reason = "" if faithful else "asserts a connector that is not shown"
            return json.dumps({"faithful": faithful, "reason": reason})
        i = min(self.describe_calls, len(self.descriptions) - 1)
        self.describe_calls += 1
        return self.descriptions[i]


def test_verify_off_leaves_status_none(_isolate_storage):
    doc = _doc(_image(_isolate_storage))
    describe_blocks(doc, stage="vlm", vlm=FakeVLM())
    assert doc.blocks[0].describe_status is None
    assert doc.blocks[0].describe_attempts is None


def test_verify_pass_first_try(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK", "1")
    doc = _doc(_image(_isolate_storage))
    vlm = CheckingVLM(verdicts=[True])

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description == "desc A"
    assert doc.blocks[0].describe_status == "ok"
    assert doc.blocks[0].describe_attempts == 1
    assert vlm.verify_calls == 1


def test_verify_retry_then_pass_feeds_reason_back(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK", "1")
    doc = _doc(_image(_isolate_storage))
    vlm = CheckingVLM(verdicts=[False, True])

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].describe_status == "ok"
    assert doc.blocks[0].describe_attempts == 2
    assert doc.blocks[0].description == "desc B"          # second attempt kept
    # the rejection reason reached the retry prompt
    assert any("was rejected" in u for u in vlm.users)


def test_verify_all_attempts_flagged(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK", "1")
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK_RETRIES", "2")
    doc = _doc(_image(_isolate_storage))
    vlm = CheckingVLM(verdicts=[False, False])

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].describe_status == "flagged"
    assert doc.blocks[0].describe_attempts == 2
    # A flagged description is still stored (best effort), like a flagged table.
    assert doc.blocks[0].description == "desc B"


def test_verify_status_is_cached(_isolate_storage, monkeypatch):
    monkeypatch.setenv("DESCRIBE_VISUAL_CHECK", "1")
    doc = _doc(_image(_isolate_storage))
    describe_blocks(doc, stage="vlm", vlm=CheckingVLM(verdicts=[True]))

    # Second run, fresh block, no VLM at all -> must come from cache incl. status.
    doc2 = _doc(_image(_isolate_storage))
    describe_blocks(doc2, stage="vlm", vlm=None)
    assert doc2.blocks[0].description == "desc A"
    assert doc2.blocks[0].describe_status == "ok"


# --- idempotency / never overwrite a better source ----------------------------


def test_structural_description_is_never_overwritten(_isolate_storage):
    """A native chart's description comes from its own XML at parse time --
    strictly more trustworthy than anything a VLM could add."""
    doc = _doc(_image(_isolate_storage, visual_type="chart",
                      description="Bar chart. Series 'X': A=1.",
                      description_source="structural"))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert doc.blocks[0].description == "Bar chart. Series 'X': A=1."
    assert doc.blocks[0].description_source == "structural"
    assert vlm.calls == 0


def test_rerun_does_not_call_the_model_again(_isolate_storage):
    doc = _doc(_image(_isolate_storage))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)
    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert vlm.calls == 1


def test_identical_crop_in_a_new_document_hits_the_cache(_isolate_storage):
    """A PARSER_VERSION bump re-parses the whole corpus for unrelated fixes;
    an unchanged crop must not be re-sent to the VLM."""
    vlm = FakeVLM()
    describe_blocks(_doc(_image(_isolate_storage)), stage="vlm", vlm=vlm)
    describe_blocks(_doc(_image(_isolate_storage)), stage="vlm", vlm=vlm)

    assert vlm.calls == 1


# --- fail-open ----------------------------------------------------------------


def test_no_vlm_leaves_the_block_intact(_isolate_storage):
    doc = _doc(_image(_isolate_storage))

    describe_blocks(doc, stage="vlm", vlm=None)

    assert doc.blocks[0].description is None
    assert doc.blocks[0].visual_type == "block_diagram"  # type survives


def test_vlm_error_is_not_a_crash(_isolate_storage):
    doc = _doc(_image(_isolate_storage))

    describe_blocks(doc, stage="vlm", vlm=FakeVLM(raises=True))

    assert doc.blocks[0].description is None


def test_empty_reply_is_not_stored(_isolate_storage):
    doc = _doc(_image(_isolate_storage))

    describe_blocks(doc, stage="vlm", vlm=FakeVLM(reply="   "))

    assert doc.blocks[0].description is None
    assert doc.blocks[0].description_source is None


def test_visual_with_no_pixels_is_skipped(_isolate_storage):
    """SmartArt has no crop anywhere -- its description comes from its XML."""
    doc = _doc(ImageBlock(id="b0", image_index=1, visual_type="block_diagram"))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert vlm.calls == 0
    assert doc.blocks[0].description is None


def test_source_crop_is_used_when_no_image_id(_isolate_storage):
    """A pdf region diverted from a fake table has only an audit crop."""
    _blob(_isolate_storage, "auditcrop")
    doc = _doc(ImageBlock(id="b0", image_index=1, visual_type="technical_drawing",
                          source_crop="auditcrop"))

    describe_blocks(doc, stage="vlm", vlm=FakeVLM())

    assert doc.blocks[0].description is not None


# --- exclusion: parse-time exclusion beats describing -------------------------


def test_excluded_at_parse_visual_is_not_described(_isolate_storage):
    doc = _doc(_image(_isolate_storage, excluded_at_parse=True))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert vlm.calls == 0
    assert doc.blocks[0].description is None


def test_excluded_type_visual_is_marked_and_skipped(_isolate_storage, tmp_path,
                                                    monkeypatch):
    ov = tmp_path / "ov.toml"
    ov.write_text('[visual]\nexclude_types = ["block_diagram"]\n', encoding="utf-8")
    monkeypatch.setenv("PARSER_CONFIG", str(ov))
    doc = _doc(_image(_isolate_storage))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert vlm.calls == 0
    assert doc.blocks[0].excluded_at_parse is True


def test_excluding_table_at_parse_time_now_actually_works(tmp_path, monkeypatch):
    """Putting "table" in `[visual] exclude_types` used to do nothing at parse
    time (image_handler only walks ImageBlocks) -- only chunk-time exclusion
    honored it. The shared pass is where a table's only generated content is
    produced, so it is where the exclusion has to bite."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    ov = tmp_path / "ov.toml"
    ov.write_text('[visual]\nexclude_types = ["table"]\n', encoding="utf-8")
    monkeypatch.setenv("PARSER_CONFIG", str(ov))
    doc = _doc(_table())
    llm = FakeLLM()

    describe_blocks(doc, stage="llm", llm=llm)

    assert llm.calls == 0
    assert doc.blocks[0].description is None
    assert doc.blocks[0].excluded_at_parse is True   # emptiness is deliberate
    assert doc.blocks[0].table.cells                  # source data still kept


# --- stage split (model affinity) ---------------------------------------------


def test_vlm_stage_ignores_tables(_isolate_storage):
    doc = _doc(_table(), _image(_isolate_storage))
    vlm = FakeVLM()

    describe_blocks(doc, stage="vlm", vlm=vlm)

    assert vlm.calls == 1                       # only the image
    assert doc.blocks[0].description is None    # table untouched here


def test_llm_stage_ignores_visuals(_isolate_storage):
    doc = _doc(_table(), _image(_isolate_storage))
    llm = FakeLLM()

    describe_blocks(doc, stage="llm", llm=llm)

    assert doc.blocks[0].description == "A spec table."
    assert doc.blocks[1].description is None    # visual untouched here


def test_table_description_records_its_source(_isolate_storage):
    doc = _doc(_table())
    describe_blocks(doc, stage="llm", llm=FakeLLM())
    assert doc.blocks[0].description_source == "llm-cells"


def test_unknown_stage_is_rejected(_isolate_storage):
    with pytest.raises(ValueError, match="stage"):
        describe_blocks(_doc(), stage="both")


def test_no_candidates_needs_no_client(_isolate_storage):
    """A document with nothing describable must not build a client at all --
    otherwise an offline run of a text-only document would fail."""
    doc = _doc(ParagraphBlock(id="p0", text="just text"))
    describe_blocks(doc, stage="llm")   # no llm passed, must not raise
    describe_blocks(doc, stage="vlm")
