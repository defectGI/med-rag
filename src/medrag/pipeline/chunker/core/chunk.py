"""Chunk output schema — the code form of the root README "Chunk schema" section.

Single node type (`ChunkNode`); `tree_level` separates leaves (real chunks, 0)
from upper-level synthetic nodes (>=1). `core/` only produces leaves and
leaves enrichment fields (`summary`, `keywords`, `parent_id`,
`cross_refs[].target_chunk_id`) empty. RAPTOR (which populated those +
added synthetic summary nodes) was removed entirely — `tree_level>=1`
is no longer filled by any producer; the field stays in the schema for
backward compatibility.

Leaf-only fields must be empty/default at summary nodes — this is a
contract, not a style choice; the `ChunkNode` validator rejects violations
by name ("fail at the boundary" principle, same as first_parse).
Exception (schema v2): `heading_path`/`page_start`/`page_end` MAY be
populated at summary nodes — they carry the aggregation over the summary's
children (common heading prefix, min/max page). Previously all three were
left empty, so summary nodes couldn't trace back to any source/page.

`CHUNK_SCHEMA_VERSION` is bumped on any contract-breaking change; readers
never silently accept a newer file (same principle as the parser repo).
This versions "how the file is read"; "which code produced the content"
is a separate axis, written to `ChunkProvenance` (v4) — chunker version,
source IR identifier, RAPTOR model/prompt version. The CLI's staleness
gate works off that.

Node ids are SET-SCOPED: `{doc_id}::c{n}` numbering restarts at zero every
time a set is produced; cross-set stability is NOT GUARANTEED (same
principle for IR block ids). What IS persistent is `doc_id` (the registry
preserves it across rescans); a vector-DB writer therefore deletes ALL
old records for that `doc_id` and rewrites on refresh — there's no
"update only changed" game.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medrag.pipeline.chunker.core.document import Provenance

# Schema version. Bump on any contract-breaking change; readers migrate.
# v2: at summary nodes, heading_path/page_start/page_end may now be
# populated (see module docstring, _LEAF_ONLY_DEFAULTS).
# v3: `scope` + `member_doc_ids` added to ChunkSet — corpus- and
# profile-level trees live in separate, purely synthetic files; the `_`
# doc_id prefix is reserved for scope identifiers.
# v4: `provenance` added to ChunkSet — which chunker version, which IR
# (raw_sha256 + parser_version + ir_version), which RAPTOR mode/model,
# when. The CLI's staleness gate works off this
# (`ChunkProvenance.same_inputs`).
# v5: `content_sha256` added to ChunkNode — the node's own hash of its
# `text` (`sha256:<hex>` format), as the node's OWN declaration. Previously
# this hash was only computed on the fly inside `facts/` pass 1
# (atom generation); since both sides used the same formula it was
# never actually cross-checked. The formula is fixed in ONE place
# (`_compute_content_sha256`): `sha256(text.encode("utf-8"))`, no
# normalize/trim. `facts/` does NOT import this module (architectural
# rule) — it implements the same formula independently and a
# cross-check test verifies consistency.
# v6: `description` added to `ImageRef` — the three image-derived text
# fields (ocr_text/alt_text/description) are now complete in the
# structured channel. `render_image` was already injecting all three
# into the chunk body, but `description` was never stored separately,
# so once mixed into the body it couldn't be untangled; the consumer
# couldn't answer "is this sentence the document's or the VLM's?".
# Older (<=v5) files remain readable: the
# field is optional, missing reads as None. But in v5 a missing
# `description` does NOT mean "no image text" — the text was in the
# body and can't be untangled; cropping code should NOT trust pre-v6
# sets (see ChunkSet.schema_version).
CHUNK_SCHEMA_VERSION = 6


def _compute_content_sha256(text: str) -> str:
    """`text`'s `sha256:<hex>` content hash — the single formula fixed at
    the protocol level. No normalize/trim: raw
    `text` string, UTF-8 encode, sha256. `facts/` pass-1's `chunk_sha256`
    computation MUST match this EXACTLY (independent implementation +
    cross-check test, common import is forbidden)."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

# Document-above scope identifiers. The `_` prefix is forbidden for real
# doc_ids (the CLI rejects at entry) — scope files live alongside
# per-doc files in the same directory, so an id collision would mean a
# scope silently impersonating a leaf.
CORPUS_SCOPE_ID = "_corpus"


def profile_scope_id(profile_id: str) -> str:
    """Profile scope id: `_profile.{profile_id}`."""
    return f"_profile.{profile_id}"


def leaf_node_id(doc_id: str, n: int) -> str:
    """Leaf node id: `{doc_id}::c{n}` (n = 0-based chunk index)."""
    return f"{doc_id}::c{n}"


def summary_node_id(doc_id: str, level: int, n: int) -> str:
    """Summary node id: `{doc_id}::s{level}.{n}` (level >= 1).

    `doc_id` may also be a scope id (`_corpus`, `_profile.{id}`) —
    synthetic nodes in the corpus/profile tree use the same pattern:
    `_corpus::s1.0` etc.
    """
    return f"{doc_id}::s{level}.{n}"


def common_heading_prefix(yollar: list[list[str]]) -> list[str]:
    """Common ancestor prefix of multiple `heading_path`s.

    Single shared location: `core/engine.py` (merging leaf/section
    chunks) used the same logic — kept here in one place.
    """
    if not yollar:
        return []
    onek = list(yollar[0])
    for yol in yollar[1:]:
        i = 0
        while i < len(onek) and i < len(yol) and onek[i] == yol[i]:
            i += 1
        del onek[i:]
    return onek


class ImageRef(BaseModel):
    """The chunk's image reference — every text DERIVED from the image is
    ALSO carried in structured form alongside any inline embedding (so the
    answer can re-show the relevant image AND the consumer can tell that
    text apart from document text).

    visual_type
        Taxonomy label (visual classification plan). An image that fell
        to a placeholder in the chunk text (excluded/contentless) is
        still visible here with its type — the structured reference is
        the "don't hide existence" guarantee independent of the body.
    description
        The image's natural-language description (VLM output, IR's
        `Image.description`). `render_image` injects this into the
        chunk body alongside `ocr_text`/`alt_text`; before v6 it had no
        structured home, so once mixed into the body it couldn't be
        untangled. The consumer (e.g. `facts/` pass 1) can only answer
        "did the document itself write this sentence or did a VLM?" via
        exact match against this field.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str | None = None
    ocr_text: str | None = None
    alt_text: str | None = None
    description: str | None = None
    visual_type: str | None = None


class SourceProvenance(BaseModel):
    """Identity of the IR that fed this chunk set.

    `raw_sha256` is `sha256:<hex>` format. All three are
    copied from the IR; None = the IR didn't carry that field (older
    parser output)."""

    model_config = ConfigDict(extra="forbid")

    raw_sha256: str | None = None
    parser_version: str | None = None
    ir_version: int | None = None


class RaptorProvenance(BaseModel):
    """RAPTOR enrichment record: which mode, which model/prompt.

    `mode="off"` runs leave ChunkSet.provenance.raptor as None; `fake`/
    `real` runs populate `model` (Summarizer's diagnostic name, e.g.
    "ollama:qwen2.5:32b", "fake") and `prompt_version` (the summary
    prompt's version — bumps when the prompt changes, so the staleness
    gate regenerates old-summary sets)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fake", "real"]
    model: str | None = None
    prompt_version: str | None = None


class ChunkProvenance(BaseModel):
    """A chunk file's production identity. The question "which bytes, which
    code, when" must be answerable by opening the file — no registry, no
    folder counting."""

    model_config = ConfigDict(extra="forbid")

    chunker_version: str
    # Offset local ISO-8601, e.g. 2026-07-17T14:03:22+03:00.
    generated_at: str
    source: SourceProvenance | None = None  # None in scope (corpus/profile) sets
    raptor: RaptorProvenance | None = None  # RAPTOR was removed: always None

    def same_inputs(self, other: ChunkProvenance) -> bool:
        """Staleness comparison: is everything except `generated_at` equal?
        If yes, the file is current (same IR + same chunker + same RAPTOR
        setup); regeneration can be skipped."""
        return (self.model_dump(exclude={"generated_at"})
                == other.model_dump(exclude={"generated_at"}))


class CrossRef(BaseModel):
    """In-document cross-reference record.

    `core/` writes the candidate with `target_chunk_id=None` (the link's
    target is in-document but the target chunk isn't known yet); after the
    full chunk set is produced, `enrichment/` resolves the target. Links
    going out of document never enter this field; they stay raw in the
    block text.

    Scope boundary: resolution happens ONLY through the source's own
    anchor id (`Heading.anchor_id`) — in practice md/html. PDF has no
    anchors; "see Table 3" style numbered references are DELIBERATELY
    left unresolved: a wrong binding via guessed numbers would deliver
    the wrong table to the answer — worse than no link. That reference
    text stays raw in the chunk body, where retrieval can see it.
    """

    model_config = ConfigDict(extra="forbid")

    source_block_id: str
    link: str
    target_chunk_id: str | None = None


# Leaf-only fields that MUST stay at default/empty at summary nodes
# (tree_level >= 1) and their defaults. The validator walks this map;
# add a new leaf field here too. `heading_path`/`page_start`/`page_end`
# are DELIBERATELY ABSENT here (v2) — summary nodes may carry the
# aggregation.
_LEAF_ONLY_DEFAULTS: dict[str, Any] = {
    "source_path": None,
    "fmt": None,
    "source_block_ids": [],
    "token_count": None,
    "token_limit": None,
    "flex_applied": False,
    "flex_amount": 0,
    "flex_reason": None,
    "split_kind": None,
    "split_index": None,
    "split_total": None,
    "overlap_units": None,
    "images": [],
    "cross_refs": [],
    "prev_node_id": None,
    "next_node_id": None,
    "provenance_summary": None,
}


class ChunkNode(BaseModel):
    """A single node in the chunk tree.

    Common fields are meaningful at every node; "Leaf-only" fields must
    stay at default at summary nodes (enforced by validator).
    """

    model_config = ConfigDict(extra="forbid")

    # -- Common (every node) --------------------------------------------------
    node_id: str
    doc_id: str
    tree_level: int = Field(ge=0)  # 0 = leaf, >=1 = summary node
    # Real chunk content at a leaf; abstractive summary at a summary node.
    text: str
    # `text`'s content hash (v5) — meaningful at EVERY node (leaf + summary),
    # not leaf-only. If None, `_kontrat` fills it; if provided, it MUST
    # match the recomputed hash from `text` (mismatch = ValueError, no
    # silent drift).
    content_sha256: str | None = None
    # Filled by RAPTOR (every node including leaves); None/empty in core output.
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    parent_id: str | None = None  # None = root (or enrichment hasn't run yet)
    child_ids: list[str] = Field(default_factory=list)

    # -- Leaf-only (tree_level == 0) ------------------------------------------
    source_path: str | None = None  # citation pass-through
    fmt: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    # Size diagnostics: `token_limit` is the effective limit for that chunk;
    # `flex_*` records what the flex formula did to that chunk.
    token_count: int | None = Field(default=None, ge=0)
    token_limit: int | None = Field(default=None, ge=1)
    flex_applied: bool = False
    flex_amount: int = Field(default=0, ge=0)
    flex_reason: str | None = None
    # Split record: if `split_kind` is set, this chunk is a part of a block
    # that didn't fit the hard ceiling. The unit of `overlap_units` depends
    # on `split_kind`: table=rows, paragraph=sentences, list=top-level items,
    # code=lines. `split_index`/`split_total` are 1-based.
    split_kind: Literal["table", "paragraph", "list", "code"] | None = None
    split_index: int | None = Field(default=None, ge=1)
    split_total: int | None = Field(default=None, ge=1)
    overlap_units: int | None = Field(default=None, ge=0)
    images: list[ImageRef] = Field(default_factory=list)
    cross_refs: list[CrossRef] = Field(default_factory=list)
    # Neighboring leaves in document reading order (None at the ends).
    prev_node_id: str | None = None
    next_node_id: str | None = None
    # Lowest confidence among contained blocks
    # (computed via Provenance.lowest; None if no block reported it).
    provenance_summary: Provenance | None = None

    @property
    def is_leaf(self) -> bool:
        return self.tree_level == 0

    @model_validator(mode="after")
    def _kontrat(self) -> ChunkNode:
        beklenen_hash = _compute_content_sha256(self.text)
        if self.content_sha256 is None:
            self.content_sha256 = beklenen_hash
        elif self.content_sha256 != beklenen_hash:
            raise ValueError(
                f"content_sha256 inconsistent: given={self.content_sha256!r} "
                f"expected from text={beklenen_hash!r} (node_id={self.node_id!r})")
        if self.tree_level > 0:
            for alan, default in _LEAF_ONLY_DEFAULTS.items():
                if getattr(self, alan) != default:
                    raise ValueError(
                        f"{alan!r} may only be populated at a leaf "
                        f"(tree_level=0); this node is tree_level={self.tree_level}")
        if self.split_kind is not None:
            if self.split_index is None or self.split_total is None:
                raise ValueError(
                    "split_index and split_total are required when split_kind is set")
            if self.split_index > self.split_total:
                raise ValueError(
                    f"split_index ({self.split_index}) > split_total "
                    f"({self.split_total}) is not allowed")
        elif (self.split_index is not None or self.split_total is not None
              or self.overlap_units is not None):
            raise ValueError(
                "split_index/split_total/overlap_units are only meaningful "
                "when split_kind is set")
        if self.flex_amount > 0 and not self.flex_applied:
            raise ValueError("flex_amount > 0 but flex_applied=False")
        if (self.page_start is not None and self.page_end is not None
                and self.page_start > self.page_end):
            raise ValueError(
                f"page_start ({self.page_start}) > page_end ({self.page_end}) is not allowed")
        return self


class ChunkSet(BaseModel):
    """One scope's chunk output: the entire tree in a single file.

    `scope="document"` (default; v2 files also read as this): one
    document's tree — `core/` produces only leaves; `enrichment/` adds
    summary nodes and rewrites the same structure.

    `scope="corpus"|"profile"` (v3): a DOCUMENT-ABOVE tree. The file
    contains ONLY synthetic nodes; level-1 nodes' `child_ids` point at
    per-doc file leaves (`{doc_id}::c{n}`, globally unique) — leaves
    are NOT copied (copying would mean two conflicting parent_id records
    for the same leaf + silent stale-text risk; reference-only means a
    re-chunked document is caught loudly via `member_doc_ids` +
    unresolvable id). Hence `node_by_id` returns None for a corpus child
    — by design; the RAG consumer already loads per-doc files, the union
    is a dict merge. The `doc_id` field holds the scope id
    (`_corpus` / `_profile.{id}`).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CHUNK_SCHEMA_VERSION
    doc_id: str
    scope: Literal["document", "corpus", "profile"] = "document"
    # When scope != "document": ordered doc_ids of the documents that fed
    # the tree (provenance + desync detection: a member file has been
    # deleted/reproduced is visible).
    member_doc_ids: list[str] = Field(default_factory=list)
    # Production identity (v4). None only in pre-v4 files; the CLI fills
    # it on every write and runs the staleness gate on it (a file without
    # provenance is stale by definition → regenerated).
    provenance: ChunkProvenance | None = None
    nodes: list[ChunkNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _surum_kapisi(self) -> ChunkSet:
        if self.schema_version > CHUNK_SCHEMA_VERSION:
            raise ValueError(
                f"file is schema v{self.schema_version}, this reader supports "
                f"up to v{CHUNK_SCHEMA_VERSION}")
        if self.scope == "document":
            if self.member_doc_ids:
                raise ValueError(
                    "member_doc_ids may only be populated for corpus/profile scope")
        else:
            if not self.member_doc_ids:
                raise ValueError(
                    f"scope={self.scope!r} sets must have non-empty member_doc_ids")
            for n in self.nodes:
                if n.is_leaf:
                    raise ValueError(
                        f"scope={self.scope!r} sets may not contain leaves "
                        f"(reference-only contract): {n.node_id!r} tree_level=0")
                if n.doc_id != self.doc_id:
                    raise ValueError(
                        f"every node in a scope set must carry doc_id={self.doc_id!r}; "
                        f"{n.node_id!r} has doc_id={n.doc_id!r}")
        return self

    def node_by_id(self, node_id: str) -> ChunkNode | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)

    def leaves(self) -> list[ChunkNode]:
        return [n for n in self.nodes if n.is_leaf]

    # -- Serialization ----------------------------------------------------------
    # None fields are not written to JSON (avoids clutter and the
    # "None vs empty" ambiguity); on read, a missing field falls back to
    # its default.

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True),
                          ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ChunkSet:
        return cls.model_validate_json(text)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ChunkSet:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))