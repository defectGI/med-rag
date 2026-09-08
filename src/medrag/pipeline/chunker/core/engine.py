"""Packer — the engine that splits the inner document model into leaf chunks.

Flow: `chunk_document(doc, cfg, tokenizer)` groups blocks into sections by
reading order, packs sections into the token budget, renders each chunk's
text via its MD renderer, and returns a `ChunkSet` (leaves only,
`tree_level=0`) with full metadata. Enrichment fields stay empty (RAPTOR
and cross-ref resolution are the next phase).

Design decisions
----------------
* **Sectioning**: content blocks are grouped by their `heading_path`; the
  heading blocks attach to the following content (a heading never stands
  alone as a chunk — the sole exception is a document that ends with a
  heading). `section_strategy = "hard"` forbids a chunk from mixing two
  sections' content; `"merge"` lets consecutive WHOLE sections merge into
  one chunk while they fit the budget (no partial-section overflow);
  the merged chunk's `heading_path` falls back to the common ancestor
  prefix.
* **Flex is asked in ONE place** (`_Paketleyici._esneme`): "does this
  block type fit at this size with flex?" — the engine asks it in two
  spots (the split/flex decision for an overflowing block, and runtime
  diagnostics), both go through the same method. The formula itself
  lives in `ChunkerConfig.effective_limit`; if the formula's inputs ever
  widen (block size, chunk fill...), only that signature changes.
* **Flex serves NOT splitting, not packing**: an already-full chunk is
  NOT given more flex to squeeze in another block (the alternative is
  opening a new chunk, not splitting); only a block that on its own
  exceeds target flexes up to the effective limit. If that's still not
  enough, the block goes to its type-specific splitter (table →
  `table_split`, paragraph/code/list → `fallback_split`); types without a
  splitter (image: always OCR) honestly overflow.
* **Measured == stored**: every packing decision is measured from the
  chunk's FINAL text (breadcrumb + render). Splitters size the part from
  its own text; once the breadcrumb/heading is added the part can slightly
  exceed target — the chunk still goes out with an honest `token_count`
  and the difference shows up in the flex diagnostic.
* **The breadcrumb is the FIRST block's `heading_path`**: when the section
  heading sits in the chunk text as an MD heading (first chunk) the
  breadcrumb shows only ancestors (the IR heading block's path IS the
  ancestors); in continuation chunks the section itself joins the chain —
  the repetition is prevented automatically. The structured `heading_path`
  metadata, however, is the COMMON PATH PREFIX of the content blocks.
* **Split-block metadata** (links/images/page range) cannot be distributed
  per part (IR doesn't carry line/sentence positions); every part carries
  the whole block's metadata — conservative / recall-friendly for back-
  reference.
* **Crumb merging**: chunks below `packing.min_chunk_tokens` (or with an
  empty body) are merged into a neighbor post-packing — standalone crumbs
  fed into summarization invited the model to invent summaries. The
  "hard" strategy's "sections never mix" guarantee is intentionally pierced
  only for these sub-threshold crumbs; split parts are never touched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from medrag.pipeline.chunker.config import ChunkerConfig
from medrag.pipeline.chunker.core.chunk import (
    ChunkNode,
    ChunkSet,
    CrossRef,
    ImageRef,
    _compute_content_sha256,
    common_heading_prefix,
    leaf_node_id,
)
from medrag.pipeline.chunker.core.document import (
    AnyBlock,
    Document,
    Heading,
    Provenance,
)
from medrag.pipeline.chunker.core.fallback_split import (
    BlockPart,
    split_code,
    split_list,
    split_paragraph,
)
from medrag.pipeline.chunker.core.markdown import render_block
from medrag.pipeline.chunker.core.table_split import (
    TablePart,
    compose_table_text,
    split_table,
)
from medrag.pipeline.chunker.tokenization.base import Tokenizer

# In-document cross-reference candidate: links whose target is an anchor.
# Every other target (http, file path, other document) stays raw — out of
# resolution scope.
_IC_LINK_ONEKI = "#"


def _tr_casefold(s: str) -> str:
    """Casefold that respects Turkish İ/I (for boilerplate matching):
    stdlib `str.casefold()` splits Turkish capital `İ` per the general
    Unicode rule into `i` + combining dot above (U+0307) — so
    `"İÇİNDEKİLER".casefold() != "içindekiler".casefold()`. When we map
    the Turkish I/İ pair ourselves (İ→i, I→ı), as long as both sides go
    through the same transformation the comparison stays consistent."""
    return s.replace("İ", "i").replace("I", "ı").casefold()


class _Bolum(NamedTuple):
    """Sectioning unit: leading headings + content sharing their path."""

    basliklar: list[Heading]
    icerik: list[AnyBlock]

    @property
    def bloklar(self) -> list[AnyBlock]:
        return [*self.basliklar, *self.icerik]


class _EsnemeKarari(NamedTuple):
    """Response of `_Paketleyici._esneme`."""

    sigar: bool
    limit: int   # effective limit for that block type


def _bolumler(blocks: list[AnyBlock]) -> list[_Bolum]:
    """Group blocks into section units by reading order (module docstring).

    Headings accumulate in a pending buffer and fall into a unit with the
    first content block; a section with no content still attaches its
    heading to the next section (no loss). A document ending with a
    heading produces a final content-less unit.
    """
    birimler: list[_Bolum] = []
    basliklar: list[Heading] = []
    icerik: list[AnyBlock] = []

    def kapat() -> None:
        nonlocal basliklar, icerik
        if basliklar or icerik:
            birimler.append(_Bolum(basliklar, icerik))
            basliklar, icerik = [], []

    for blok in blocks:
        if blok.kind == "heading":
            if icerik:
                kapat()
            basliklar.append(blok)
        else:
            if icerik and blok.heading_path != icerik[-1].heading_path:
                kapat()  # path changed without a heading in between (rare but happens)
            icerik.append(blok)
    kapat()
    return birimler


def _ic_bloklar(blok: AnyBlock) -> Iterable[AnyBlock]:
    """Block + (if list) the blocks inside its items — for metadata scan."""
    yield blok
    if blok.kind == "list":
        for oge in blok.items:
            yield from oge.blocks


class _Paketleyici:
    def __init__(self, doc: Document, cfg: ChunkerConfig, tokenizer: Tokenizer):
        self.doc = doc
        self.cfg = cfg
        self.tok = tokenizer
        self.target = cfg.limits.target_tokens
        self.nodes: list[ChunkNode] = []
        # source_block_ids always points at TOP-LEVEL block ids (NOT blocks
        # inside list items) — to re-derive a previously-added breadcrumb
        # during crumb merging (_birlesmis).
        self._blok_by_id = {b.id: b for b in doc.blocks}

    # -- flex: the ONE question ---------------------------------------------------

    def _esneme(self, kind: str, token_count: int) -> _EsnemeKarari:
        """The single question: does this block type fit at this size with
        flex? The formula is in `ChunkerConfig.effective_limit`; the engine
        computes flex nowhere else — if the formula's inputs ever widen,
        only that signature changes."""
        limit = self.cfg.effective_limit(kind)
        return _EsnemeKarari(token_count <= limit, limit)

    # -- text generation ----------------------------------------------------------

    def _kirinti(self, yol: list[str]) -> str | None:
        """Heading-chain prepend line (MD-friendly: bold single line)."""
        hi = self.cfg.heading_injection
        if not hi.prepend_to_text or not yol:
            return None
        secilen = yol[-hi.max_levels:] if hi.max_levels > 0 else yol
        return f"**{hi.separator.join(secilen)}**"

    def _blok_metni(self, blok: AnyBlock) -> str:
        # Tables go through compose: inject_facts applies even in unsplit tables.
        if blok.kind == "table":
            return compose_table_text(blok, self.cfg)
        return render_block(blok, self.cfg)

    def _chunk_metni(self, bloklar: list[AnyBlock]) -> str:
        parcalar = [self._kirinti(bloklar[0].heading_path)]
        parcalar += [self._blok_metni(b) for b in bloklar]
        return "\n\n".join(p for p in parcalar if p)

    # -- chunk publishing ----------------------------------------------------------

    def _yayinla(
        self,
        bloklar: list[AnyBlock],
        *,
        metin: str | None = None,
        kaynak_ids: list[str] | None = None,
        split_kind: str | None = None,
        split_index: int | None = None,
        split_total: int | None = None,
        overlap_units: int | None = None,
    ) -> None:
        if metin is None:
            metin = self._chunk_metni(bloklar)
        n = self.tok.count(metin)

        # Flex diagnostic: when target is exceeded, ask the single question.
        flex_applied, flex_amount, flex_reason, token_limit = False, 0, None, self.target
        if n > self.target:
            kind = split_kind or bloklar[-1].kind
            karar = self._esneme(kind, n)
            flex_applied, flex_amount, token_limit = True, n - self.target, karar.limit
            if not karar.sigar:
                flex_reason = (f"unsplittable {kind} atom overflows the "
                               f"ceiling even with flex")
            elif split_kind:
                flex_reason = "split part overflowed target with breadcrumb/heading"
            else:
                flex_reason = f"{kind} block flexed up to avoid being split"

        icerik = [b for b in bloklar if b.kind != "heading"]
        yol_kaynagi = icerik or bloklar
        tum_ic = [ib for b in bloklar for ib in _ic_bloklar(b)]
        sayfa_baslar = [b.page_start for b in bloklar if b.page_start is not None]
        sayfa_biter = [b.page_end for b in bloklar if b.page_end is not None]

        self.nodes.append(ChunkNode(
            node_id=leaf_node_id(self.doc.doc_id, len(self.nodes)),
            doc_id=self.doc.doc_id,
            tree_level=0,
            text=metin,
            source_path=self.doc.source_path,
            fmt=self.doc.fmt,
            source_block_ids=kaynak_ids or [b.id for b in bloklar],
            heading_path=common_heading_prefix(
                [list(b.heading_path) for b in yol_kaynagi]),
            page_start=min(sayfa_baslar) if sayfa_baslar else None,
            page_end=max(sayfa_biter) if sayfa_biter else None,
            token_count=n,
            token_limit=token_limit,
            flex_applied=flex_applied,
            flex_amount=flex_amount,
            flex_reason=flex_reason,
            split_kind=split_kind,
            split_index=split_index,
            split_total=split_total,
            overlap_units=overlap_units,
            # description included (v6): render_image injects all three
            # texts into the body, all three must also live in the
            # structured channel — otherwise the consumer can't tell a
            # VLM sentence apart from document text.
            images=[ImageRef(image_id=ib.image_id, ocr_text=ib.ocr_text,
                             alt_text=ib.alt_text, description=ib.description,
                             visual_type=ib.visual_type)
                    for ib in tum_ic if ib.kind == "image"],
            cross_refs=[CrossRef(source_block_id=ib.id, link=link.target)
                        for ib in tum_ic for link in ib.links
                        if link.target.startswith(_IC_LINK_ONEKI)],
            provenance_summary=Provenance.lowest(ib.provenance for ib in tum_ic),
        ))

    # -- overflowing block ---------------------------------------------------------

    def _tasani_isle(self, blok: AnyBlock, basliklar: list[Heading]) -> None:
        """A block (with its attached headings) that doesn't fit target on
        its own: first flex, then type-specific splitter, then honest
        overflow if there's no splitter."""
        bloklar = [*basliklar, blok]
        n = self.tok.count(self._chunk_metni(bloklar))
        if self._esneme(blok.kind, n).sigar:
            self._yayinla(bloklar)  # flex diagnostic is computed at publish
            return

        if blok.kind == "table":
            parcalar: list[TablePart] | list[BlockPart] = \
                split_table(blok, self.cfg, self.tok)
        elif blok.kind == "paragraph":
            parcalar = split_paragraph(blok, self.cfg, self.tok)
        elif blok.kind == "code":
            parcalar = split_code(blok, self.cfg, self.tok)
        elif blok.kind == "list":
            parcalar = split_list(blok, self.cfg, self.tok)
        else:
            self._yayinla(bloklar)  # image etc.: no splitter, honest overflow
            return

        for parca in parcalar:
            onek = basliklar if parca.split_index == 1 else []
            ilk_blok = onek[0] if onek else blok
            metin_parcalari = [self._kirinti(ilk_blok.heading_path)]
            metin_parcalari += [render_block(h) for h in onek]
            metin_parcalari.append(parca.text)
            self._yayinla(
                [*onek, blok],
                metin="\n\n".join(p for p in metin_parcalari if p),
                kaynak_ids=[*(h.id for h in onek), blok.id],
                split_kind=blok.kind,
                split_index=parca.split_index,
                split_total=parca.split_total,
                overlap_units=parca.overlap_units,
            )

    # -- section packing ---------------------------------------------------------

    def _bolum_paketle(self, bolum: _Bolum) -> None:
        """Fill a section's blocks into target (full path under hard mode;
        also for a single section that doesn't fit alone under merge)."""
        if not bolum.icerik:
            self._yayinla(bolum.bloklar)  # document ended with a heading (degenerate)
            return

        bekleyen: list[AnyBlock] = list(bolum.basliklar)
        for blok in bolum.icerik:
            aday = [*bekleyen, blok]
            if self.tok.count(self._chunk_metni(aday)) <= self.target:
                bekleyen = aday
                continue
            # Adding this block overflows. If the chunk already has content,
            # close it and start a new chunk with the block; flex is NOT
            # asked here (the alternative is opening a new chunk, not splitting).
            if any(b.kind != "heading" for b in bekleyen):
                self._yayinla(bekleyen)
                if self.tok.count(self._chunk_metni([blok])) <= self.target:
                    bekleyen = [blok]
                    continue
                bekleyen = []
                self._tasani_isle(blok, [])
            else:
                # Pending is empty or only headings: block overflows alone
                # with its attached headings.
                self._tasani_isle(blok, [b for b in bekleyen])
                bekleyen = []
        if bekleyen:
            self._yayinla(bekleyen)

    def _boilerplate_mi(self, birim: _Bolum) -> bool:
        """Does this section's OWN heading match `[boilerplate] drop_headings` —
        sections like Contents / Revision History carry no answer to any
        question and must not enter any chunk."""
        if not self.cfg.boilerplate.drop_headings:
            return False
        engelli = {_tr_casefold(h) for h in self.cfg.boilerplate.drop_headings}
        return any(_tr_casefold(h.text.strip()) in engelli for h in birim.basliklar)

    def _paketle(self) -> None:
        birimler = [b for b in _bolumler(self.doc.blocks)
                   if not self._boilerplate_mi(b)]
        if self.cfg.packing.section_strategy == "hard":
            for birim in birimler:
                self._bolum_paketle(birim)
            return

        # merge: consecutive WHOLE sections merge into one chunk while they fit.
        tampon: list[AnyBlock] = []
        for birim in birimler:
            bloklar = birim.bloklar
            if tampon:
                aday = [*tampon, *bloklar]
                if self.tok.count(self._chunk_metni(aday)) <= self.target:
                    tampon = aday
                    continue
                self._yayinla(tampon)
                tampon = []
            if self.tok.count(self._chunk_metni(bloklar)) <= self.target:
                tampon = bloklar
            else:
                self._bolum_paketle(birim)  # section that doesn't fit alone
        if tampon:
            self._yayinla(tampon)

    # -- crumb merging ------------------------------------------------------------

    def _kirinti_mi(self, n: ChunkNode) -> bool:
        return (not n.text.strip()
                or (n.token_count or 0) < self.cfg.packing.min_chunk_tokens)

    def _bastaki_breadcrumb_soyulmus(self, n: ChunkNode) -> str:
        """Strip n's leading breadcrumb (if any): when merging, A's context
        is already continuing, so B's own breadcrumb would repeat. The
        breadcrumb is reconstructed byte-exactly via `_kirinti` from
        `n`'s FIRST source block's `heading_path` and looked for at the
        head of the text — no heuristic, no false-positive risk."""
        if not n.source_block_ids:
            return n.text
        ilk_blok = self._blok_by_id.get(n.source_block_ids[0])
        if ilk_blok is None:
            return n.text
        breadcrumb = self._kirinti(ilk_blok.heading_path)
        if breadcrumb and n.text.startswith(breadcrumb + "\n\n"):
            return n.text[len(breadcrumb) + 2:]
        return n.text

    def _birlesebilir_mi(self, a: ChunkNode, b: ChunkNode) -> bool:
        """Split parts are never touched (their split metadata would be
        corrupted); the merged text must fit the hard ceiling (in practice
        always fits because the crumb is small)."""
        if a.split_kind is not None or b.split_kind is not None:
            return False
        metin = "\n\n".join(
            t for t in (a.text, self._bastaki_breadcrumb_soyulmus(b)) if t)
        return self.tok.count(metin) <= self.cfg.limits.hard_max_tokens

    def _birlesmis(self, a: ChunkNode, b: ChunkNode) -> ChunkNode:
        """`a` (before) + `b` (after) into one chunk. Texts are concatenated
        in their rendered form (not re-rendered from blocks); b's leading
        breadcrumb is stripped because A's context already carries it.
        Metadata is merged, `heading_path` falls to the common ancestor
        prefix. Ids are renumbered at the end."""
        metin = "\n\n".join(
            t for t in (a.text, self._bastaki_breadcrumb_soyulmus(b)) if t)
        n = self.tok.count(metin)
        baslar = [p for p in (a.page_start, b.page_start) if p is not None]
        biter = [p for p in (a.page_end, b.page_end) if p is not None]
        tasti = n > self.target
        return a.model_copy(update={
            "text": metin,
            "content_sha256": _compute_content_sha256(metin),
            "token_count": n,
            "token_limit": max(a.token_limit or self.target,
                               b.token_limit or self.target),
            "flex_applied": tasti,
            "flex_amount": n - self.target if tasti else 0,
            "flex_reason": ("merged crumb chunk with its neighbor exceeded target"
                            if tasti else None),
            "source_block_ids": [*a.source_block_ids, *b.source_block_ids],
            "heading_path": common_heading_prefix(
                [list(a.heading_path), list(b.heading_path)]),
            "page_start": min(baslar) if baslar else None,
            "page_end": max(biter) if biter else None,
            "images": [*a.images, *b.images],
            "cross_refs": [*a.cross_refs, *b.cross_refs],
            "provenance_summary": Provenance.lowest(
                [a.provenance_summary, b.provenance_summary]),
        })

    def _kirintilari_birlestir(self) -> None:
        """Merge sub-`min_chunk_tokens` chunks (or empty-bodied chunks —
        e.g. image-only with meaningless OCR) into a neighbor: chunks fed
        alone into summarization invited the model to invent. First try
        to grow forward toward the next chunk (a cover crumb belongs to
        the following content), else backward; if no neighbor fits, the
        crumb stays as-is."""
        if self.cfg.packing.min_chunk_tokens == 0 or len(self.nodes) < 2:
            return
        sonuc: list[ChunkNode] = []
        i = 0
        while i < len(self.nodes):
            cur = self.nodes[i]
            while (self._kirinti_mi(cur) and i + 1 < len(self.nodes)
                   and self._birlesebilir_mi(cur, self.nodes[i + 1])):
                cur = self._birlesmis(cur, self.nodes[i + 1])
                i += 1
            if (self._kirinti_mi(cur) and sonuc
                    and self._birlesebilir_mi(sonuc[-1], cur)):
                sonuc[-1] = self._birlesmis(sonuc[-1], cur)
            else:
                sonuc.append(cur)
            i += 1
        self.nodes = [
            n.model_copy(update={"node_id": leaf_node_id(self.doc.doc_id, j)})
            for j, n in enumerate(sonuc)]

    def calistir(self) -> ChunkSet:
        self._paketle()
        self._kirintilari_birlestir()
        for onceki, sonraki in zip(self.nodes, self.nodes[1:]):
            onceki.next_node_id = sonraki.node_id
            sonraki.prev_node_id = onceki.node_id
        return ChunkSet(doc_id=self.doc.doc_id, nodes=self.nodes)


def chunk_document(
    doc: Document, cfg: ChunkerConfig, tokenizer: Tokenizer,
) -> ChunkSet:
    """Split a document into leaf chunks; returns a `ChunkSet` with
    enrichment fields empty."""
    return _Paketleyici(doc, cfg, tokenizer).calistir()