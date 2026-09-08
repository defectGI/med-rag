"""Prompt construction for the LLM intent classifier.

The system prompt is built from ``core.IntentLabel`` (single source of truth for
the label set) plus a short definition and a few examples per class. Queries are
typically Turkish and informal; the few-shot examples reflect that.

Examples here are deliberately NOT taken from the eval golden set — reusing
golden queries as few-shots would inflate eval scores dishonestly.
"""

from __future__ import annotations

from medrag.api.retrieval.core import IntentLabel

# One-line definition per intent (what the query is asking for).
#
# "price" is deliberately absent from PRODUCT_FACT/AGGREGATION: the `product`
# table genuinely HAS list_price/price_break_* columns (see
# facts/db/schema_rag.sql), so a query classified into either of these labels
# can reach real SQL retrieval and surface an actual DB price number. That
# contradicts QUOTE_OR_CONTACT's own strategy file (chatbot/
# strategies/quote_or_contact.md: "A price quote (a concrete figure) is NOT
# here and must not be generated -- don't make up a number, direct the user
# to the channels above") -- the intent existed and routed correctly (see
# [routing.intents] in config/default.toml), but listing price here steered
# price questions AWAY from it. Purchasing-adjacent questions (delivery date,
# stock quantity, bulk/volume pricing) are folded into QUOTE_OR_CONTACT below,
# where they belong (all commercial, none are in the catalogue data).
_DEFINITIONS: dict[IntentLabel, str] = {
    IntentLabel.PRODUCT_FACT: (
        "a single technical fact/attribute/code of ONE specific product"
    ),
    IntentLabel.AGGREGATION: (
        "counting, filtering, listing, or an extreme value of ANY technical "
        "attribute (weight, temperature range, power, speed, ...) across MANY "
        "products in the catalogue"
    ),
    IntentLabel.DOC_QUESTION: (
        "a question whose answer is in the document text (how it works, specs "
        "explained, procedures, certifications)"
    ),
    IntentLabel.COMPARISON: "comparing two or more products/options",
    IntentLabel.RECOMMENDATION: (
        "asking for a suggestion for the user's own use case/budget"
    ),
    IntentLabel.VISUAL_REQUEST: (
        "wanting to SEE an image/photo/technical drawing of a product"
    ),
    IntentLabel.DOC_DOWNLOAD: (
        "wanting to obtain a FILE (datasheet, manual, brochure, catalogue, stp)"
    ),
    IntentLabel.QUOTE_OR_CONTACT: (
        "any commercial/purchasing question: a price quote, delivery date/"
        "lead time, stock/inventory quantity, bulk/volume purchase or "
        "quantity discount, or wanting to reach sales/contact"
    ),
    IntentLabel.OUT_OF_SCOPE: (
        "anything unrelated to the product catalogue or documents"
    ),
}

# Few-shot examples: (query, label). Kept distinct from the golden set.
_FEWSHOT: list[tuple[str, IntentLabel]] = [
    ("de2200 agirligi kac kg", IntentLabel.PRODUCT_FACT),
    ("10 amperin uzerinde kac model var", IntentLabel.AGGREGATION),
    ("en genis sicaklik araliginda calisan urun hangisi", IntentLabel.AGGREGATION),
    ("bu modul hangi protokolleri destekliyor", IntentLabel.DOC_QUESTION),
    ("de2200 ile de2300 hangisi hizli", IntentLabel.COMPARISON),
    ("laboratuvar kurulumu icin ne alsam iyi olur", IntentLabel.RECOMMENDATION),
    ("bunun arkadan gorunusu nasil resmi var mi", IntentLabel.VISUAL_REQUEST),
    ("el kitabini pdf yollayin", IntentLabel.DOC_DOWNLOAD),
    ("bayilik icin kiminle gorusmeliyim", IntentLabel.QUOTE_OR_CONTACT),
    ("de2200 fiyati ne kadar", IntentLabel.QUOTE_OR_CONTACT),
    ("bu urunun teslim suresi ne kadar", IntentLabel.QUOTE_OR_CONTACT),
    ("stokta kac adet var", IntentLabel.QUOTE_OR_CONTACT),
    ("100 adet alirsam indirim yapar misiniz", IntentLabel.QUOTE_OR_CONTACT),
    ("mac skoru kac oldu", IntentLabel.OUT_OF_SCOPE),
]


def labels_list() -> list[str]:
    """The valid label strings, in enum order."""
    return [label.value for label in IntentLabel]


def intent_catalogue_block() -> str:
    """Label definitions + few-shot examples, as a standalone block.

    Single source of truth for "what each label means" — any consumer that
    needs an LLM to tell these labels apart (not just this module's own
    classify-only prompt) should reuse this instead of restating the label
    set as a bare name list. A bare list with no definitions/examples gives
    the model nothing to disambiguate near-miss pairs (e.g. recommendation
    vs. aggregation), which biases it toward whichever label "sounds right"
    by default.
    """
    defs = "\n".join(f"- {label.value}: {desc}" for label, desc in _DEFINITIONS.items())
    examples = "\n".join(f'  "{q}" -> {label.value}' for q, label in _FEWSHOT)
    return f"Intent labels:\n{defs}\n\nExamples:\n{examples}"


def build_system_prompt() -> str:
    """Assemble the system prompt: task, label definitions, examples, rules."""
    valid = ", ".join(labels_list())
    return (
        "You classify a user's chatbot query into exactly ONE intent label.\n"
        "The queries are about an industrial product catalogue and its "
        "documents. Queries are usually in Turkish and written informally "
        "(lowercase, missing diacritics, typos, no punctuation). Classify by "
        "meaning, not surface form.\n\n"
        f"{intent_catalogue_block()}\n\n"
        "Rules:\n"
        "- Answer with the label string ONLY — no punctuation, no explanation, "
        "no quotes.\n"
        f"- The answer MUST be exactly one of: {valid}\n"
        "- If the query does not clearly fit any product/document intent, use "
        "out_of_scope."
    )


__all__ = ["build_system_prompt", "intent_catalogue_block", "labels_list"]
