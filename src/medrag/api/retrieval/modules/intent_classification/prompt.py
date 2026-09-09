"""Prompt construction for the LLM intent classifier.

The system prompt is built from ``core.IntentLabel`` (single source of truth for
the label set) plus a short definition and a few examples per class. Queries are
typically Turkish and informal; the few-shot examples reflect that.

med-rag is a medical document Q&A assistant: a physician asks her OWN uploaded
library (drug KUB/prospectuses, textbooks, guidelines, notes). The label set
reflects that domain. The key distinction is between a factual question whose
answer is in the documents (`medical_fact`) and a request for a clinical
decision (`clinical_decision`) -- the two are routed to the SAME retrieval but
different answer policies (the decision one never makes the decision for the
physician).

Examples here are deliberately NOT taken from the eval golden set — reusing
golden queries as few-shots would inflate eval scores dishonestly.
"""

from __future__ import annotations

from medrag.api.retrieval.core import IntentLabel

_DEFINITIONS: dict[IntentLabel, str] = {
    IntentLabel.MEDICAL_FACT: (
        "a factual question whose answer is in the uploaded medical documents: "
        "a drug's dose/indication/contraindication/side effect/interaction, a "
        "disease definition, a mechanism, a guideline recommendation"
    ),
    IntentLabel.CLINICAL_DECISION: (
        "asking what to DO for a patient: whether to give a drug, what to "
        "prescribe/recommend, how to manage a case, an explicit or implicit "
        "\"should I / what would you recommend\""
    ),
    IntentLabel.COMPARISON: (
        "comparing two or more drugs/doses/treatment options"
    ),
    IntentLabel.INTERACTION: (
        "whether two or more drugs can be used together / a drug-drug "
        "interaction check"
    ),
    IntentLabel.LIBRARY: (
        "a question ABOUT the uploaded documents themselves: which documents "
        "are available, where something is written, listing sources"
    ),
    IntentLabel.OUT_OF_SCOPE: (
        "anything unrelated to medicine or the uploaded documents (weather, "
        "sports, code, personal chatter)"
    ),
}

# Few-shot examples: (query, label). Kept distinct from the golden set.
_FEWSHOT: list[tuple[str, IntentLabel]] = [
    ("arveles dozu kac mg", IntentLabel.MEDICAL_FACT),
    ("bu ilacin kontrendikasyonu ne", IntentLabel.MEDICAL_FACT),
    ("hipertansiyon tedavisinde ilk secenek nedir", IntentLabel.MEDICAL_FACT),
    ("deksketoprofenin yan etkileri neler", IntentLabel.MEDICAL_FACT),
    ("hastama arveles verebilir miyim basi agriyor", IntentLabel.CLINICAL_DECISION),
    ("bu hastaya ne yazayim", IntentLabel.CLINICAL_DECISION),
    ("arveles mi dolorex mi daha guvenli", IntentLabel.COMPARISON),
    ("ibuprofen ile aspirin arasindaki fark", IntentLabel.COMPARISON),
    ("arveles ile varfarin birlikte kullanilabilir mi", IntentLabel.INTERACTION),
    ("bu iki ilaci ayni anda verebilir miyim", IntentLabel.INTERACTION),
    ("kutuphanemde hangi belgeler var", IntentLabel.LIBRARY),
    ("bu hangi belgede geciyor", IntentLabel.LIBRARY),
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
    the model nothing to disambiguate near-miss pairs (e.g. medical_fact
    vs. clinical_decision), which biases it toward whichever label "sounds
    right" by default.
    """
    defs = "\n".join(f"- {label.value}: {desc}" for label, desc in _DEFINITIONS.items())
    examples = "\n".join(f'  "{q}" -> {label.value}' for q, label in _FEWSHOT)
    return f"Intent labels:\n{defs}\n\nExamples:\n{examples}"


def build_system_prompt() -> str:
    """Assemble the system prompt: task, label definitions, examples, rules."""
    valid = ", ".join(labels_list())
    return (
        "You classify a physician's chatbot query into exactly ONE intent "
        "label.\n"
        "The queries are about the user's OWN uploaded medical library (drug "
        "KUB/prospectuses, textbooks, guidelines, notes). Queries are usually "
        "in Turkish and written informally (lowercase, missing diacritics, "
        "typos, no punctuation). Classify by meaning, not surface form.\n\n"
        f"{intent_catalogue_block()}\n\n"
        "Rules:\n"
        "- Answer with the label string ONLY — no punctuation, no explanation, "
        "no quotes.\n"
        f"- The answer MUST be exactly one of: {valid}\n"
        "- A clinical question (\"should I give X to my patient\") is IN scope "
        "-- classify it as clinical_decision, NOT out_of_scope.\n"
        "- Only genuinely non-medical/non-document topics (weather, sports, "
        "code) are out_of_scope."
    )


__all__ = ["build_system_prompt", "intent_catalogue_block", "labels_list"]
