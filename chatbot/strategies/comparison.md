# Strategy: comparison

**Intent:** `comparison`
**Meaning:** comparing two or more drugs/doses/treatment options.

## Both sides, cited

Retrieve and present BOTH (all) sides of the comparison from the uploaded
documents. Every factual claim about a side MUST carry an inline citation
(in the user's language), exactly as in `medical_fact`:
`(Kaynak: ARVELES_KUB.pdf, s. 2; DOLOREX_KUB.pdf, s. 4)`.

Prefer a side-by-side structure when the chunks support it: name the
attribute (endikasyon, doz, kontrendikasyon, yan etki...) and give each
side's value under it, each with its citation.

## Rules

- Do NOT invent an attribute or value that is not in the retrieved chunks.
- If the documents only cover ONE side, say so plainly and give only what
  is covered -- do not fabricate the missing side.
- If two sources give DIFFERENT values for the same side, present both
  with citations and state that the sources differ (never average them).
- Do not pick a "winner" unless the documents themselves state a clinical
  preference; a comparison is not a recommendation.

## Safety line

End with exactly one short line, in the user's language:
`Klinik kararlarda güncel kaynakları ve hastanın bireysel durumunu
değerlendiriniz.` / `Please verify against current guidelines and the
individual patient context.`

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer Notes

**Flow:** `default_topn`

- The industrial `comparison` flow depended on SQL (dormant); the medical
  comparison falls back to the single vector path. Multi-entity splitting
  (splitting the query into the two drugs and running a separate top_n)
  could later be added as a dedicated flow.
