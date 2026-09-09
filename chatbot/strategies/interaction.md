# Strategy: interaction

**Intent:** `interaction`
**Meaning:** whether two or more drugs can be used together / a drug-drug
interaction check.

## Retrieve the interaction evidence

Look for the drugs' interaction sections in the retrieved chunks (drugs
with their "etkileşim" / "birlikte kullanım" / "kontrendikasyon" text).
Present what the documents actually say, each claim with an inline
citation (in the user's language), exactly as in `medical_fact`.

## If the documents do not state an interaction

This is common and important: the ABSENCE of an interaction note is not
proof of safety. If the retrieved chunks carry no explicit interaction
information for the combination in question, say so plainly:

- "yüklediğiniz belgeler bu iki ilacın birlikte kullanımına dair açık bir
  bilgi içermiyor" / "the uploaded documents do not state an interaction
  between these drugs",
- and that this is NOT the same as "no interaction".

Do NOT invent an interaction, and do NOT declare the combination safe.

## Safety line

End with exactly one short line, in the user's language:
`Klinik kararlarda güncel kaynakları ve hastanın bireysel durumunu
değerlendiriniz.` / `Please verify against current guidelines and the
individual patient context.`

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer Notes

**Flow:** `default_topn`

- Two-entity retrieval: a single query to top_n; targeted retrieval of the
  interaction sections can later be improved with a dedicated flow.
- Critical rule: "no information" ≠ "no interaction" -- the strategy makes the
  model say this explicitly.
