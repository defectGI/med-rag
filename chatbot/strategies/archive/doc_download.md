# Strategy: doc_download

**Intent:** `doc_download`
**Meaning:** wanting to obtain a FILE (datasheet, manual, brochure,
catalogue, CE declaration of conformity, technical drawing, STP/CAD, product
image, quick-start guide).

The evidence you receive lists files that REALLY exist for this product/these
products -- each line shaped like `file_name (type) -- product: MODEL`.
Don't hesitate to tell the user the **real file name** (`file_name`) -- this
field is safe to show. But NEVER write/make up the real file PATH on disk
(folder/path) -- that information never even reaches you, only the file name
+ type do. The actual download happens through a separate channel in the
user interface (a download link/button) -- you just describe naturally which
file was found (e.g. "I've prepared the datasheet file for PN1309
(ACME_PN5106_Datasheet.pdf), you can download it below.").

**Ambiguity -- evidence is completely EMPTY:** this usually means it's
unclear which PRODUCT the file was requested for (e.g. "can you send the
datasheet" -- which product?). Ask which ACME product code (e.g. PN1309)
they meant. Don't make up/guess a code.

**Ambiguity -- multiple file types are listed:** Among the evidence you may
see a row with `id="doc_download_types"` -- this tells you multiple DIFFERENT
file types exist for the product (brochure, datasheet, CE declaration, ...).
In this case, ask the user to pick ONLY among the types listed in that row --
don't INVENT/suggest a type not in the list, and don't PICK one file as
"this is it" before it's clarified.

**File not found (evidence empty AND a product code was already given):**
Say that no file (of the requested type) was found for that product. NEVER
try to present a nonexistent file as if it exists.

**Follow-up question:** Below you may receive a follow-up note (if any) --
about other REALLY existing files for the same product. Only use it if it
fits the flow of the conversation, as a natural closing sentence (e.g. "I can
also send you this product's brochure if you'd like"); NEVER suggest a file
not in that list.

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `doc_download` -- unlike `sql_topn`, this NEVER goes to LLM
SQL generation; it's a deterministic query against the `document` table;
see `chatbot/flows/doc_download.py`.

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely source: the `document` table (`facts/db/schema.yaml`) --
  `is_active=1` rows, filtered by `model` + (if present) `doc_type`.
- `chatbot/flows/doc_download.py::extract_product_codes`/`extract_doc_type`
  extract the product code (regex, "DE" + digits) and file type (a small
  Turkish/English surface-form dictionary) from the query without an LLM; if
  neither is found or the type is ambiguous, the flow falls into one of the
  clarification flows above without ever going to retrieval, and the session
  is pinned to this flow for the next turn (SAME pinned-flow pattern as
  `ComparisonFlow`/`RecommendationFlow`) -- the user's answer goes
  directly to this flow, bypassing reconcile/classify.
- **`source_path` (the real file location on disk) NEVER reaches this
  strategy/the main model** -- `DocumentRow`/`_to_result` (flows/doc_download.py)
  never carries this field (a type-level guarantee, not dependent on
  discipline/docstrings). The actual download happens via the user
  interface's `GET /api/documents/<doc_id>` (webapp.py) endpoint --
  `chatbot/chatbot/document_store.py` is the ONLY place that reads
  `source_path`.
- The evidence may also include rows marked `requested=False` -- these are
  OTHER real files of the same product that the user did NOT request THIS
  turn; don't force these into the main answer, only use them to the extent
  the "Follow-up question" section above allows.
