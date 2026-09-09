# Strategy: library

**Intent:** `library`
**Meaning:** a question ABOUT the uploaded documents themselves: which
documents are available, where something is written, listing sources.

## Answer about the documents

- "hangi belgeler yüklü" / "ne var kütüphanede": list the documents that
  appear in the retrieved context, by their `doc=` file name -- do not
  invent documents that are not in the context.
- "bu hangi belgede geçiyor" / "nerede yazıyor": give the document name
  plus section/page from the `doc=`/`section=`/`page=` info, exactly as
  `medical_fact` cites.

## Limits

- If the question needs the FULL list of uploaded documents and the
  retrieved context only carries a few, say which ones you can see and
  note that a complete list is shown in the library view -- do not
  fabricate the rest.
- This intent never cites a file-system path or a `doc_id`; use the
  human-readable file name only.

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer Notes

**Flow:** `default_topn`

- The full document list comes from medrag.api.library (GET /api/documents);
  wiring a deterministic "library list" flow into the chat side is a future
  improvement. For now the model answers honestly from the documents that
  appear in the top_n context.
