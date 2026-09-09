# med-rag — Usage Guide

> One-page summary. For setup/operations see [`DEPLOY.md`](DEPLOY.md),
> for configuration see [`CONFIG.md`](CONFIG.md).

## Sign in

Open `http://<server-address>:8507` in a browser and enter your password
(`MEDRAG_SIFRE`). You stay signed in for the life of the session.

## Library

- **Add documents**: drag files onto the upload area (top-left) or click to select.
  Multiple files can be chosen at once. Supported formats: PDF (digital + scanned/photo),
  Word, PowerPoint, Excel, HTML, Markdown.
- **Status badges**: each file is tracked as `queued → processing → ready` (or `error`).
  Scanned PDFs are slow and advance page by page. On error, the reason is shown on the
  badge.
- **Read a document**: click a document in the list — its rendered, readable form opens
  (tables/lists/headings preserved). Use "view original" to open the source PDF.
- **Search**: filter the list by file name.
- **Delete**: trash icon → confirm. Every trace (text, vector, source list) is removed
  immediately; older chat answers can no longer cite that document.
- **Replace**: upload a new version with the same name — the old content drops away and
  the new one is processed.

## Chat

- Type your question; the stages it passes through stream into the right panel as the
  answer is produced.
- The **[1] [2]** badges in answers are evidence: clicking one jumps to the relevant
  document, page and section. A numbered source list sits under the answer.
- If no source is found the system says **"I couldn't find that"** — it never invents.
  Clinical answers carry a short **"consult a clinician"** note underneath.
- If sources conflict, both are shown with a warning box.
- Start a **new chat** from the top right; older chats are restored from history.

## Notes

- The notes tab lets you create/edit text notes (e.g. a patient summary, a reminder).
- A saved note is processed automatically and the chatbot **uses your note as a source** —
  it appears as a citation in answers.
- Deleting a note is the same as deleting a document: its traces are removed.

## FAQ

**My document has been "processing" for a long time?** Scanned/photo PDFs naturally take
minutes even on a normal run. If it hangs too long an error badge drops; re-uploading the
file (same name) is enough to reprocess it.

**The answer cites old content?** If you changed the file, re-upload it under the same
name; the old derivations are deleted automatically.

**I forgot my password?** Update the `MEDRAG_SIFRE` line in the `.env` file on the server
and restart the web service with `docker compose up -d web`.
