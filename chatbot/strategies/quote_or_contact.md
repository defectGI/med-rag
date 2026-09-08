# Strategy: quote_or_contact

**Intent:** `quote_or_contact`
**Meaning:** asking for a price quote or to reach sales/contact.

If the user wants a price quote or asks a sales/contact-related question,
use the ACME contact information below. Pick the location(s) RELEVANT to
the context of the question -- don't dump all of them at once:

- If the user didn't specify a location/country (a general "who should I
  talk to" type question): give **ACME Head Office**.
- If they ask about something like "production/factory": **ACME Production
  Facility**.
- If they ask about something related to Germany/Europe: **ACME Germany
  GmbH**.
- If unclear, briefly ask what they're interested in, don't dump the whole
  list.

Only include business hours if the user asks when they can reach someone, or
if it's otherwise relevant.

### ACME Head Office
1 Example Street, Example District, Example City, Country
Tel: +00 000 000 00 00
Email: info@example.com
KEP: contact@example.com

### ACME Production Facility
2 Example Industrial Park, Example City, Country
Tel: +00 000 000 00 00
Email: info@example.com

### ACME Germany GmbH
3 Example Allee, Example City, Country
Tel: +00 000 000 00 00
Email: contact@example.com

### Business hours
Monday - Friday: 07:30 - 16:30 (UTC+3)

This data is static/corporate info that doesn't vary by product. A price
quote (a concrete figure) is NOT here and must not be generated -- don't
make up a number, direct the user to the channels above.

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `no_retrieval` -- this intent needs no product/document data;
it is wired to the same flow as `out_of_scope` so it does not depend on
retrieval; see `config/default.toml [routing.intents]`.

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- This data comes directly from this strategy file rather than from
  retrieval (can be updated without a code change, not hardcoded). Addresses
  are kept in their original local-language form (real postal data, not
  instructional text) -- same as any other factual record in this system.
