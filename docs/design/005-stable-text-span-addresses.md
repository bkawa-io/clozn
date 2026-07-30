# ADR 005 — Stable text-span addresses

Status: accepted  
Contract: `clozn.text-span-addresses.v1`

## Decision

Prompt, source, answer, and future claim views use one derived address format. An address is anchored
to one immutable run and one native artifact location. It does not replace the context receipt,
influence map, or future claim artifact, and it is not written back into old runs.

A resolved address contains:

- a deterministic run-scoped `address_id`;
- a text-free `relation_key` derived from the native logical slot for parent/child matching;
- the native artifact schema, collection, and identifier;
- a half-open `[start, end)` range measured in Unicode code points;
- an SHA-256 of the exact canonical basis string;
- an SHA-256 of the selected substring.

Canonical text means the exact Python/JSON string as recorded. Hashing encodes that string as UTF-8
without trimming, Unicode normalization, line-ending conversion, or template reconstruction. Byte
offsets from another artifact are not converted into code-point offsets.

The supported address kinds are:

- delivered message;
- rendered prompt segment;
- attached source span;
- answer span;
- claim.

The generic constructor can address a caller-supplied claim span, but this decision adds no claim
extractor or verifier.

## Privacy and unavailable evidence

`full` projection includes only the selected substring, never the entire canonical basis solely for
addressing. `metadata_only` includes hashes and offsets but no text. The schema uses a separate closed
metadata canonical shape so adding `text` to a metadata-only resolution is invalid, and the projection
helper has model-free no-secret tests.

A metadata-only influence export already contains full SHA-256 hashes and code-point offsets, so it
can produce a resolved metadata address. A redacted context receipt may retain only a 16-hex legacy
hash and a UTF-8 byte count. Those fields cannot prove a full SHA-256 or a code-point range, so the
projection keeps a typed redacted reference and does not invent a canonical span.

Literal redaction or artifact drift can leave stored text inconsistent with an older receipt hash.
That produces `drifted` with an explicit reason and the newly observed hashes; the historical native
hash remains attached to the native reference. Drifted resolutions never disclose the disputed
literal, even when the requested projection privacy is `full`.

## Lineage

Parent/child mapping first matches the text-free `relation_key`. A span is `inherited` only when its
kind, offsets, canonical-basis hash, and selected-span hash all match. Changed offsets, basis hashes,
or span hashes are distinct drift reasons. Missing, ambiguous, redacted, and otherwise unresolved
matches are unavailable rather than assumed unchanged.

The relation key is a mapping aid, not evidence that two independently generated spans have the same
meaning. Mapping never turns a structural match or difference into a causal explanation.

## Existing artifact projection

The pure projection helpers cover:

- current context-receipt delivered segments and whole rendered prompt;
- pre-schema run messages without migrating them;
- every influence prompt source, including sources omitted from measurement selection;
- measured influence prompt subspans;
- scored-answer spans.

Influence status, availability, and method are copied unchanged into `source_artifacts`. Numeric
measurements remain in the influence artifact. This contract creates no universal influence score and
does not collapse artifact-native evidence states.

## Read-only investigation surface

`GET /runs/<id>/span-addresses` returns the metadata-only projection. The investigation document links
to it through its additive `text_span_addresses` section; the full address array is not duplicated
inside investigation.

The run store resolves a persisted influence-map blob and verifies its digest before the projection
runs. The projection distinguishes:

- `not_recorded`: no influence artifact was attached;
- the artifact's own native status and method when it loaded;
- `unavailable`: a referenced blob is missing, unreadable, or corrupt;
- `failed`: a persisted object has no supported schema or cannot validate against its native influence
  schema.

Context addresses remain usable when influence evidence is unavailable. Neither the route nor the
investigation builder calls a worker, starts an influence job, or writes a repaired artifact back to
the run.
