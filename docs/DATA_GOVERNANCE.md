# Data Governance

## Roles

Every record has exactly one immutable role: `train`, `development`,
`holdout`, or `terminal`. Roles are assigned before rendering and before any
model or metric is run.

## Required provenance

Every generated or newly sourced record must include:

- source or generator identifier and license;
- template-family ID and deterministic seed;
- render engine, font sources, resolution, and degradation parameters;
- source table content and direct Canonical Table Gold;
- OCR tokens, ownership, confidence, and physical geometry;
- Raw-like prior state generated without terminal data;
- order-invariant structural difference from Raw-like prior to Gold;
- difficulty and phenomenon tags;
- document, template, content, and renderer family identifiers;
- SHA256 entries for every payload and manifest record.

## Terminal exclusion

Customer50 and OmniDocBench terminal pages, crops, annotations, strings,
coordinates, HTML, LaTeX, page metadata, identifiers, Gold records, embeddings,
and transformations are prohibited inputs. No record may be selected because
it resembles a specific terminal case.

## License gate

Private access does not override a source license. Each source must have a
machine-readable inventory stating acquisition method, license version,
redistribution permission, derivative-work permission, commercial-use status,
attribution requirements, and review decision.

Unclear or incompatible records are quarantined. If a source may be processed
but not redistributed, collaborators receive the compiler and hashes and must
rebuild from their own legally obtained copy.

## Split isolation

Split before rendering by:

- document cluster;
- template family;
- content family;
- renderer family;
- generation seed.

No template family may cross roles. Audits must cover exact files, perceptual
images, normalized structures, normalized text, source documents, and geometry
signatures. Any unresolved overlap is a hard stop.

## Acceptance accounting

Every build reports requested, generated, accepted, quarantined, and failed
counts with reason codes. Failed records are preserved in a quarantine manifest
instead of being silently deleted.
