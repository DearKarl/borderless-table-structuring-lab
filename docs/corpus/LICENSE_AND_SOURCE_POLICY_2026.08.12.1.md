# License and Source Policy 2026.08.12.1

Status: `FROZEN_FOR_GENERATOR_SMOKE`

## Scope

This policy governs every content item, font, texture, rendering asset, source
table, prior parser, and external dataset referenced by
`data-smoke-2026.08.12.1` and later corpus builds derived from its generator.

## Default source policy

The smoke uses newly authored synthetic content. It does not ingest evaluation
pages, scraped documents, customer records, or third-party table annotations.
Names, values, formulas, and prose are generated from project-authored grammar
and deterministic seed tables.

## Allowed source decisions

Every source receives one decision:

- `APPROVED_REDISTRIBUTABLE`: redistribution and derivative use are compatible
  with the intended research release.
- `APPROVED_REBUILD_ONLY`: local processing is allowed, but payloads may not be
  redistributed; only compilers, acquisition instructions, and hashes may be
  shared.
- `QUARANTINED`: rights, attribution, origin, or downstream-use terms are
  incomplete, incompatible, or disputed.

Private access is not evidence of redistribution permission.

## Required inventory fields

The machine-readable source inventory records:

- source identifier, title, owner, and acquisition URL or authored origin;
- acquisition date and immutable revision where available;
- license name, version, URL, and license-text SHA256;
- redistribution, modification, derivative, research, and commercial-use
  decisions;
- attribution and notice requirements;
- reviewer, decision date, decision, and rationale;
- payload or source-manifest SHA256.

## Font and rendering assets

The initial smoke permits project-authored assets, public-domain assets, and
fonts with explicit redistribution and embedding rights, such as compatible
SIL Open Font License or Apache License releases. Each font file is pinned by
family, source URL, license, revision, and SHA256. System fonts with unclear
redistribution rights may be used only in `APPROVED_REBUILD_ONLY` outputs.

## Benchmark exclusion

Current OmniDocBench and Customer50 images, crops, text, coordinates, HTML,
LaTeX, identifiers, metadata, annotations, embeddings, or transformations are
prohibited. Public benchmark attribute names may inform coverage categories;
they may not provide record content.

## External dataset rule

An external dataset may enter a later dated corpus only through a new source
decision, an isolation audit, and a dated specification revision. Dataset
licenses remain independent of code-repository licensing. A noncommercial
benchmark cannot be silently converted into a redistributable or commercial
training source.

## Hard stops

Generation stops when a required license file, source revision, attribution,
permission decision, or hash is missing. Failed records remain in the source
quarantine manifest with a reason code; they are not replaced silently.
