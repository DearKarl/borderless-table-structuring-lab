# Shared Corpus Specification 2026.08.12

Status: `FROZEN_FOR_NON_OVERWRITING_BUILD`

Dataset identifier: `shared-corpus-2026.08.12`

Generator identifier: `2026.08.12.5`

Schema identifier: `synthetic-table-record-2026.08.12.5`

## Research purpose

This release builds a shared, source-traceable corpus for independent Explicit
Layout Transformer and LoRA Table Model research. It emphasizes conservative
KEEP behavior, direct final Canonical Table supervision, and minimal legal
structural correction without using terminal benchmark content.

## Frozen scale and roles

| Category | Train | Development | Holdout | Total |
|---|---:|---:|---:|---:|
| Exact KEEP | 7,000 | 2,000 | 1,000 | 10,000 |
| Hard KEEP | 7,000 | 2,000 | 1,000 | 10,000 |
| Single Minimal Edit | 7,000 | 2,000 | 1,000 | 10,000 |
| Complex Correction | 7,000 | 2,000 | 1,000 | 10,000 |
| **Total** | **28,000** | **8,000** | **4,000** | **40,000** |

Hard KEEP and Single Minimal Edit form 10,000 same-role counterfactual pairs.
Each pair shares one rendered observation, one complete Gold state, and all
declared family identities. The KEEP member uses Gold unchanged. The EDIT
member uses one legal, non-identity Raw-like prior.

## Supervision views

- Gate: KEEP or EDIT, with false editing weighted four times a missed edit.
- Explicit: Raw-like prior, Raw text frozen, and an order-invariant structural
  difference to Gold.
- LoRA: one complete table-only Canonical Table target.

An arbitrary ordered merge/split program is not a primary target.

## Source and license boundary

All accepted content is project-authored synthetic content rendered with
rebuild-only runtime assets recorded in the source-license manifest. No
OmniDocBench or Customer50 page, crop, image, string, coordinate, annotation,
identifier, embedding, or transformed derivative may enter the build.

## Isolation

Role assignment occurs before rendering. Document, source, template, content,
renderer, font, and seed families are role-scoped. The build must report exact
payload/table matches, family overlaps, perceptual near matches with supporting
text or geometry evidence, and normalized structure/text/geometry signature
statistics. Any cross-role exact or unresolved near overlap is a hard stop.

## Execution

The generator is CPU-first and streams records, images, checksums, and compact
audit signatures. It must never retain more than two full records in memory.
The output path is new and non-overwriting. Failed seeds are not replaced.

## Acceptance

- Requested, generated, and accepted: 40,000.
- Quarantined and failed: 0.
- Exact KEEP / Hard KEEP / Single / Complex: 10,000 each.
- Train / development / holdout: 28,000 / 8,000 / 4,000.
- Complete same-role counterfactual pairs: 10,000.
- Schema, Canonical legality, complete grid, token ownership, geometry,
  deterministic semantic replay, and deterministic normalized-pixel replay:
  40,000/40,000.
- KEEP identity: 20,000/20,000.
- EDIT legal non-identity replay: 20,000/20,000.
- Cross-role overlap audit: PASS.
- Terminal inputs used: false.
- Complete SHA256 manifest and independent verification: PASS.

## Boundary

A sealed corpus authorizes bounded, preregistered candidate pilots only. It
does not authorize full model training.
