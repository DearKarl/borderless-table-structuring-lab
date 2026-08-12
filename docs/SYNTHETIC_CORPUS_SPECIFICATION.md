# Synthetic Corpus Specification

Status: `DRAFT_FOR_STAGE_5_4_REVIEW`

## Primary record contract

Every record contains a source-traceable rendered table, direct Canonical Table
Gold, OCR token sidecar, physical geometry, a Raw-like prior, and an
order-invariant structural difference. Ordered action programs may be retained
as replay evidence but are not primary supervision.

Required identity fields:

- `sample_id`;
- `document_cluster_id`;
- `template_family_id`;
- `content_family_id`;
- `renderer_family_id`;
- `generation_seed`;
- `role`;
- `schema_version`.

Required provenance fields:

- generator and version;
- source and license decision;
- font sources;
- render and degradation parameters;
- record and payload SHA256 values;
- explicit confirmation that terminal inputs were not used.

## Phenomenon coverage matrix

| Family | Required variants | Safety counterexample |
|---|---|---|
| Headers | single, multi-level, irregular | correct Raw structure requiring KEEP |
| Spans | row, column, mixed 2D | visually aligned cells that must not merge |
| Corrections | split-only, merge-only, joint | zero-edit identity |
| Borders | weak, missing, partial, low contrast | visible guides that are not cell borders |
| Empty cells | empty, near-empty, whitespace | meaningful sparse text |
| Text | dense, small, long, multiline, mixed script | Raw OCR text that must remain frozen |
| Formulas | formula-only, mixed formula/text | ordinary symbols that are not formulas |
| Imaging | rotation, blur, compression, scan noise | clean high-resolution render |
| Local errors | one minimal reversible topology error | broad edits prohibited |

## Distribution principles

- KEEP examples must be substantial enough that editing is not the default
  prior.
- Difficulty is balanced by template family rather than post-hoc metric
  selection.
- Correction types are sampled before rendering.
- Degradation parameters are sampled independently within preregistered ranges.
- No distribution parameter is calibrated using terminal scores.

Exact numerical ranges and requested counts must be frozen in a separate
preregistration before the bounded generator smoke.

## Split policy

Assign document, template, content, renderer, and seed families to one role
before rendering. No template family crosses train, development, holdout, or
terminal roles.

## Acceptance tests

- Canonical legality and complete grid coverage.
- Direct Gold recompilation.
- Complete, unique OCR token ownership.
- Finite and non-degenerate physical geometry.
- Deterministic semantic replay.
- Source and license completeness.
- Exact, perceptual, structure, text, source, and geometry overlap audit.
- Complete generated, passed, quarantined, and failed counts.

An unresolved overlap, license ambiguity, terminal-data risk, missing
geometry, token loss, illegal topology, or nondeterministic replay is a hard
stop.
