# Split and Isolation Policy 2026.08.12.1

Status: `FROZEN_FOR_GENERATOR_SMOKE`

## Role assignment

Roles are assigned before rendering, corruption, model inference, or metric
calculation. The smoke contains:

| Role | Records per category | Total records | Permitted use |
|---|---:|---:|---|
| train | 44 | 176 | Generator and future model fitting after authorization |
| development | 12 | 48 | Nonterminal validation and bounded selection |
| holdout | 8 | 32 | One-shot bounded evidence after configuration freeze |

The smoke creates no terminal role. A later terminal role requires a separately
sealed seed range and one-shot protocol after all generator parameters are
frozen.

## Family-level grouping

The following identifiers must each map to exactly one role:

- `document_cluster_id`;
- `template_family_id`;
- `content_family_id`;
- `renderer_family_id`;
- `font_family_id` when font appearance is part of a paired design;
- `counterfactual_pair_id`;
- `source_family_id`;
- generation-seed family.

Counterfactual members always remain in the same role. A role-assignment file
is produced before any image exists and is frozen by SHA256.

## Overlap audits

Every build runs these cross-role audits:

1. payload SHA256 and normalized-pixel SHA256 equality;
2. perceptual-image similarity under a preregistered threshold;
3. normalized Canonical Table topology equality;
4. normalized text-content equality and high-overlap shingles;
5. document, source, template, renderer, font, and pair-family equality;
6. quantized geometry-signature equality;
7. composite near-duplicate linkage across two or more weak signals.

Exact overlap must be zero. A near-duplicate candidate is quarantined until an
English review record explains and resolves it. No unresolved cross-role link
is permitted.

## Benchmark and collaborator isolation

Current OmniDocBench and Customer50 artifacts are not audit inputs because they
are prohibited generator sources. A separate restricted environment may later
run one-way contamination checks that emit only aggregate overlap status and
hashes; it may not expose terminal content to generator authors.

Collaborators receive only the roles and payloads necessary for their assigned
work. Anyone optimizing a generator or model does not receive terminal labels.

## Non-overwrite policy

Each build uses a new CalVer path. `data-smoke-2026.08.12.1` may never be
overwritten. Any correction creates `data-smoke-2026.08.12.2` or a later dated
identifier with a new manifest and Evidence Card.
