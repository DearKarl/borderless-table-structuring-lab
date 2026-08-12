# Contributing

Thank you for contributing to Borderless Table Structuring Lab. The repository
supports collaborative research on table representations, data generation,
explicit layout reasoning, and parameter-efficient generative modeling.

## Shared research method

All model tracks operate on the same Canonical Table representation of logical
topology, physical geometry, cell content, and OCR-token ownership.

- **Explicit Layout Transformer** predicts sparse, order-invariant topology
  changes. Structural proposals remain separable from text and geometry.
- **LoRA Table Model** predicts one complete Canonical Table state through
  parameter-efficient adaptation.
- **Shared data and evaluation** provide identical document roles, schemas,
  OCR evidence, geometry, metrics, and overlap audits for both tracks.

Keep the two model implementations independent until a comparison explicitly
studies their interaction. Shared infrastructure belongs in neutral modules;
track-specific assumptions belong under the corresponding method or model
area.

## Calendar versioning

Project-authored snapshots, models, datasets, schemas, experiment records, and
evaluation artifacts use calendar versioning without a `v` prefix.

| Artifact | Preferred identifier | Same-day revision |
|---|---|---|
| Research snapshot or Git tag | `2026.08.12` | `2026.08.12.1` |
| Explicit model | `explicit-2026.08.12` | `explicit-2026.08.12.1` |
| LoRA model | `lora-2026.08.12` | `lora-2026.08.12.1` |
| Dataset release | `data-2026.08.12` | `data-2026.08.12.1` |
| Schema release | `canonical-table-record-2026.08.12` | `canonical-table-record-2026.08.12.1` |
| Experiment | `experiment-2026.08.12-<topic>` | append `.1` when needed |

Use the most precise meaningful date. The identifier does not claim that every
artifact was created at exactly the same time; its purpose is to distinguish
research states reproducibly. Upstream packages, datasets, and benchmarks keep
their official external version names.

## Branches and pull requests

Use a focused branch from the current default branch:

- `explicit/<topic>` for explicit topology modeling;
- `lora/<topic>` for generative adaptation and ablations;
- `data/<topic>` for corpus construction and validation;
- `eval/<topic>` for route-independent evaluation;
- `docs/<topic>` for research documentation.

A pull request should state:

1. the research question or engineering problem;
2. the hypothesis and changed variable;
3. the affected CalVer release;
4. the data roles, sources, licenses, and isolation implications;
5. the metrics and acceptance criteria;
6. the commands used for validation.

Use an imperative English commit summary and keep each commit scoped to one
logical change, for example `Add deterministic span-template generator`.

## Data contributions

Data-related contributions must include source and license provenance,
deterministic seeds, family-level split assignments, record and payload hashes,
complete generated/pass/quarantine/failure counts, and exact plus near-duplicate
audits. Dataset payloads remain outside Git; commit schemas, generators,
manifests, and small synthetic fixtures only.

Do not use one ordered split/merge action path as the sole target when multiple
paths produce the same table. Prefer the final Canonical Table state and an
order-invariant structural difference.

## Model contributions

Model code must declare its input representation, output contract, model CalVer
identifier, checkpoint provenance, and supported metrics. Add a small
deterministic correctness test before any performance experiment. Explicit and
LoRA results should be reported against the same frozen data roles and
route-independent metrics.

## Required checks

- `pytest` passes.
- All project-authored files, filenames, documentation, and comments are English.
- No dataset payload, model weight, credential, or private evaluation artifact
  is committed.
- Generated outputs use new, non-overwriting paths.
- SHA256 manifests and complete failure accounting are present where required.
- Exact and near-duplicate audits pass before a corpus role is released.

## Restricted content

Do not commit restricted benchmark pages, private evaluation material,
annotations without redistribution rights, model weights, credentials, or
derived near-duplicates. Do not commit source datasets unless redistribution
and downstream-use rights have been explicitly approved.
