# Synthetic Data Specification 2026.08.12.2

Status: `FROZEN_FOR_GENERATOR_SMOKE`

Dataset identifier: `data-smoke-2026.08.12.2`

Schema identifier: `synthetic-table-record-2026.08.12.2`

Supersedes: `2026.08.12.1`

## Purpose of this revision

This revision resolves one contradiction discovered during implementation
review. The earlier coverage matrix requested four accepted records whose prior
contained incomplete grid coverage, while the acceptance gate correctly
required complete and legal prior geometry for all 256 accepted records.

Incomplete-grid examples remain useful validator rejection fixtures, but they
must be `QUARANTINE` records outside the accepted 256-record smoke. The four
accepted records are now legal row-or-column assignment errors that preserve
complete grid coverage while requiring a deterministic topology correction.

## Inherited research contract

All unchanged requirements of
[`SYNTHETIC_DATA_SPECIFICATION_2026.08.12.1.md`](SYNTHETIC_DATA_SPECIFICATION_2026.08.12.1.md)
remain binding, including:

- `KEEP`, `EDIT`, and `QUARANTINE` gate semantics;
- direct Canonical Table and order-invariant supervision;
- Explicit topology-only targets with prior text frozen;
- complete LoRA Canonical Table targets;
- a four-to-one false-edit versus missed-edit cost;
- uncertainty resolving to KEEP;
- 64 Exact KEEP, 64 Hard KEEP, 64 Single Minimal Edit, and 64 Complex
  Correction records;
- 176 train, 48 development, and 32 holdout records;
- at least 64 same-role counterfactual pairs;
- no failed-seed replacement;
- source, license, provenance, deterministic replay, and overlap gates;
- no model training or performance claim.

## Active frozen artifacts

- Coverage matrix:
  [`COVERAGE_MATRIX_2026.08.12.2.csv`](COVERAGE_MATRIX_2026.08.12.2.csv)
- Generation parameters:
  [`generation_parameters_2026.08.12.2.json`](../../configs/generation_parameters_2026.08.12.2.json)
- Record schema:
  [`synthetic_table_record_2026.08.12.2.json`](../../schemas/synthetic_table_record_2026.08.12.2.json)
- Acceptance criteria:
  [`ACCEPTANCE_CRITERIA_2026.08.12.2.md`](ACCEPTANCE_CRITERIA_2026.08.12.2.md)
- License policy:
  [`LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md`](LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md)
- Split policy:
  [`SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md`](SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md)

## Negative rejection fixtures

The generator test suite must additionally create at least one intentionally
invalid incomplete-grid fixture. It must be rejected with
`GRID_INCOMPLETE`, remain outside the 256 accepted records, and appear in test
evidence rather than the released corpus manifest.

## Escalation boundary

A sealed pass of `data-smoke-2026.08.12.2` authorizes a new non-overwriting
shared-corpus build. It does not authorize Explicit or LoRA model training.
