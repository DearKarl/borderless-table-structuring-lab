# Synthetic Data Specification 2026.08.12.4

Status: `FROZEN_FOR_GENERATOR_SMOKE`

Dataset identifier: `data-smoke-2026.08.12.4`

Schema identifier: `synthetic-table-record-2026.08.12.4`

Supersedes: `2026.08.12.3`

## Purpose of this revision

The `2026.08.12.3` smoke stopped after 150 generated payloads when an EDIT pair
used the configured two-column boundary. The Gold builder required at least
three columns before creating a span, so `extra_split` again produced an
identity prior. Its non-overwriting output is preserved as failed evidence.

This revision makes the edit-capable Gold contract valid across the complete
configured range of two through fourteen columns. It adds explicit boundary
tests at 2, 3, 5, and 14 columns while retaining the pairing and overlap-audit
corrections introduced in `2026.08.12.3`.

## Inherited research contract

All unchanged requirements of
[`SYNTHETIC_DATA_SPECIFICATION_2026.08.12.3.md`](SYNTHETIC_DATA_SPECIFICATION_2026.08.12.3.md)
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
  [`COVERAGE_MATRIX_2026.08.12.4.csv`](COVERAGE_MATRIX_2026.08.12.4.csv)
- Generation parameters:
  [`generation_parameters_2026.08.12.4.json`](../../configs/generation_parameters_2026.08.12.4.json)
- Record schema:
  [`synthetic_table_record_2026.08.12.4.json`](../../schemas/synthetic_table_record_2026.08.12.4.json)
- Acceptance criteria:
  [`ACCEPTANCE_CRITERIA_2026.08.12.4.md`](ACCEPTANCE_CRITERIA_2026.08.12.4.md)
- License policy:
  [`LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md`](LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md)
- Split policy:
  [`SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md`](SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md)

## Required regression evidence

- The previously failing paired `extra_split` request must produce a legal,
  non-identity EDIT prior at the minimum configured two-column boundary.
- Every same-role counterfactual pair must share the complete Gold-state hash,
  rendered-image hash, and all declared family identities.
- A single matching structure hash across roles must be reported but must not
  be treated as a hard duplicate without corroborating evidence.
- Record hashes and every entry in `SHA256SUMS` must be independently
  recomputed during verification.

## Negative rejection fixtures

The generator test suite must additionally create at least one intentionally
invalid incomplete-grid fixture. It must be rejected with
`GRID_INCOMPLETE`, remain outside the 256 accepted records, and appear in test
evidence rather than the released corpus manifest.

## Escalation boundary

A sealed pass of `data-smoke-2026.08.12.4` authorizes a new non-overwriting
shared-corpus build. It does not authorize Explicit or LoRA model training.
