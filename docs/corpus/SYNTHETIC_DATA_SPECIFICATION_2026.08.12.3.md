# Synthetic Data Specification 2026.08.12.3

Status: `FROZEN_FOR_GENERATOR_SMOKE`

Dataset identifier: `data-smoke-2026.08.12.3`

Schema identifier: `synthetic-table-record-2026.08.12.3`

Supersedes: `2026.08.12.2`

## Purpose of this revision

The `2026.08.12.2` smoke stopped after 128 generated payloads because the first
paired EDIT request produced an identity prior. Its non-overwriting output is
preserved as failed evidence. The defect was in generator pairing, not in the
frozen coverage quota: the paired Gold inherited only the Hard KEEP appearance
and did not always contain the structural affordance required by the EDIT.

This revision requires every counterfactual pair to share one edit-capable Gold
state and one rendered image. The KEEP member uses that state unchanged; the
EDIT member receives a legal non-identity prior. It also clarifies that one
shared weak structural signature is diagnostic evidence, while cross-role
rejection requires an exact payload/table match, a shared provenance family, or
a perceptual near match supported by text or geometry evidence.

## Inherited research contract

All unchanged requirements of
[`SYNTHETIC_DATA_SPECIFICATION_2026.08.12.2.md`](SYNTHETIC_DATA_SPECIFICATION_2026.08.12.2.md)
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
  [`COVERAGE_MATRIX_2026.08.12.3.csv`](COVERAGE_MATRIX_2026.08.12.3.csv)
- Generation parameters:
  [`generation_parameters_2026.08.12.3.json`](../../configs/generation_parameters_2026.08.12.3.json)
- Record schema:
  [`synthetic_table_record_2026.08.12.3.json`](../../schemas/synthetic_table_record_2026.08.12.3.json)
- Acceptance criteria:
  [`ACCEPTANCE_CRITERIA_2026.08.12.3.md`](ACCEPTANCE_CRITERIA_2026.08.12.3.md)
- License policy:
  [`LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md`](LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md)
- Split policy:
  [`SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md`](SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md)

## Required regression evidence

- The previously failing paired `extra_split` request must produce a legal,
  non-identity EDIT prior.
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

A sealed pass of `data-smoke-2026.08.12.3` authorizes a new non-overwriting
shared-corpus build. It does not authorize Explicit or LoRA model training.
