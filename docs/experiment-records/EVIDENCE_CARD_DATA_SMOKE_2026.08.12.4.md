# Evidence Card: Synthetic Data Smoke 2026.08.12.4

Status: `PASS_SEALED_FOR_SHARED_CORPUS_BUILD`

## Hypothesis

A deterministic generator can produce the frozen four-category smoke with
complete Canonical Gold and legal Raw-like priors while preserving same-role
KEEP/EDIT counterfactual identity and preventing cross-role leakage.

## Frozen inputs

- Configuration: `configs/generation_parameters_2026.08.12.4.json`
- Coverage matrix: `docs/corpus/COVERAGE_MATRIX_2026.08.12.4.csv`
- Record schema: `schemas/synthetic_table_record_2026.08.12.4.json`
- Generator: `src/borderless_table_structuring/synthetic_data.py`
- Entry point: `scripts/generate_data_smoke_2026.08.12.4.py`

The payload was written outside Git to the non-overwriting local dataset
release `data-smoke-2026.08.12.4`.

## Accounting

| Measure | Result |
|---|---:|
| Requested | 256 |
| Generated | 256 |
| Accepted | 256 |
| Quarantined | 0 |
| Failed | 0 |
| Exact KEEP | 64 |
| Hard KEEP | 64 |
| Single Minimal Edit | 64 |
| Complex Correction | 64 |
| Train / development / holdout | 176 / 48 / 32 |
| Complete same-role counterfactual pairs | 64 |

## Acceptance evidence

- Schema, Canonical legality, complete grids, token ownership, and geometry:
  `256/256` pass.
- KEEP identity and EDIT non-identity replay: pass.
- Two-run semantic and normalized-pixel replay: pass.
- Counterfactual pair Gold and image identity: pass.
- Cross-role exact and unresolved near-overlap audit: `PASS`.
- Incomplete-grid negative fixture: rejected as `GRID_INCOMPLETE`.
- Terminal inputs used: `false`.
- Model training or benchmark evaluation: none.
- Data-free regression suite: `36 passed`.

## Failure history preserved

- `data-smoke-2026.08.12.2` stopped after 128 payloads because a paired
  `extra_split` request produced an identity prior.
- `data-smoke-2026.08.12.3` stopped after 150 payloads because the same edit
  was not supported at the configured two-column boundary.
- Neither failed output was overwritten or used as accepted data.

## Sealed payload hashes

- Root `SHA256SUMS` SHA256:
  `8f9b524e576afdc87d39baefe6d80d95eb0366d4335848db1b221d600db7449c`
- Acceptance report SHA256:
  `36b4e845921ed63f407c969a353748ce9867689666a6ca582c3afea9bc316ac8`
- Overlap report SHA256:
  `5126cb7b53cae7cad4841dddef979f96646c941af42934a6f41fe18bf1ff30cc`
- Failed `2026.08.12.2` report SHA256:
  `b70d67a04f7cb9d52334292f2bb941f825b86516cd8ff9c0ecae0dee908fe77f`
- Failed `2026.08.12.3` report SHA256:
  `a79bba6b898b590fb6b96ef53c0d44d2e8f2a7a2d53029c158e9daa5203c8801`

## Decision

The bounded data-smoke gate passes and authorizes specification of a new
non-overwriting shared-corpus build. This card does not authorize model
training and does not support a model-performance claim.
