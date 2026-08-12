# Evidence Card: Synthetic Data Specification 2026.08.12.2

Status: `PASS_SEALED_FOR_BOUNDED_GENERATOR_IMPLEMENTATION`

## Objective

Resolve the implementation-review contradiction between an accepted
incomplete-grid prior and the mandatory complete-grid acceptance rule, while
preserving the frozen 256-record composition and all source, license, split,
determinism, and non-training boundaries.

## Change

- Replaced four accepted `localized_grid_coverage_error` records with four
  accepted `row_or_column_assignment_error` records.
- Required both Gold and prior to be legal Canonical Tables with complete grid
  coverage for all 256 records.
- Moved incomplete-grid corruption to an explicit negative rejection fixture
  outside the accepted corpus manifest.
- Advanced the specification, configuration, schema, dataset smoke, and public
  research snapshot to `2026.08.12.2`.

## Unchanged boundary

- Requested/generated/accepted records: `256/256/256`.
- Category counts: `64/64/64/64`.
- Role counts: `176/48/32` train/development/holdout.
- Current OmniDocBench or Customer50 contents used: `false`.
- Model training started: `false`.
- Current official benchmark evaluation run: `false`.
- Historical tags and sealed artifacts modified: `false`.

## Validation

- All JSON and JSON Schema documents parsed successfully.
- Coverage and role totals closed exactly.
- All local Markdown links resolved.
- The data-free regression suite passed `18/18` tests under Python 3.12.
- Repository diff passed `git diff --check`.

## Frozen artifact hashes

| Artifact | SHA256 |
|---|---|
| `configs/generation_parameters_2026.08.12.2.json` | `4d5730db1e814256bd0c3287e3a8f36bc6f968038da01fcb0ef7300ebd29b625` |
| `docs/corpus/SYNTHETIC_DATA_SPECIFICATION_2026.08.12.2.md` | `0c66f64d6e9be7fbb8e554e57500b176b7486a93445ad6f7d259b01668818cb7` |
| `docs/corpus/COVERAGE_MATRIX_2026.08.12.2.csv` | `dd35f920d5614e609213dc9bce22bf0dd0549d2c8b78a64b7788aa22fd9ed8d2` |
| `docs/corpus/ACCEPTANCE_CRITERIA_2026.08.12.2.md` | `c232d2a638005ef13ecb72c1fbe697c7b31e37be64413981398e3a56ecc7c147` |
| `schemas/synthetic_table_record_2026.08.12.2.json` | `ccab367184358f75cf653235034eb33de0688618b2e9923b0c58871fae059f0e` |

## Result

The `2026.08.12.2` specification revision passes and is the only active input
to the bounded generator smoke. The immutable `2026.08.12.1` tag remains
preserved as historical research evidence.
