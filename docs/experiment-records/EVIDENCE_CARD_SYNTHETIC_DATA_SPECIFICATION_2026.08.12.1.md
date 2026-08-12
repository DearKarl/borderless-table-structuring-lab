# Evidence Card: Synthetic Data Specification 2026.08.12.1

Status: `PASS_SEALED_FOR_BOUNDED_GENERATOR_IMPLEMENTATION`

## Objective

Freeze an English, CalVer-addressed specification for a shared synthetic table
corpus before any batch generation or model training. The specification must
make KEEP supervision, deterministic structural correction, provenance,
license decisions, split isolation, and hard acceptance gates executable.

## Authorization and boundary

- Work type: documentation, schema, configuration, and preregistration.
- Output release: `2026.08.12.1`.
- Next bounded dataset: `data-smoke-2026.08.12.1`.
- Data generated in this stage: `0` records.
- Model training started: `false`.
- Current OmniDocBench or Customer50 contents used: `false`.
- Current official benchmark evaluation run: `false`.
- Historical sealed artifacts modified: `false`.

## Frozen decisions

1. Gate labels are `KEEP`, `EDIT`, and `QUARANTINE`.
2. Semantic equivalence is KEEP even when serialization differs.
3. Explicit supervision is an order-invariant topology difference with prior
   text frozen; LoRA supervision is a complete Canonical Table state.
4. The 256-record smoke contains 64 Exact KEEP, 64 Hard KEEP, 64 Single
   Minimal Edit, and 64 Complex Correction records.
5. The smoke split is 176 train, 48 development, and 32 holdout records; it has
   no terminal role.
6. At least 64 same-role counterfactual pairs are required.
7. False edits cost four times missed edits, and uncertainty resolves to KEEP.
8. Failed seeds are not silently replaced.
9. All acceptance checks must pass before bulk generation.
10. Neither a data-smoke pass nor a corpus build authorizes model training.

## Research basis

- PubTables-1M motivates canonical ground truth and removal of inconsistent
  over-segmentation.
- GriTS motivates separate topology, location, and content views.
- OmniDocBench official attributes motivate generic coverage categories without
  supplying page-level training material.
- Synthetic table-recognition research motivates explicit control of structural
  complexity and spanning-cell distributions.

## Validation performed

- All JSON files parsed successfully.
- Coverage matrix totals closed at `256/176/48/32` for
  requested/train/development/holdout.
- Each of the four primary categories totaled 64 records.
- All local Markdown links resolved.
- Repository diff passed `git diff --check`.
- The data-free regression suite passed `18/18` tests under Python 3.12.
- Public-tree naming audit found no project-owned legacy dataset-name,
  baseline-number, or internal contract-version identifiers.
- Upstream dependency and benchmark version names were preserved.

## Frozen artifact hashes

| Artifact | SHA256 |
|---|---|
| `configs/generation_parameters_2026.08.12.1.json` | `700c94b47a0f1e3e0e42c60e2eec6c33ff19c438f6afee047ffe49e80cf6d02d` |
| `docs/corpus/SYNTHETIC_DATA_SPECIFICATION_2026.08.12.1.md` | `430a6758f22043e647bb42d1bebaa23f99702b08ccf1e92ca022e92e00054bb4` |
| `docs/corpus/COVERAGE_MATRIX_2026.08.12.1.csv` | `833c750ddd72ec27b218ca96e3309f15d2b59f6d389e9af88d76df28e20d0795` |
| `docs/corpus/LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md` | `962840f33fdc4187ef9c1a8106b1d1efd7a00f8b12f5091db03ae2f0bf5e1dae` |
| `docs/corpus/SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md` | `1af384360ea2726558dfb9625eebbe647e4170b7b2abb4324b55f45bda763b97` |
| `docs/corpus/ACCEPTANCE_CRITERIA_2026.08.12.1.md` | `2a57ecc701ae1c77d9dcd8a51b70c8b4b8121cedb212c2e58d4631316cccbfae` |
| `schemas/synthetic_table_record_2026.08.12.1.json` | `b8ae27175fe6147a8d14c380f92cbb82d13cffa4673735aaba79182c1e17fcf3` |
| `schemas/source_license_record_2026.08.12.1.json` | `5f4194eab163e36b2fc1090e808d9baf48085317cca1a6a9c868266e96316203` |

## Result

The specification stage passes and authorizes implementation of the bounded
256-record generator smoke in a new non-overwriting path. Bulk corpus
generation remains conditional on a sealed smoke pass. Model training remains
unauthorized.
