# Evidence Card: Shared Safety Layer V1

Status: `PASS_SEALED_FOR_STAGE3`

Authority: `POST_OMNIDOCBENCH_EXECUTION_CONTRACT_V4.md`

## Hypothesis

A shared fail-closed validator can preserve Raw MinerU exactly when a table-only candidate is missing, invalid, or lacks positive preregistered expected-gain evidence, while accepting only Canonical-valid candidates and freezing all non-table page state.

## Scope

This is correctness evidence only. It does not measure model quality and does not authorize training. The fixture set is the first 32 records in the frozen order of the compliant 169-sample nonterminal development manifest. Fixture selection read neither development Gold nor metrics. Customer50 and OmniDocBench contents were not visible.

## Frozen interfaces

- Canonical interval, declared-shape, unique-cell, non-overlap, and complete-grid validation.
- Physical geometry presence, finiteness, and non-degeneracy validation.
- OCR-token ownership coverage and uniqueness protection when token ownership is present.
- Frozen-Raw or OCR-grounded text protection selected by an explicit policy.
- Required sample, producer, producer-version, purpose, image-hash, and terminal-visibility provenance.
- Caller-supplied nonterminal expected-gain policy; this module does not estimate or calibrate a threshold.
- Exact Raw rollback and identity `PASS_THROUGH`.
- Table-only page assembly with non-table blocks hash-frozen and full-page candidates rejected.

## Preregistered fixture result

| Check | Result |
|---|---:|
| Compliant nonterminal fixtures | 32/32 |
| Raw baseline validator PASS | 32/32 |
| Identity pass-through | 32/32 |
| Identity output SHA equals Raw SHA | 32/32 |
| Deterministic corruptions | 160 |
| Corruptions rolled back | 160/160 |
| Rollback output SHA equals Raw SHA | 160/160 |
| Terminal benchmark visibility | false |
| Gold or metric visibility during fixture selection | false |

The five corruption classes were grid overlap, grid hole, geometry removal, text mutation, and terminal-visibility provenance violation.

## Unit and integration tests

Twenty-two tests passed across the new safety-layer tests and the existing atomic executor rollback tests. They cover identity pass-through, overlap, token loss, text hallucination, geometry, provenance, missing gain, insufficient gain, valid positive-gain acceptance, table-only assembly, full-page candidate rejection, and pre-existing executor rollback behavior.

## Preserved failed attempt

`fixture_run_v1` is retained unchanged. Its behavioral checks all passed, but aggregate status was falsely reported as FAIL because two required non-visibility facts were represented as false-valued check entries before applying `all(checks.values())`. `FAILED_FIXTURE_RUN_V1.json` records the cause. The corrected non-overwrite run is `fixture_run_v2`.

## Hard-gate decision

All Stage 2 hard gates pass. Stage 3 candidate-interface implementation may begin. No smoke, bounded performance experiment, or training result is claimed here.

## SHA256 record

- Active V4 contract: `9a9b8e1fb0ce7729004e897715f2dba695af7cced14bb81203822ae3f9f3a343`
- Canonical Data Layer vNext Evidence Card seal: `1d390e59875a28cc0b9b4e4b58ca64d14e4b97ce26a3f1d98a3f420790fe59bf`
- Frozen compliant fixture manifest: `c8ce22b7cbc3964ddc1de895842110db7d59123274772b8458678f43539ff103`
- Shared safety implementation: `6cd41d5a01d5942f6ce7b82f06c17d0b4302b02836242117ab295698d71d2e90`
- Fixture runner: `94a50e323be94c168fb11fdfebe7d45540d7a2adf3c921490d7d6a8d6e42b433`
- Safety tests: `47867a5fcfe3c36c5f90ac76c76920299c0e96c056fad0dc2ab88f156a1d233c`
- Stage 2 contract: `d641fd3db15b64b99c2d65c02324db24c5e0f401ca2413b2d39810211ccf0957`
- Stage 2 preregistration: `1079efd030141e1d5a2d8bf6e1a3784a30f8865fc89c96bae76dc4de5dcd4c2f`
- Corrected fixture report: `5948ba25645cb669356ec3349828178dc27e900e47ffc8f5ca32b273f9481683`
- Corrected fixture SHA manifest: `82396c3d54e0d66ab9a3de0988e445342e13952dbcb61b23883ebc6f3c58a635`
