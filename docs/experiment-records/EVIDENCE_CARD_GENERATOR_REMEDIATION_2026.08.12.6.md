# Evidence Card: Generator Remediation 2026.08.12.6

Status: `STAGE_B_PASS_AWAITING_USER_NOTIFICATION_BEFORE_STAGE_C`

## Objective

Remove the scale-sensitive causes of the failed 40,000-record candidate before
materializing a new corpus. The remediated release must construct latent
families without split roles, build duplicate components from realized
signatures, assign complete components to one split, provide collision-resistant
deterministic content and renderer diversity, and block training on every
unsealed corpus.

## Preserved failure

The previous candidate at `data_builds/shared-corpus-2026.08.12` remains
immutable `FAILED_ISOLATION_EVIDENCE`. It was not repaired in place and remains
prohibited for model training.

## Implemented remediation

- Generative identity is derived from realized latent structure, content,
  geometry, renderer, and degradation components rather than role, record ID,
  path, or cosmetic metadata.
- Cell content uses a collision-resistant deterministic stream and no bounded
  modulo cycle.
- Geometry and renderer profiles expose variable dimensions, margins, padding,
  fonts, alignment, wrapping, borders, backgrounds, resolution, rotation, and
  degradation parameters.
- Counterfactual KEEP/EDIT members are constructed as role-free families.
- Duplicate components are built from realized family, structure, geometry,
  text, renderer, and seed signatures before split assignment.
- Complete components are assigned to train, development, or holdout while
  preserving the frozen category and phenomenon quotas.
- A dedicated latent-plan builder and independent verifier were added.
- A training interlock rejects missing, invalid, failed, or manifest-mismatched
  train-ready seals before any training loader may read records.

## Verification

- Full data-free test suite: `54 passed`.
- Family construction remains identical when only requested roles change.
- All 40,000 coverage requests form role-free latent families before split.
- Exactly 10,000 intentional hard-KEEP/minimal-EDIT counterfactual families are
  preserved.
- The previous cross-role signature collision mechanism is injected as a
  negative fixture and produces a hard audit failure.
- Small non-overwriting latent-plan build and independent verification: PASS.
- Training interlock negative and positive fixtures: PASS.
- Repository whitespace audit: PASS.
- Customer50 or accessed OmniDocBench inputs used: `false`.
- Model training started: `false`.

## Frozen Stage B artifact hashes

| Artifact | SHA256 |
|---|---|
| `src/borderless_table_structuring/synthetic_data.py` | `ec33a428408f45e625eb44befd259998525271cc875cf505dfc9ce62233e779f` |
| `src/borderless_table_structuring/corpus_release.py` | `9ec9c83ee1c77b0273d783d4724cffd29c3e4330fe450021f04df658f1feaf74` |
| `src/borderless_table_structuring/training_interlock.py` | `0b620355a4a68493841a84da3ff4ba6c2024698e6bb5428b8d1bdc7bd3798170` |
| `scripts/plan_shared_corpus_2026.08.12.1.py` | `e8414b7c11d3cc906f7d4a8760adf83ef351e8b61b7ff51151b7f0bbf6c661c1` |
| `configs/corpus_release_parameters_2026.08.12.6.json` | `fbf8791b32422835a19aed32d623fb7adefa0ed9514e6f5091e9f7fe9d75e686` |
| `schemas/synthetic_table_record_2026.08.12.6.json` | `473221f689c478422797a493ced8af79f26ff58af2ea9ee70de2cf42eced8489` |
| `tests/test_corpus_release_2026_08_12_6.py` | `1e0264838cd6dd5ea4b758a6959462acf1dec2302f1f13f5730ad87bd1393aba` |
| `tests/test_training_interlock_2026_08_12_1.py` | `a73ad5a1890f64d050b3072c4e357fa7f895d77b34a73bc82f7d0726da78ba6d` |

## Decision

Stage B passes its implementation and data-free verification gate. No 40,000
record latent plan and no rendered corpus has been generated. Per the user's
explicit instruction, Stage C must not start until a separate user-facing
notification is issued. Passing Stage B does not authorize model training.
