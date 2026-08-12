# Acceptance Criteria 2026.08.12.4

Status: `FROZEN_FOR_GENERATOR_SMOKE`

## Preregistered smoke gate

`data-smoke-2026.08.12.4` passes only when all criteria below pass. No failed
seed may be silently replaced, no category quota may be rebalanced after
generation, and no model performance metric may be used to select records.

| Criterion | Required result | Failure action |
|---|---:|---|
| Requested records | 256 | Stop |
| Generated records | 256 | Stop |
| Accepted records | 256 | Stop |
| Exact KEEP / Hard KEEP / Single / Complex | 64 / 64 / 64 / 64 | Stop |
| Counterfactual pairs | at least 64 complete same-role pairs | Stop |
| Schema validation | 256/256 | Stop |
| Canonical legality of Gold and prior | 256/256 | Stop |
| Complete grid coverage of Gold and prior | 256/256 | Stop |
| Gold recompilation | 256/256 semantic identity | Stop |
| Unique OCR-token ownership | 100% of owned tokens | Stop |
| Finite complete geometry | 256/256 | Stop |
| KEEP semantic identity | 128/128 | Stop |
| EDIT replay to Gold | 128/128 | Stop |
| Explicit KEEP empty diff | 128/128 | Stop |
| LoRA target completeness | 256/256 | Stop |
| Semantic replay | identical across 2 runs | Stop |
| Normalized-pixel replay | identical across 2 runs | Stop |
| Source and license inventory | complete and approved | Stop |
| Cross-role exact overlap | 0 | Stop |
| Cross-role unresolved near overlap | 0 | Stop |
| Terminal-input usage | false for every record | Stop |
| Incomplete-grid rejection fixture | rejected as `GRID_INCOMPLETE` | Stop |
| Complete count accounting | requested/generated/accepted/quarantined/failed | Stop |

## Validation reason codes

Every failure uses a stable reason code, including:

- `SCHEMA_INVALID`;
- `CANONICAL_ILLEGAL`;
- `GRID_INCOMPLETE`;
- `GOLD_RECOMPILE_MISMATCH`;
- `TOKEN_OWNERSHIP_INVALID`;
- `GEOMETRY_INVALID`;
- `KEEP_NOT_IDENTITY`;
- `EDIT_NOT_REPLAYABLE`;
- `NONDETERMINISTIC_SEMANTICS`;
- `NONDETERMINISTIC_PIXELS`;
- `SOURCE_OR_LICENSE_INCOMPLETE`;
- `CROSS_ROLE_EXACT_OVERLAP`;
- `CROSS_ROLE_NEAR_OVERLAP`;
- `TERMINAL_DERIVATION_RISK`.

Quarantine manifests preserve every failed record identifier, frozen seed,
reason code, and diagnostic hash. The intentionally incomplete-grid fixture is
test evidence and is not part of the 256-record corpus manifest.

## Post-pass boundary

A sealed pass authorizes a new non-overwriting shared-corpus build with frozen
distributions. It does not authorize Explicit or LoRA training. Corpus scale is
chosen from measured generation cost, coverage, and quarantine rate, not from
current official benchmark scores.
