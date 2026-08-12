# Evidence Card: Canonical Data Layer vNext V1

Status: `SEALED_COMPLETE`

Execution date: 2026-08-11 (Asia/Shanghai)

## 1. Hypothesis

Direct Canonical Table state and order-invariant partition differences remove arbitrary action-path supervision while preserving the verified source evidence and Canonical Gold.

## 2. Authorization and isolation

- Active authority: `POST_OMNIDOCBENCH_EXECUTION_CONTRACT_V4.md`.
- Dataset role: Formal20k source data and Gold, used only to compile a new versioned derived-label layer.
- Customer50 read: `false`.
- OmniDocBench terminal page or metric read: `false`.
- Full model training started: `false`.
- Formal20k v1 modified: `false`.
- Final output directory existed before the run: `false`.

## 3. Frozen inputs

- Formal20k v1 records: `20,000`.
- Formal20k v1 manifest SHA256: `716ad18e1c26a58ced016dcf8424049a4081c92e39215ad7b56268b3fd43aa99`.
- Contract V4 SHA256: `9a9b8e1fb0ce7729004e897715f2dba695af7cced14bb81203822ae3f9f3a343`.
- Canonical Data Layer contract SHA256: `a0215b45230471b3093eae160ac6a4eaef86d590c90be939306fec1bd79c33cf`.
- Schema SHA256: `c5793628fb19bdac859f9123ac3c8c0110305ac0e4726088f4f3d768c00efb30`.
- Preregistration SHA256: `b8f5cb736d7cebcfcb40a7dabd817326465c304bd7f292636e132a98ca7f5067`.
- Final compiler SHA256: `534e160d5caf7d734d399247153f15e14787cfffee585c24e4befa95752ad094`.
- Final compiler-test SHA256: `86ac57db19bec4a19e55777dfa943fb970b2059600ba3de30f048333bfe85a1e`.
- Input seal V2 SHA256: `2a74abbe7c2cd3ba9065d3294f02ce4731c95743e4ff19bb0f9eedaa0978cab9`.

## 4. Preserved pre-commit failure

Attempt 1 stopped before final output commit because historical `ocr_gold_pointer` values index primitive-grid owners rather than Canonical Gold cells. The temporary evidence was preserved and the source manifest was not modified.

- Failure evidence SHA256: `d30bf40e0d4157956517c6a5d7fd277945335970b6c4335079e3d6126a9c377b`.
- Corrective single change: compile Canonical cell token ownership from audited `text_edit_labels.cells[].ocr_token_indexes`.
- Final unit checks: `3/3 PASS`.

## 5. Final compile result

| Item | Result |
|---|---:|
| Records processed | 20,000 |
| Included records | 19,992 |
| Quarantined grid-hole records | 8 |
| Canonical Gold cells with geometry | 901,304 |
| Canonical Gold cells missing geometry | 0 |
| Historical non-unique merge-path records | 1,444 |
| Historical non-unique components | 4,175 |
| Historical non-unique Raw cells | 21,807 |
| Compile failures | 0 |

Source counts remain FinTabNet.c `8,000`, PubTables-1M `8,000`, and TabRecSet `4,000`.

## 6. Primary supervision change

- Primary truth: `DIRECT_CANONICAL_TABLE_STATE`.
- Explicit target: order-invariant topology partition difference with default `KEEP` and Raw-text-preserving policy.
- LoRA target: complete table-only Canonical candidate reference with OCR-copy preference and parallel/shared-sidecar geometry.
- Historical action programs: replay evidence only, identified by SHA256.
- Grid-hole policy: `PRESERVE_AND_QUARANTINE`.

## 7. Determinism

The complete compilation was repeated into a second non-overwriting output directory. The following files were byte-identical across runs:

- `canonical_targets.jsonl`.
- `replay_map.jsonl`.
- `included_manifest.jsonl`.
- `quarantined_manifest.jsonl`.

Deterministic replay status: `PASS`.

## 8. Final output hashes

- Canonical targets SHA256: `e4246fee34269d75d5dc24db4596a24628667b732c8495b6a2d834543def60ee`.
- Replay map SHA256: `47fa36f5657625f58c207970c57eca72ccbcaa82b814fbc198367bf568829498`.
- Included manifest SHA256: `054283401f13bac733bacd87d0e71a7085d2f096043358347e40574c7ed352fe`.
- Quarantined manifest SHA256: `97e0325820b19a0e8348a84c79e0f7d0f891f9f5544ab8da292767dad4ea29ca`.
- Compile report SHA256: `0aee80854a839716fb35a86a6aedfffa4472e82e71c43f55ee7930d0f7640931`.
- Output SHA256SUMS SHA256: `899c3e2c8df19490009c1db53bc8946dc7cfcf85d6cc0df1643f8b4e60eb5f0e`.

## 9. Hard-gate outcome

All Stage 1 hard gates passed. Canonical Data Layer vNext may progress to the shared validator and deterministic Raw rollback stage. This Evidence Card does not authorize full model training.
