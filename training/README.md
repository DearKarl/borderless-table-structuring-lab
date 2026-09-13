# Training route — Explicit-v2 Original

This is the best project-trained Explicit checkpoint in completed full Official-protocol evaluations, separate from the [Hybrid route](../hybrid/README.md). The exact weight has been recovered intact and verified against the archived prediction seal.

| Property | Value |
|---|---|
| Checkpoint directory label | `checkpoint-003111-joint_low_lr` |
| Full Table TEDS | 93.0862668980718 |
| Structure-only TEDS | 95.70961193860337 |
| Adopted edits | 0 across 764 table records |
| Interpretation | Equal to fixed MinerU Raw, not a learned net gain |
| Public leaderboard acceptance | Not verified |
| Weight | 4,754,162 bytes; custom non-pickle safe-state |
| Portable verification | Strict checkpoint loading and CPU tensor-level inference |

The directory label is historical. Saved metadata records **update 894**, epoch 2, stage `joint_low_lr`, and 14,298 exposures. Do not reinterpret `003111` as the saved update. Identity comes from the weight SHA and prediction seal. No lower-scoring checkpoint is substituted.

## Download and run on your computer

From the repository root, using Python 3.10 or newer:

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r training/requirements.txt
python training/download_model.py --output models/training/model.safe-state
python training/verify_checkpoint.py --checkpoint-dir models/training
python training/run_inference.py --checkpoint models/training/model.safe-state --smoke
```

On Windows, activate `.venv-training\Scripts\activate` instead. The downloader verifies the fixed release file's size and SHA256 before publication and refuses an existing destination. Installation requires network access; CPU inference runs offline without a VPN, cluster account, GPU, optimizer, or benchmark data. The packaging runtime tested is Python 3.10, NumPy 2.2.6, and Torch 2.12.0, not a claim of original evaluator runtime parity.

The smoke command loads all **135 state tensors** strictly and performs one forward pass on a project-authored two-cell synthetic input. It reports shapes and identity, **not accuracy, TEDS, correction quality, or score reproduction**. The archived model and safe-state reader are unchanged. File hashes are in [model_manifest.json](model_manifest.json).

## Real inputs and limitations

This is **not a standalone PDF/image-to-table parser**. Upstream MinerU/bridge/OCR preprocessing must already have produced the appropriate tensor encodings. Use `--input tensors.json --output logits.json` for your own precomputed tensors. The JSON object must contain exactly these fields:

| Field | Shape / meaning |
|---|---|
| `image` | `[B,1,H,W]`, preprocessed grayscale image |
| `cell_features` | `[B,C,16]`, archived cell feature encoding |
| `cell_boxes` | `[B,C,4]`, normalized x0/y0/x1/y1 |
| `cell_mask` | `[B,C]`, Boolean valid cells |
| `grid_shape` | `[B,2]`, row/column counts in 1–64 |
| `token_features` | `[B,T,12]`, archived OCR token feature encoding |
| `token_mask` | `[B,T]`, Boolean valid tokens |
| `token_owner_candidate_mask` | `[B,T,C+1]`; final unresolved class available |
| `text_candidate_token_weights` | `[B,C,K,T]`, nonnegative token weights |
| `text_candidate_mask` | `[B,C,K]`, Boolean available candidates |
| `text_candidate_is_raw` | `[B,C,K]`; Raw fallback present with zero token-copy weights |

`B`, `C`, `T`, and `K` mean batch size, cells, tokens, and text candidates. The synthetic fixture is an interface example, not a replacement feature compiler. Do not use arbitrary features and claim the historical score.

The wrapper returns logits and hidden states. It does **not** run the historical bridge, confidence-gated decoder, SafeExecutor, page assembly, or Official evaluator. Proposals are not committed edits. The complete historical correction pipeline is not claimed portably reproduced here. For a runnable image/Raw-HTML correction adapter, see the separate [Hybrid guide](../hybrid/README.md) and its limitations.

Cross-modal and grounding-aware decision fusion remain enabled. Frozen thresholds and aggregate provenance remain in [model_manifest.json](model_manifest.json); the tensor wrapper does not apply or change that historical policy.

## Provenance and preservation

Weight SHA256:
`c1615ce37de058e82b0ec33a84fb06afb25d4f6d4dfa401ff1e67f9d547019f0`.

Original private checkpoint-manifest SHA256:
`5db28f62ee2f40e1e7a56903f76d6bc52089c8c358646c5e7969d079b4d1cafd`.

The public release includes only the exact inference weight and sanitized model metadata. The full original checkpoint, optimizer state, training state, and manifest are preserved in the owner's local archive. Internal paths, credentials, benchmark/customer data, and historical contracts are not distributed. The historical score remains below the strict >95 objective.
