# Training route — Explicit-v2 Original

This is the best project-trained Explicit checkpoint found in completed full
Official-protocol evaluations, distinct from the [Hybrid route](../hybrid/README.md).

| Property | Value |
|---|---|
| Checkpoint | `checkpoint-003111-joint_low_lr` |
| Full Table TEDS | 93.0862668980718 |
| Structure-only TEDS | 95.70961193860337 |
| Adopted edits | 0 across 764 table records |
| Interpretation | Equal to MinerU Raw fallback, not a learned net gain |
| Public leaderboard acceptance | Not verified |
| Downloadable trained weights | **Not available in this release yet** |

The exact winning weights are missing from the local handoff workspace. A
digest is not the model. A locally available older epoch-08 checkpoint scored
lower and is not substituted. Hybrid's third-party MinerU weights are NOT this
project-trained checkpoint either.

The model uses cross-modal and grounding-aware decision fusion. Frozen
thresholds, source closure, and aggregate evidence are recorded in
[model_manifest.json](model_manifest.json). Do not retune them while claiming
the archived result. Later training did not establish an improved deployed
repair policy; this checkpoint remains an honest reference.

## Restore the exact artifact

Ask the project owner for the original checkpoint folder, including
`MANIFEST.json` and `model.safe-state`. Once a copy is available:

```bash
python training/verify_checkpoint.py --checkpoint-dir /path/to/checkpoint-003111-joint_low_lr
```

This is read-only file hashing, not tensor deserialization, inference, or a
performance check. Incorrect or missing files are rejected. Only after this
passes can the owner package the checkpoint with matching inference code and
publish a Training Release. No portable inference command is claimed tested
without the exact weights.

Weight SHA256:
`c1615ce37de058e82b0ec33a84fb06afb25d4f6d4dfa401ff1e67f9d547019f0`.

Checkpoint-manifest SHA256:
`5db28f62ee2f40e1e7a56903f76d6bc52089c8c358646c5e7969d079b4d1cafd`.

This is a restoration blocker, not a completed weight upload or a request to
restart training. No historical contracts or benchmark payloads are included.
