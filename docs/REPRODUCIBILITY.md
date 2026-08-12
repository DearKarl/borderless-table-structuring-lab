# Reproducibility

## Immutable inputs

Each run records the Git commit, dataset repository and revision, root manifest
SHA256, schema version, configuration hash, generator seeds, split roles, and
runtime environment.

## Non-overwriting outputs

Never reuse an output directory. A run writes to a new directory containing:

- `PREREGISTRATION.md`;
- `run_config.json`;
- `input_manifest.json`;
- `report.json`;
- `quarantine.jsonl`;
- `SHA256SUMS`;
- `EVIDENCE_CARD.md`;
- an Evidence Card seal.

## Determinism

Data compilers and generators must replay the same semantic records and hashes
from the same declared inputs. Image encoders that are not byte-deterministic
must still reproduce normalized pixels and semantic records and must document
the normalization procedure.

## Evaluation discipline

Development metrics may compare bounded single-variable changes against the
frozen Raw baseline. Holdout and terminal roles are excluded from generator,
threshold, checkpoint, architecture, and data selection. Any future full
training command requires a separate explicit approval at the contract gate.

## Evidence review

Reviewers verify source rights, role isolation, complete failure accounting,
terminal non-use, and exact Raw rollback before accepting a stage transition.
