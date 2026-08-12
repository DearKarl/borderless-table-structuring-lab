# Borderless Table Structuring Lab

Research infrastructure for auditable borderless-table structure recognition,
Canonical Table supervision, safe Raw MinerU refinement, and independent
Explicit and LoRA candidate routes.

This repository is designed as the long-lived project home. The initial
revision contains the data-engineering and safety-integration layers only. It
does **not** contain model weights, training payloads, terminal benchmark pages,
Customer50 artifacts, or per-sample terminal predictions.

## Table of contents

- [Research objective](#research-objective)
- [Current repository scope](#current-repository-scope)
- [Repository layout](#repository-layout)
- [System design](#system-design)
- [Data strategy](#data-strategy)
- [Installation](#installation)
- [Tests](#tests)
- [Reproducibility and evidence](#reproducibility-and-evidence)
- [Collaboration workflow](#collaboration-workflow)
- [Roadmap](#roadmap)
- [Governance and licensing](#governance-and-licensing)

## Research objective

The project targets table-quality improvement under the OmniDocBench document
parsing protocol while preserving the Raw MinerU document baseline. The core
engineering principle is selective, auditable table correction:

1. Raw MinerU remains the default output.
2. The Explicit route may propose minimal topology-only corrections with Raw
   OCR text frozen.
3. The LoRA route may propose one complete, table-only Canonical Table state.
4. Both routes pass through the same legality, token-preservation, geometry,
   provenance, expected-gain, assembly, and exact-Raw-rollback controls.
5. Unsafe or unsupported candidates are rejected without modifying Raw.

The target of Table TEDS above 95 is an engineering objective, not a guaranteed
unobserved result. Public benchmark-aware development and independent terminal
generalization must be reported separately.

## Current repository scope

Included in the first revision:

- Canonical Table normalization and structural label utilities.
- Direct-state and order-invariant target compilation.
- Candidate-integrity checks.
- Shared fail-closed validation and deterministic Raw rollback.
- Explicit topology-only and LoRA complete-table candidate interfaces.
- Synthetic unit fixtures and regression tests.
- Canonical record schema.
- Evidence Cards and the active execution contract.
- Dataset governance, storage, reproducibility, and collaborator handoff
  documentation.
- Public OTSL normalization and fixed-denominator paired-metric utilities.
- Synthetic-data provenance guidance and manifest validation.

Explicitly excluded:

- Model implementations, adapters, checkpoints, or weights.
- Full training corpora or rendered sample payloads.
- Formal20k source records and compiled record payloads.
- Customer50 content.
- OmniDocBench pages, crops, annotations, recognized strings, coordinates,
  HTML, LaTeX, page identifiers, or Gold records.
- Per-sample terminal predictions or case-selection artifacts.

## Repository layout

```text
borderless-table-structuring-lab/
├── .github/
│   └── workflows/             # Continuous integration
├── dataset/
│   └── README.md              # External dataset registry and access policy
├── docs/
│   ├── COLLABORATOR_HANDOFF.md
│   ├── DATA_GOVERNANCE.md
│   ├── DATASET_STORAGE_AND_SHARING.md
│   ├── REPRODUCIBILITY.md
│   └── SYNTHETIC_CORPUS_SPECIFICATION.md
├── evidence/                  # Nonterminal Evidence Cards only
├── governance/                # Active execution contract and seal
├── schemas/                   # Versioned record schemas
├── scripts/                   # Deterministic data-layer compilers
├── src/mpr_tsr_splitmerge_v2/
│   ├── canonical.py
│   ├── candidate_integrity.py
│   ├── candidate_interfaces.py
│   ├── labels.py
│   └── safety_layer.py
├── tests/                     # Synthetic and deterministic unit tests
├── CONTRIBUTING.md
└── pyproject.toml
```

## System design

The active representation is the final Canonical Table state plus an
order-invariant structural difference. Historical ordered KEEP/SPLIT/MERGE
programs are replay evidence, not primary supervision, because multiple action
paths may lead to the same correct table.

The shared safety boundary checks:

- Canonical legality and complete grid coverage.
- OCR token ownership and text preservation.
- Finite, non-degenerate physical geometry.
- Provenance and table-only scope.
- Frozen positive expected-gain evidence.
- Non-table page-state stability.
- Exact deterministic Raw rollback.

See [the synthetic corpus specification](docs/SYNTHETIC_CORPUS_SPECIFICATION.md)
for the data contract and [the active execution contract](governance/POST_OMNIDOCBENCH_EXECUTION_CONTRACT_V6.md)
for the current authorization boundary.

## Data strategy

Training payloads are intentionally stored outside this Git repository.
GitHub stores schemas, manifests, checksums, generators, validators, and
Evidence Cards. A separate private dataset repository stores authorized data
shards and is pinned by immutable revision and SHA256 manifest.

Recommended dataset repository name:

```text
DearKarl/borderless-table-structuring-data
```

The preferred format is versioned WebDataset or Parquet shards with a small
JSONL manifest. See [dataset storage and sharing](docs/DATASET_STORAGE_AND_SHARING.md)
and [the dataset registry](dataset/README.md).

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Tests

Run the complete data-engineering test suite:

```bash
pytest
```

The committed tests use synthetic fixtures. They do not require terminal
benchmark data, model weights, or a GPU.

## Reproducibility and evidence

Every data stage must freeze:

- source and permitted-use inventory;
- schema version and compiler revision;
- deterministic seeds and render parameters;
- train, development, holdout, and terminal roles;
- exact and near-duplicate isolation reports;
- generated, accepted, quarantined, and failed counts;
- complete SHA256 manifests;
- an English Evidence Card and seal.

Detailed requirements are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Collaboration workflow

1. Read the active contract and seal before project work.
2. Work on a dedicated branch.
3. Change one declared variable per experiment.
4. Add tests and an Evidence Card for data-affecting changes.
5. Never commit datasets, credentials, weights, or terminal artifacts.
6. Submit a pull request with source, license, isolation, and replay evidence.

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[COLLABORATOR_HANDOFF.md](docs/COLLABORATOR_HANDOFF.md).

## Roadmap

- [x] Canonical Data Layer vNext.
- [x] Shared validator and exact Raw rollback.
- [x] Independent Explicit and LoRA candidate interfaces.
- [x] Bounded correctness smoke.
- [ ] Freeze the synthetic-corpus specification and collaborator handoff.
- [ ] Implement bounded generators and pass the data smoke.
- [ ] Build a frozen shared corpus with complete overlap audits.
- [ ] Run independent bounded Explicit and LoRA candidate pilots.
- [ ] Stop at the mandatory pre-full-training discussion gate.
- [ ] Add model code only after the corresponding governance decision.

## Governance and licensing

The repository is private and currently has no public redistribution license.
Project code and newly authored synthetic assets require an explicit licensing
decision before public release. Third-party datasets retain their own licenses
and must not be redistributed merely because this repository is private.

OmniDocBench and Customer50 terminal contents are prohibited from the training
and development corpus. Any source with uncertain redistribution or downstream
use rights must remain external until the license review passes.
