# Dataset Storage and Sharing

## Recommended architecture

Use two repositories with different responsibilities:

1. `DearKarl/borderless-table-structuring-lab` stores code, schemas,
   generators, manifests, checksums, tests, and reproducibility records.
2. `DearKarl/borderless-table-structuring-data` stores authorized dataset
   payloads in a private dataset service or object store.

The code repository pins the dataset by immutable dataset revision and the
SHA256 of its root manifest. Mutable names such as `latest` are not valid
experiment inputs.

## Preferred payload format

For images and JSON records, use WebDataset shards between 256 MiB and 1 GiB.
For metadata-only tables, Parquet is acceptable. Keep one small JSONL manifest
that maps each sample to its shard, byte member, role, provenance, license,
family IDs, and hashes.

Suggested layout:

```text
borderless-table-structuring-data/
├── README.md
├── schemas/
├── licenses/
├── manifests/
│   ├── corpus.jsonl
│   ├── train.jsonl
│   ├── development.jsonl
│   ├── holdout.jsonl
│   └── quarantine.jsonl
├── audits/
│   ├── exact_overlap.json
│   ├── perceptual_overlap.json
│   ├── structure_overlap.json
│   ├── text_overlap.json
│   └── geometry_overlap.json
└── shards/
    ├── train-000000.tar
    ├── development-000000.tar
    └── holdout-000000.tar
```

## Service choice

A private Hugging Face Dataset repository is the simplest option for a small
research team when the corpus is redistributable: it supports private access,
versioned revisions, large files, metadata cards, and direct loading.

Use private S3-compatible object storage with DVC when licensing, institutional
policy, scale, or access logging requires tighter control. GitHub LFS is not the
preferred primary dataset store because storage and bandwidth quotas make
iterative image-corpus work expensive and fragile.

## Sharing with a collaborator

Grant the collaborator access to the code repository and, separately, to the
minimum dataset role they need. Do not share credentials. Record access by
account identity, dataset revision, role, and date.

The collaborator should initially receive the synthetic training and
development roles. Holdout and terminal roles remain inaccessible to anyone
performing generator or model optimization.

## Existing derived records

Do not upload compiled records until the source-license inventory confirms
redistribution and downstream-use rights for every source.
If redistribution is not authorized, share only:

- the compiler;
- the schema;
- a source acquisition guide;
- an expected count and SHA256 manifest;
- deterministic replay tests.

Each collaborator then rebuilds the derived layer from a legally obtained
source copy.
