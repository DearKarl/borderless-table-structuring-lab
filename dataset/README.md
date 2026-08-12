# Dataset Registry

This directory contains no dataset payloads. It records the immutable external
dataset references used by reproducible experiments.

## Active dataset entry template

```yaml
dataset_name: data-2026.08.12.1
service: private-hugging-face-or-approved-object-store
repository: DearKarl/borderless-table-structuring-data
revision: null
root_manifest_sha256: null
schema_release: synthetic-table-record-2026.08.12.1
research_snapshot: 2026.08.12.1
roles_available: []
license_manifest_sha256: null
overlap_audit_sha256: null
experiment_record_sha256: null
status: not_yet_frozen
```

Do not fill this entry with a mutable branch name. Record the immutable
revision and hashes only after the corpus build and audits are sealed.
