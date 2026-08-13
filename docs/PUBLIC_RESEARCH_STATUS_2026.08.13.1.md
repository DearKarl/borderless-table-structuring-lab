# Public Research Status 2026.08.13.1

## Shared corpus

The shared project-generated deterministic corpus is sealed at 40,000 records:
28,000 training records, 8,000 development records, and 4,000 holdout records.
It contains four balanced 10,000-record categories: exact KEEP, hard KEEP,
single minimal edit, and complex correction. It also includes 10,000 complete
counterfactual groups.

Targets represent final Canonical Table states and order-invariant structural
differences rather than one arbitrary ordered edit program. Each record keeps
token ownership, geometry, renderer and source provenance, and role-isolation
metadata. Dataset payloads are not distributed in this repository.

## Route status

The Explicit adapter and E0 assurance have passed. E1 real-data local-smoke
work is in preparation. The Explicit route is topology-only, defaults to KEEP,
uses frozen text, proposes minimal split/merge changes, and uses shared safety
validation with exact Raw rollback.

The LoRA model-free adapter has passed. L0 processor and token-audit work is in
preparation. The LoRA route uses a frozen visual path with language-decoder
adapters to produce one complete table-only Canonical candidate; strict parsing,
shared safety validation, and exact Raw rollback remain mandatory.

Neither local training smoke nor full training has completed for either route.
