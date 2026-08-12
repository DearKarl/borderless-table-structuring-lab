# Contributing

## Before starting

Read the active execution contract and its seal in `governance/` in full. A
contribution must stay inside the current authorization boundary.

## Branch and review workflow

1. Create a focused branch from the current default branch.
2. Declare the hypothesis and single variable in the pull request.
3. Add or update deterministic tests.
4. Record source, license, split role, and isolation implications.
5. Add an Evidence Card for any data-affecting stage.
6. Request review before merging.

## Required checks

- `pytest` passes.
- No dataset payload, model weight, credential, or terminal artifact is added.
- All new filenames, documentation, code comments, and manifests are English.
- Generated outputs use new, non-overwriting paths.
- SHA256 manifests and complete failure accounting are present where required.
- Exact and near-duplicate audits pass before a corpus role is frozen.

## Prohibited content

Do not commit Customer50 or OmniDocBench terminal pages, crops, annotations,
recognized strings, coordinates, HTML, LaTeX, identifiers, Gold, predictions,
or derived near-duplicates. Do not commit source datasets unless their
redistribution and downstream-use rights have been explicitly approved.

## Commit messages

Use an imperative English summary and keep each commit scoped to one logical
change. Example:

```text
Add deterministic span-template generator
```
