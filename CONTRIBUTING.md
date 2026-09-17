# Contributing

Keep Training and Hybrid changes separate. Do not promote a different checkpoint, policy, runtime, or unscored candidate under an existing result. Model releases are dated and SHA256-bound.

For each change, state its purpose, evidence, affected input/output contract, checks, and limitations. Separate integration checks from benchmark performance. Preserve failures and Raw fallback.

Do not commit benchmark/customer payloads, credentials, internal handoffs, historical contracts, or model binaries. Use licensed Release assets or pinned upstream model downloads. Verify checkpoint identity, not just its filename.

Run the data-free package checks:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
```

The native-table research evaluation completed on 2026-09-17. The integrated
application is now the main Hybrid entry point. Keep its historical scored
result, any new portable execution, the old OCR sidecar and Training distinct.
Do not claim package tests reproduce benchmark performance. Changes to an
active run's code/input bindings are rejected; use a new output directory.
Keep the extracted scientific functions and frozen upstream evaluator source
unchanged unless a new, explicit recipe and its evidence are provided.
