# Contributing

Keep Training and Hybrid changes separate. Do not promote a different checkpoint, policy, runtime, or unscored candidate under an existing result. Model releases are dated and SHA256-bound.

For each change, state its purpose, evidence, affected input/output contract, checks, and limitations. Separate integration checks from benchmark performance. Preserve failures and Raw fallback.

Do not commit benchmark/customer payloads, credentials, internal handoffs, historical contracts, or model binaries. Use licensed Release assets or pinned upstream model downloads. Verify checkpoint identity, not just its filename.

Run the data-free package checks:

```bash
python -m pip install pytest Pillow
python -m pytest -q tests
```

The current phase is handoff and consolidation. Another optimization or training campaign requires a new owner decision. Earlier research snapshots remain in Git history.
