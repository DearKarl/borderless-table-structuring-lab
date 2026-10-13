# Contributing

Keep scientific outputs and reproduction sources under their V0–V8 directory.
Shared application, evaluator and testing code may remain at the root.

Record metric units, full denominators, input/model/evaluator identity, failures,
controls and costs with results. Preserve negative findings and distinguish
local measurements from accepted leaderboard results. Do not publish private
inputs, ground truth, predictions, credentials or deployment bindings.

Keep original third-party license texts with the corresponding code/model
references. An upstream result is not a result measured by this project.

Run `python -m pytest -q tests` for shared implementation or path changes.
Check version-specific interfaces when their source changes. These tests verify
program behavior; they do not reproduce model accuracy. Check documentation
links and `git diff --check` before committing.

Local workflow, handoffs and operating records belong in ignored `codex/`.
