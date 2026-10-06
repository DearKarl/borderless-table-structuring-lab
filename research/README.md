# Research guide

The project objective is to exceed the OmniDocBench leaderboard leader with an
officially recognized Overall score. TeleOCR is the current target; V0–V4 are
attempts, not commitments to a particular method.

Start with [CURRENT_STATE.md](CURRENT_STATE.md) for completed evidence and limits,
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what can be checked or reproduced,
and [ROADMAP.md](ROADMAP.md) for the remaining milestones. Use the
[experiment record template](EXPERIMENT_TEMPLATE.md) for future work. The current
index covers this project's work; future group contributions need their own
attributed evidence records before they enter the result summary.

| Location | Purpose and entry point |
|---|---|
| [V4 release](../releases/v4-gbdt-evaluation-20261006/README.md) | GBDT + TeleOCR experimental delivery, original model, frozen source, results and provenance |
| [Version archive](../versions/README.md) | V0-V3.2 implementations and version-specific runtime contracts |
| [Hybrid](../hybrid/README.md) | Historical OCR sidecar and [native-table route](../hybrid/native_tables/README.md); route guides retain their dated scope |
| [Training](../training/README.md) | Separate Explicit-v2 checkpoint and tensor interface; not the V4 GBDT model |
| [btsl/](../btsl/) | Earlier table-structuring package, also used by preserved V0 paths |
| [Whole-page results](../artifacts/whole-page-results.json) | Historical aggregate scores and protocol/source identities |
| [Table result archive](../artifacts/results/README.md) | Training, sidecar and native-table aggregate evidence, including failures |
| [Frozen evaluator](../evaluation/omnidocbench/README.md) | Preserved OmniDocBench source; no ground-truth or prediction payloads |
| [scripts/](../scripts/) and [examples/](../examples/) | Existing utility and example entry points; read each route's prerequisites before use |
| [tests/](../tests/) | Data-free implementation checks, not benchmark reproduction |
| [Notices](../notices/THIRD_PARTY.md) | Attribution and dependency boundaries |

There is no root `docs/` directory in this checkout. Existing route and release
documentation remains in place. Run archived versions from their documented
directories because package names are deliberately preserved. New work must not
overwrite frozen source, released models, results, or original reports.

See [contribution rules](../CONTRIBUTING.md) and the [root README](../README.md).
