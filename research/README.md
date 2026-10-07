# Research guide

The project objective is to exceed the OmniDocBench leaderboard leader with an
officially recognized Overall score. TeleOCR is the current target; V0–V4 are
attempts, not commitments to a particular method.

The [completed V5 study](../versions/v5/README.md) compares five input and
regional-rereading systems; V5.5 reaches local Overall 98.347514 on 1,651 pages.
The [original proposal](V5_PROPOSAL.md) preserves its design rationale and the
CCF A publication aspiration; future research steps remain proposals.

Start with [RESULTS.md](RESULTS.md) for the experimental results and
[CURRENT_STATE.md](CURRENT_STATE.md) for completed evidence and limits,
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what can be checked or reproduced,
and [ROADMAP.md](ROADMAP.md) for the remaining milestones. Use the
[experiment record template](EXPERIMENT_TEMPLATE.md) for future work. The current
index covers this project's work; future group contributions need their own
attributed evidence records before they enter the result summary.

| Location | Purpose and entry point |
|---|---|
| [V5 study](../versions/v5/README.md) | Completed five-arm methods, results, costs and limits |
| [Source inventory](SOURCE_INVENTORY.md) | Historical V0-V4 implementation map and evidence gaps |
| [V4 study](V4.md) | Learned input selection: method, experiment, limitations, source and model |
| [Experimental results](RESULTS.md) | V0–V5 measurements, controls, component scores and supplied V3 results |
| [Version archive](../versions/README.md) | V0-V3.2 and V5 implementations and version-specific runtime contracts |
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
