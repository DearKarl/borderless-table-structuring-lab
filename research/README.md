# Research guide

All engineering is paused as of2026-10-07. Start with [current state](CURRENT_STATE.md)
for the completed V6 evidence, V7 negative training result and next-route discussion.

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
| [V6 study](../versions/v5_native/README.md) | Seven completed outputs with all metrics and negative results |
| [V7 study](../versions/v7/README.md) | CPU model and four configs; independent confirmation paused |
| [Literature scan](LITERATURE_SCAN_20261007.json) | Primary-source evidence, reading limits and unexecuted generator proposal |
| [Progress archive](../releases/v6-v7-progress-20261007/README.md) | First-party checkpoints, source and publication hashes |
| [V5 study](../versions/v5/README.md) | Completed five-arm methods, results, costs and limits |
| [Default-input adaptive scale plan](ADAPTIVE_NATIVE_SCALE_PLAN.md) | Historical design: numerical scale control with native input and effective-pixel verification |
| [Learned candidate selection](LEARNED_CANDIDATE_SELECTION.md) | Proposed trained selection with invocation, effective-change and quality checks |
| [Distillation plan](DISTILLATION_PLAN.md) | Proposed student training, external data, controls and quality/cost criteria; not executed |
| [Source inventory](SOURCE_INVENTORY.md) | Historical V0-V4 implementation map and evidence gaps |
| [V4 study](V4.md) | Learned input selection: method, experiment, limitations, source and model |
| [Experimental results](RESULTS.md) | V0–V7 measurements, controls, component scores and supplied V3 results |
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
