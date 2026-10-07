# Reproduction guide

Current update2026-10-07: all engineering is paused for research discussion.
See [current state](CURRENT_STATE.md), [V6 full results](../versions/v5_native/README.md),
[V7 training and development](../versions/v7/README.md), and the
[progress archive](../releases/v6-v7-progress-20261007/README.md).
The sections below retain historical evidence or earlier milestones; they do not
authorize resuming experiments. No official ranking or new generator model is claimed.


## Choose the claim you want to check

| Task | Entry point | What it establishes |
|---|---|---|
| Inspect reported scores | [Historical aggregate](../artifacts/whole-page-results.json), [V4 report](../releases/v4-gbdt-evaluation-20261006/results/REPORT.json) | Saved measurements and declared provenance |
| Check root implementation | `python -m pytest -q tests` after root test installation | Data-free implementation behavior |
| Verify V4 source and model files | Command below | Files match the frozen experiment manifest |
| Reproduce an archived route | [Version guides](../versions/README.md), [native-table guide](../LEGACY_NATIVE_TABLES.md), [Training guide](../training/README.md) | Requires each route's pinned environment, assets and protocol |
| Reproduce V4 inference and scoring | [V4 archive guide](../releases/v4-gbdt-evaluation-20261006/README.md) and its dependency manifests | Requires external assets and new validated deployment bindings |
| Establish an official rank | [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Requires official comparison conditions and acceptance; not established here |

The root package does not install all archived versions. Identically named Python
packages in different snapshots must be run from their documented directories.

## V5 source and completed evidence

The [V5 report](../versions/v5/README.md), [protocol](../versions/v5/protocol.json)
and [aggregate](../artifacts/results/v5/AGGREGATE.json) identify the completed
five-arm experiment. With the test dependencies available, run from the root:

```bash
python -m unittest versions.v5.test_regions versions.v5.test_analysis -v
```

These 13 synthetic checks do not rerun inference. Full reproduction additionally
requires the pinned TeleOCR source, all model/processor files, native runtime,
benchmark inputs, evaluator dependencies and a newly bound execution manifest.
Private runtime manifests and benchmark payloads are not distributed. A fresh
clone alone is not a complete one-command benchmark installation. The requested
model repository revision was not verified; the completed report records the
actual file identities and the unresolved revision distinction.

All five runs retain the 1,651-page denominator. Separate native runs produced
some different outputs, so reproducing a system comparison is distinct from
isolating a patch's causal contribution on a shared native prediction.

## Data-free V4 archive check

From the repository root, with Python available:

```bash
cd releases/v4-gbdt-evaluation-20261006
python -B verify_delivery.py
```

The verifier reads files without loading the model pickle, running inference or
accessing the network. Passing checks file integrity; it does not reproduce
model predictions or accuracy.

## V4 environment and evidence

The archive includes the project-trained GBDT, source, dependency records and
aggregate metrics. External TeleOCR weights/source, auxiliary assets, the native
environment, benchmark images and ground-truth bodies are not bundled. Inspect
[external asset identities](../releases/v4-gbdt-evaluation-20261006/dependencies/EXTERNAL_ASSETS.json)
and the [publication boundary](../releases/v4-gbdt-evaluation-20261006/PUBLICATION.md).
Public deployment placeholders are not runnable machine bindings.

The model's disk configuration keeps `learned_enabled=false`. The measured
experimental entry enabled the same model in memory with margin 0.0, fallback B
and candidate actions A/B. Loading the default disk configuration does not
reproduce that policy. Do not remove validation gates to make a new deployment
start; give it explicit bindings, fresh output paths and a new provenance record.

The recorded full run contains 1,651 pages: 1,644 completed and seven timeout
predictions retained empty. Overall is 97.17188999355706. The exact evaluator
commit and configuration hash are in the report; a matching evaluator source
alone does not establish matching runtime settings or predictions.

## Comparing results

Overall = (100 × (1 − Text ED) + Formula CDM + Table TEDS) / 3, when CDM and
TEDS are expressed on a 0–100 scale. The V4 JSON stores those component values on
a 0–1 scale. Reading-order ED and structure TEDS do not enter Overall.

Historical TeleOCR V2 raw 97.435109, Hybrid V2 97.060019, V4 97.171890 and the
frozen public TeleOCR reference 96.91 must remain distinct. The V2 pair supports
a negative hybrid result. The cross-version and public-reference comparisons do
not establish causal GBDT benefit. Truncation and fallback policies differ across
some historical groups; consult the original result records.

Before reporting a new improvement, freeze dataset revision and hashes, input
rendering, model revision, prompts/decoding, preprocessing, failure handling,
evaluator configuration, denominator, baseline and cost accounting. Preserve
ground truth for evaluation only. Retain failed pages under the declared rule.

Use the [experiment template](EXPERIMENT_TEMPLATE.md) to record a future run.
The source archive is useful evidence, but complete fresh-clone benchmark
reproduction and official leaderboard acceptance remain separate milestones.

## V3 reported results

The [current results](RESULTS.md) include V3.1 and V3.2 measurements supplied by
the researcher on 2026-10-07. Their source-code entry points are available, but
raw scoring files, exact scored revisions and some metrics remain unspecified.
Those values are recorded as supplied results, not as reproduced experiments.
