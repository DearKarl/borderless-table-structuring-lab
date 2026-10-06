# Current research state

Evidence snapshot: 2026-10-06. This document indexes saved project evidence;
it does not report a new experiment or certify a fresh runtime.

## Project objective

Achieve an officially recognized OmniDocBench Overall score above the leaderboard
leader, currently TeleOCR. This is the sole project objective clarified by the
researcher. V0–V4 are historical attempts. Input-scale selection, a frozen TeleOCR
backbone, one recognition call per page, and GBDT are V4 design choices rather
than permanent requirements for subsequent research. No official leaderboard win
has been established by the saved results below.

## Completed V4 evidence

The completed local **GBDT+Tele experimental-001** run used the official
OmniDocBench v1.6 protocol on the frozen **1,651-page** input/ground-truth set.
There were **1,644 completed pages and seven timeouts**, with timeout predictions
retained empty in the full denominator; zero pages were unstarted.
Overall was **97.17188999355706**. Source:
[REPORT.json](../releases/v4-gbdt-evaluation-20261006/results/REPORT.json),
fields `local_protocol`, `official_components`, and `version`.

| Component | Saved value | Direction and scale |
|---|---:|---|
| Overall | 97.17188999355706 | Higher; 0-100 |
| Text edit distance | 0.03029528433169655 | Lower; 0-1 |
| Formula CDM | 0.9849933527825513 | Higher; 0-1 in report |
| Table TEDS | 0.9604586313558573 | Higher; 0-1 in report |
| Structure TEDS | 0.9738249860361294 | Higher; 0-1 in report |
| Reading-order edit distance | 0.12103739355037403 | Lower; 0-1 |

Overall averages text accuracy, formula CDM and Table TEDS after conversion to
0-100. Structure TEDS and reading order do not enter Overall. The report records
evaluator commit `147cd5ac9472002f5751221d390bf00abdbc0d2f` and configuration
SHA256 `56db52659910273376223a5646eabfaff21b25dc30f434e1c90aab1a233d335a`.

The original project-trained model used 30 PDFs / 30 groups / 90 rows. Its model
SHA256 is `78a8454b8d763f622e5d0ea9caa43695207a20c07461ef36e905753054a42518`.
The release preserves the model and its default disk configuration; the measured
experimental entry enables that same GBDT in memory with margin 0.0, fallback B,
candidate actions A/B, and at most one native recognition call per page.
Sources: [release guide](../releases/v4-gbdt-evaluation-20261006/README.md),
[model manifest](../releases/v4-gbdt-evaluation-20261006/model/manifest.json), and
[model origin](../releases/v4-gbdt-evaluation-20261006/provenance/MODEL_ORIGIN.json).

## Interpretation boundaries

The frozen public TeleOCR reference is **96.91**, giving a contextual difference
of +0.26188999355706244 points. It is not a paired control: identity of the
published dataset/settings with this local run is not established. This is not
a verified leaderboard rank or evidence of causal GBDT gain. Source:
`leaderboard_comparison` in the saved [report](../releases/v4-gbdt-evaluation-20261006/results/REPORT.json).
The reference is a saved 2026-10-05 snapshot, not a newly checked live leaderboard.

The current package also does not establish continuous optimal DPI. A bounded
candidate selector and one completed system score do not identify such an
optimum. Multiscale learned selection remains a proposed direction requiring a
separately agreed question, controls and protocol; no new experiment is approved.
No publication venue is selected by this state document.

The public release is a redacted export of the original delivery, with preserved
original/public hash mappings. It was not separately benchmarked. External model
assets, native environment, data and private bindings are prerequisites for any
future run. See [publication boundary](../releases/v4-gbdt-evaluation-20261006/PUBLICATION.md)
and [delivery manifest](../releases/v4-gbdt-evaluation-20261006/DELIVERY_MANIFEST.json).

## Earlier evidence remains distinct

The [historical whole-page record](../artifacts/whole-page-results.json) and
[root README](../README.md) retain local TeleOCR V2 control **97.435109** and
Hybrid V2 **97.060019**. That paired Hybrid V2 was 0.3751 Overall points below its
raw TeleOCR control. Those archived measurements are not one unified paired
experiment with V4 or the public TeleOCR 96.91 reference. V3.1/V3.2 remain without
a completed full-benchmark result in the published whole-page table.

The earlier native-table system achieved **98.75068667701048 full Table TEDS**
and **99.2913812392524 structure TEDS** under its older protocol. These are table
metrics, not Overall. The separate Training result adopted zero edits and matched
the fixed raw baseline. Negative and failed routes remain in the
[aggregate archive](../artifacts/results/README.md); do not relabel them as V4.
Some route guides contain dated pre-completion descriptions; use the completed
aggregate and dated root result records when determining current result status.

## Next scientific discussion

The immediate evidence gap is the relationship between historical local raw
TeleOCR 97.435109, V4 97.171890, and the public reference 96.91. Reconcile run
settings and baseline comparability before claiming an improvement or choosing
a subsequent method. Then determine what is needed for an official submission.
The [roadmap](ROADMAP.md) records these milestones without selecting a new
experiment. Preserve released source, model bytes, scores, failure handling and
existing paths. Use the [research map](README.md) to locate evidence.
