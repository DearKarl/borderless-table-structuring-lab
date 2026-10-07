# Current research state

Evidence updated 2026-10-07 after completion of V5.1-V5.5. This document indexes
completed local experiments and separates observations from future proposals.

## Project objective

Achieve an officially recognized OmniDocBench Overall score above the leaderboard
leader, with TeleOCR as the benchmarked reference backbone. This is the primary performance objective clarified by the
researcher. V0–V4 are historical attempts. Input-scale selection, a frozen TeleOCR
backbone, one recognition call per page, and GBDT are V4 design choices rather
than permanent requirements for subsequent research. No official leaderboard win
has been established by the saved results below.

On 2026-10-07, the researcher additionally requested a research contribution
aimed at a CCF A venue. The leaderboard objective remains primary; a high score
alone does not establish a publishable contribution. The [V5 proposal](V5_PROPOSAL.md)
sets out baseline reconciliation, novelty assessment and staged validation.
Eight A100 80GB PCIe GPUs are available. On 2026-10-07 the researcher approved
all five comparisons as V5.1-V5.5, requested parallel full-set execution, and
authorized execution. All five arms and their scoring are now complete. The
[V5 proposal](V5_PROPOSAL.md) retains the original design rationale; completed
measurements and deviations are recorded in the V5 report below.

## Completed V5 evidence

All five independent arms completed inference and official-protocol local scoring
on 2026-10-07. Each contains 1,651 pages, with zero inference failures and zero
empty predictions: 8,255 arm-page predictions in total. The same pinned evaluator
and ground truth were used throughout this round.

| Arm | Method | Overall ↑ |
|---|---|---:|
| V5.1 | Native image-to-PDF input | 97.568231 |
| V5.2 | Original RGB input | 98.173146 |
| V5.3 | Original RGB + table 1x | 98.222117 |
| V5.4 | Original RGB + formula 1.25x | 98.250602 |
| V5.5 | Original RGB + formula + guard + table | 98.347514 |

V5.2 minus V5.1 is +0.604916 points; V5.5 minus V5.2 is +0.174368 points.
The highest observed system score is V5.5, 98.347514. No official rank or
statistical significance is established. Five-arm inference used 9.758604
allocated GPU-hours; including preparation and stopped startup attempts, the
total was 10.231368 GPU-hours. Two-GPU arms cannot be compared by wall time alone.

Sources: [V5 methods and full report](../versions/v5/README.md),
[aggregate values](../artifacts/results/v5/AGGREGATE.json), and
[frozen protocol](../versions/v5/protocol.json).

Independent original-image native outputs were not identical across arms.
Compared with V5.2, 313 V5.3 pages, 297 V5.4 pages and 280 V5.5 pages had different
native Markdown. Matching first-request pixel/prompt/settings records does not
explain the cause. End-to-end differences therefore do not isolate regional
intervention effects. Accepted syntax-valid crops are not automatically correct.
The requested upstream model revision was not verified; actual inference file
hashes were recorded and matched the existing reference pins. See the full report
for the precise revision limitation and component-specific coverage.

## Completed V4 evidence

The completed local **V4 GBDT input selection + TeleOCR** run used the official
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
separately agreed question, controls and protocol. The newly approved V5.1-V5.5
comparison does not establish a new learned-selector experiment.
No publication venue is selected by this state document.

The [V4 study](V4.md) describes the method and measured policy. The frozen source,
model and results identify the historical experiment; reproducing it requires
the documented model assets, environment, inputs and evaluation configuration.
See the [reproduction guide](REPRODUCIBILITY.md).

## Earlier evidence remains distinct

The [historical whole-page record](../artifacts/whole-page-results.json) and
[root README](../README.md) retain local TeleOCR V2 control **97.435109** and
Hybrid V2 **97.060019**. That paired Hybrid V2 was 0.3751 Overall points below its
raw TeleOCR control. Those archived measurements are not one unified paired
experiment with V4 or the public TeleOCR 96.91 reference. The older whole-page
JSON remains a historical snapshot. Newly supplied V3 results are recorded in
the [current experiment summary](RESULTS.md) and a separate
[machine-readable supplement](../artifacts/results/v3-reported-20261007.json).

## Researcher supplied V3 results

On 2026-10-07, the researcher reported V3.1 Overall **97.6973** for the original
formula-replacement pipeline with original-image input. Other rows in the
provided screenshot do not belong to this project and are excluded. V3.1's
component metrics, denominator and paired baseline were not supplied.

V3.2 evaluated **1,651 pages per arm**, using the same official evaluator and GT
according to the researcher. OFF scored **93.4951**, ON **93.7208**: **+0.2257**
Overall points, with all failures retained. Table TEDS was **86.9556 / 87.3918**
and formula CDM **98.2000 / 98.4454**. Text quality reportedly regressed slightly;
its numeric metric is unknown. Each arm's own successful-page scores were
**97.6657 / 97.5607**, but the subsets are not confirmed identical and cannot be
used as a paired comparison. Raw evaluator files and scored revisions remain
to be added. These records are supplied observations, not a fresh rescore.

The earlier native-table system achieved **98.75068667701048 full Table TEDS**
and **99.2913812392524 structure TEDS** under its older protocol. These are table
metrics, not Overall. The separate Training result adopted zero edits and matched
the fixed raw baseline. Negative and failed routes remain in the
[aggregate archive](../artifacts/results/README.md); do not relabel them as V4.
Some route guides contain dated pre-completion descriptions; use the completed
aggregate and dated root result records when determining current result status.

## Next scientific discussion

V5.5 is the current local submission candidate; V5.2 is its strong same-round
baseline. First reuse saved native and intermediate predictions to determine
which stages repair or damage the same pages. Independent native variation must
be separated from patch effects before claiming a causal mechanism. This is a
proposed analysis, not a completed counterfactual experiment.

A research contribution should address measured residual errors and demonstrate
benefit beyond the benchmark used for development. Portable fresh-environment
reproduction and the unresolved model-revision identity remain submission
preparation tasks. No new architecture, training run, sweep or official submission
is implied by publishing these results. See the [roadmap](ROADMAP.md).
