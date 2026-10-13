# V6 statistics coverage

This page indexes all requested aggregate statistics. Source JSON files are
byte-identical copies of the previously published measurements. CSV files are
navigation/export views, not new scoring. [Complete report](README.md).

| Required statistics | Evidence | Interpretation |
|---|---|---|
| All eight official fields, category breakdowns, per-field directions and units | [AGGREGATE.json](statistics/AGGREGATE.json), [metrics CSV](statistics/OFFICIAL_METRICS.csv) | Overall is 0-100; other seven fields are 0-1. Includes full category detail, deltas and evaluator diagnostics. |
| Every failed/empty/timeout output and missing field | [AGGREGATE.json](statistics/AGGREGATE.json), [ACTIVATION.json](statistics/ACTIVATION.json), [COSTS.json](statistics/COSTS.json) | Full benchmark inference failures are zero. One offline scale view exceeded context; retained in failure and cost accounting, without replay. Missing timers remain null. |
| Training, packaging, scorer durations; shared and incremental computation | [COSTS.json](statistics/COSTS.json), [ACTIVATION.json](statistics/ACTIVATION.json) | Four CPU controllers. Shared benchmark 4.168781 GPU-hours, total task 5.792213. Stage sums overlap; standalone estimates are labeled and are not independent measured runs. |
| Controller API calls, batches and predicted regions | [ACTIVATION.json](statistics/ACTIVATION.json) | Each family physically predicts 3041 rows in 3041 single-row calls; combined outputs reuse these calls. Do not sum attribution across all six outputs. |
| Nominal scale, realized requested pixels, actual model inputs and final outputs; region and unique-page counts | [Activation counts CSV](statistics/ACTIVATION_COUNTS.csv), [ACTIVATION.json](statistics/ACTIVATION.json) | Includes changed/unchanged/unknown tensors and applied edits. Tensor comparison is against the same regional crop at 1x, not an assumed native-page tensor. |
| Improved, worsened, tied and unmatchable results; full eligible and applied-edit populations | [PAIRED_QUALITY.json](statistics/PAIRED_QUALITY.json) | All scored samples, page outcomes, eligible regions and applied edits retained. Unique-page sets can overlap across outcomes. Output changes do not establish quality improvement. |
| External validation/test quality, fixed controls, uncertainty and scale regression | [EXTERNAL_ACTOR_QUALITY.json](statistics/EXTERNAL_ACTOR_QUALITY.json), [SCALE_REGRESSION.json](statistics/SCALE_REGRESSION.json) | Includes negative results; small exploratory paired intervals are not proof of benchmark significance. |
| Model/source/runtime identity | [RUNTIME_PROVENANCE.json](statistics/RUNTIME_PROVENANCE.json), [model package](../releases/v6-v7-progress-20261007/models/v6/PACKAGE_MANIFEST.json) | Six configurations share four controllers; upstream requested model revision remains unverified. |
| Explicit missingness and source hashes | [COVERAGE.json](statistics/COVERAGE.json) | Null values remain unknown. Raw prediction/GT bodies and private deployment details are not published. |

## Complete export

[ALL_AGGREGATE_FIELDS.csv](statistics/ALL_AGGREGATE_FIELDS.csv) contains every leaf
from all seven JSON reports, including category fields, failures, costs, call
counts and explanations. Use its `report` and JSON-pointer columns to recover
the source location. `value_json` retains types, nulls and empty collections.

## Fixed counting rules and limits

- Every output retains 1651 pages. Six adaptive outputs total 9906 arm-pages;
  including D0 gives 11557 arm-pages, from 1651 distinct pages.
- PDF rendering is fixed at 200 DPI; the number of pages with changed DPI is 0.
  Regional pixel resizing is reported separately.
- Shared native parsing and regional OCR reuse are charged once. Per-output
  standalone estimates cannot be added to the shared measured total.
- Both table-only and both combined configurations regress. Formula-only gains
  are small; this does not establish statistical significance or rank.
- Detection misses are not established by the eligible-region counts. Missing
  input comparisons and unmatchable quality outcomes must not be called successes.
- The preserved ledger's hash is in the [original publication manifest](../releases/v6-v7-progress-20261007/PUBLICATION_MANIFEST.json).
  The raw ledger is private; all requested aggregate page/region counts are public.
- No official leaderboard submission or ranking is claimed. Local baseline
  comparability to the published TeleOCR row remains unresolved.
