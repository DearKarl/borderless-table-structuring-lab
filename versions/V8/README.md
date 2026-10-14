# V8: table-specialist LoRA

[Versions](../README.md) · [Project](../../README.md)

| Version | Method |
| --- | --- |
| V8.1 | Ordinary rank-8 attention LoRA with a trainable visual merger |
| V8.2 | LoRA-GA initialization with the same adapter and merger training |
| V8.3 | LoRA-GA with an additional training-only cell-location head |
| V8.4 | Ordinary LoRA with complete TDATR-style adapted detail/fusion supervision |
| V8.5 | LoRA-GA with the same TDATR-style detail/fusion supervision |

## Full benchmark: all eight fitted runs

Every row retains all **1,651 pages**. Higher Table TEDS and Structure TEDS are
better; lower Table Edit Distance is better. TEDS, Structure TEDS and Overall
use **0–100** here; edit distance uses **0–1**. The full-precision CSV uses 0–1
for TEDS and preserves all eight official metrics. Official table scoring covers
458 pages and 665 table rows.

| Run | Table TEDS ↑ | Structure TEDS ↑ | Table Edit Distance ↓ | Overall ↑ | Technical failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| V8.1/seed0 | 90.298645 | 93.212727 | 0.085605 | 84.879353 | 161 |
| V8.1/seed1 | 93.295334 | 96.321028 | 0.060427 | 96.316806 | 0 |
| V8.2/seed0 | 90.944314 | 93.891982 | 0.081719 | 85.094576 | 161 |
| V8.2/seed1 | 92.309071 | 95.435168 | 0.067085 | 95.988052 | 0 |
| V8.3/seed0 | 92.471062 | 95.439443 | 0.065634 | 96.042049 | 0 |
| V8.3/seed1 | 92.669366 | 95.649207 | 0.064097 | 96.108150 | 0 |
| V8.4 | 91.700415 | 94.916362 | 0.070172 | 95.785166 | 0 |
| V8.5 | 91.684010 | 94.861013 | 0.070156 | 95.779698 | 0 |

Each seed is shown separately. V8.4 and V8.5 have one fit each; these rows are
neither seed averages nor a best-seed selection. The historical V8.1/seed0 and
V8.2/seed0 runs retain 161 technical failures each. V8.4 and V8.5 each have zero
technical failures and zero empty page predictions in the valid repaired run.

On this full benchmark, both new versions regress on all three main table metrics
against the matched native reference and all four older fits without technical
failures. Their confirmation TEDS gains over native do not carry over to the full
benchmark. This observed result does not establish a stable ranking across seeds.

### Matched references

| Run | Table TEDS ↑ | Structure TEDS ↑ | Table Edit Distance ↓ | Overall ↑ | Technical failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 96.962297 | 98.295915 | 0.034118 | 97.539151 | 0 |
| legacy_simple_rule | 96.604146 | 98.164562 | 0.037184 | 97.419767 | 0 |

The base and legacy rule share the preserved native benchmark path. The complete
[metrics CSV](results/tdatr-v8-45-001/METRICS.csv),
[category breakdowns and contrasts](results/tdatr-v8-45-001/RESULTS.json), and
[paired outcomes](results/tdatr-v8-45-001/PAIRED_OUTCOMES.json) retain all ten
policies. [The report](results/tdatr-v8-45-001/REPORT.md) includes confirmation,
DEV selection, analysis, costs and figures.

## Training, comparability and execution

All fitted methods use the frozen 20,000 real tables from 11,217 TRAIN documents,
256 DEV sources and 512 confirmation sources. Inputs use predicted native crops
at PDF 200 DPI. DEV alone selects checkpoints. Older complete fits have 60,000
exposures and 3,750 updates; each new complete fit has 120,000 exposures and
7,500 updates. Both new nominees are fusion epoch 2, selected at 100,000 exposures
and 6,250 updates. Older selected-checkpoint budgets vary and are reported in
the full report. These comparisons do not isolate supervision from training amount.

V8.3's auxiliary head is training-only. The new localization branch replays the
model's own OTSL in an extra forward, without geometry labels or rewriting table
text/HTML. This is an adaptation of TDATR to the native backbone, not exact
paper reproduction. The Transformers backend does not apply presence/frequency
penalties; historical vLLM scores are not interchangeable. The confirmation and
benchmark sets were previously exposed, and one new fit per method does not
establish seed stability or statistical superiority.

The first V8.4/V8.5 benchmark attempt failed before any model call because of
incompatible wrapper keyword names. Its invalid scores are excluded. The repaired
attempt preserved the frozen checkpoints, native inputs and page list; only
previously unexecuted table inference was performed. Failed-attempt costs remain
in the report. Both valid attempts completed full inference and official scoring.

## Scientific outputs and source

- V8.4/V8.5: [dataset and protocol](results/tdatr-v8-45-001/DATASET_AND_PROTOCOL.md), [complete report](results/tdatr-v8-45-001/REPORT.md), [numeric training histories](results/tdatr-v8-45-001/training_curves/MANIFEST.json), [model-package identities](results/tdatr-v8-45-001/MODEL_PACKAGES.json), and [source](source/tdatr_adaptation/README.md).
- V8.1–V8.3: [original method summary](results/large-lora-001/benchmark/metrics/METRICS.csv), [confirmation](results/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json), [training records](results/large-lora-001/training/metrics/training_scalars.csv), [DEV checkpoints](results/large-lora-001/training/metrics/dev_epochs.csv), [figures](results/large-lora-001/FIGURES.md), [dataset](results/large-lora-001/DATASET_CARD.md), [methods](results/large-lora-001/METHODS.md), [model packages](results/large-lora-001/MODEL_PACKAGES.json), and [source](source/generator_pilot/LARGE_CAMPAIGN_RUNTIME.md).

These are local official-protocol measurements, not an accepted leaderboard entry.
