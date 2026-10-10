# Borderless Table Structuring Lab

Research on reliable document parsing: input resolution, specialist models, and
selective correction of text, formulas, and table structure.

Our goal is to improve end-to-end parsing on OmniDocBench and develop methods
whose gains hold beyond a single benchmark. We study when an existing parser
benefits from a different input view or specialist, and when an intervention
damages a correct prediction. The project includes successful and unsuccessful
historical experiments and the completed V6 study. The current V8 campaign studies
generator fine-tuning at a larger data scale; six V8 fits are complete and evaluation is running.

[Results](research/RESULTS.md) · [Methods](#methods) ·
[Reproduction](research/REPRODUCIBILITY.md) · [V5 study](versions/v5/README.md) ·
[V6 complete report](V6/README.md) · [V6 all statistics](V6/STATISTICS.md) ·
[Research status](research/CURRENT_STATE.md) · [Contributing](CONTRIBUTING.md)

## Research questions

- **Input resolution:** when does changing the input scale improve recognition,
  and can an adaptive policy outperform a strong fixed policy at comparable cost?
- **Specialist integration:** when does replacing a region with an expert
  prediction improve the complete document rather than only an isolated crop?
- **Selective correction:** how can a system recover errors while preserving
  correct content and controlling additional computation?

OmniDocBench is the primary benchmark. The performance target is an officially
recognized Overall score above the leaderboard leader. The accompanying research
goal is a generalizable contribution suitable for a CCF A venue; no paper or
official leaderboard result is claimed at this stage.

## V8: large table-specialist LoRA campaign

The [large LoRA progress report](artifacts/results/v7_finetuning/large-lora-001/REPORT.md)
records the frozen source-backed dataset and six completed training runs.
TRAIN contains 20,000 real tables from 11,217 documents. The study compares V8.1 ordinary
LoRA, V8.2 LoRA-GA and V8.3 LoRA-GA with cell-location supervision, each with seeds
0 and 1. The former V7.3.0/V7.3.1/V7.3.2 labels are aliases for these same six fits;
existing paths and immutable evidence retain their original names. Read the
[version mapping](artifacts/results/v7_finetuning/large-lora-001/VERSION_MAPPING.json),
[dataset card](artifacts/results/v7_finetuning/large-lora-001/DATASET_CARD.md)
and [methods](artifacts/results/v7_finetuning/large-lora-001/METHODS.md).
The [dataset figures](artifacts/results/v7_finetuning/large-lora-001/DATASET_FIGURES.md)
show measured composition, token lengths, image sizes and input exclusions.
All six DEV-selected models will receive the full 1,651-page benchmark, with the
three seed0 models evaluated before the three seed1 models and matched controls
shared across them. The [training analysis](artifacts/results/v7_finetuning/large-lora-001/TRAINING_ANALYSIS.md)
publishes all six training curves and 18 DEV endpoints. Independent confirmation
and full-benchmark quality results are pending.

## V6: complete results and statistics

**[Open V6](V6/README.md)** for the full report or
**[all requested V6 statistics](V6/STATISTICS.md)** for scores, category details,
failures, timing, shared/incremental costs, controller calls and page/region counts.
The root `V6/` directory is the canonical study entry. The historical executable
module name `versions.v5_native` remains unchanged for reproducibility.

Each output includes 1651 pages with zero inference failures. Six named
configurations share four CPU controllers and one D0 baseline. Overall is 0-100.

| V6 output | Overall | Difference from D0 |
|---|---:|---:|
| D0 | 97.574874 | 0.000000 |
| GBDT_V6.1.1 | 97.475604 | -0.099270 |
| GBDT_V6.1.2 | 97.582948 | +0.008074 |
| GBDT_V6.1.3 | 97.483678 | -0.091196 |
| MLP_V6.2.1 | 97.307734 | -0.267140 |
| MLP_V6.2.2 | 97.592369 | +0.017495 |
| MLP_V6.2.3 | 97.325229 | -0.249645 |

[Eight official metrics CSV](V6/statistics/OFFICIAL_METRICS.csv) ·
[All aggregate fields CSV](V6/statistics/ALL_AGGREGATE_FIELDS.csv) ·
[Activation counts CSV](V6/statistics/ACTIVATION_COUNTS.csv).
Table-only and combined arms regress; formula-only differences are small.
These are local paired results, not an official ranking.

The [historical archive](releases/v6-v7-progress-20261007/README.md) preserves source,
first-party controller weights and prior keep/replace evidence. That archived
controller study was formerly labeled V7; it is not the current fine-tuning model.
The
[baseline audit](artifacts/results/v7/BASELINE_COMPARABILITY.md) does not establish
comparability to the published TeleOCR row. [Current state](research/CURRENT_STATE.md)
separates completed evidence from paused work and unexecuted research proposals.

## Completed V5 study: input paths and regional rereading

All five arms completed **1,651 pages each**, with zero inference failures or
empty predictions, using the same pinned official evaluator and ground truth.
Overall, TEDS and CDM below use a 0-100 scale.

| Arm | Method | Overall ↑ | Table TEDS ↑ | Formula CDM ↑ |
|---|---|---:|---:|---:|
| V5.1 | Native image-to-PDF input | 97.5682 | 96.8432 | 98.5886 |
| V5.2 | Original RGB input | 98.1731 | 98.6611 | 98.3813 |
| V5.3 | Original RGB + table 1x | 98.2221 | 98.8184 | 98.3782 |
| V5.4 | Original RGB + formula 1.25x | 98.2506 | 98.6616 | 98.6514 |
| V5.5 | Original RGB + formula + guard + table | 98.3475 | 98.9095 | 98.6513 |

The original-image baseline V5.2 scores **98.1731**, +0.6049 points above V5.1.
The combined V5.5 pipeline reaches **98.3475**, +0.1744 above V5.2. This round
uses unchanged TeleOCR weights; regional rules build on the attributed
ki-OCR-v1 reference. It is a system comparison, not a newly trained model.

Independent native outputs differed across arms despite matched first-request
inputs and settings. The observed gains therefore do not isolate the causal
benefit of each intervention. These are local measurements, not an official
leaderboard result or a claim of statistical significance.

See the [complete study](versions/v5/README.md) for component scores, repair and
regression counts, cost, provenance and limitations, and
[AGGREGATE.json](artifacts/results/v5/AGGREGATE.json) for full-precision values.

## Historical results

Selected local results are shown below. Scores use a 0–100 scale. Rows belong to
different experimental groups and are not a single controlled ranking. The
[results record](research/RESULTS.md) includes controls, available component
metrics, sources, and evaluation conditions. A dash means the metric was not supplied.

| Experimental group | Method | Overall ↑ | Table TEDS ↑ | Formula CDM ↑ |
|---|---|---:|---:|---:|
| V0 / V1 | MinerU control | 93.1680 | 90.4345 | 95.5532 |
| V0 / V1 | V0: MinerU + NaviDC tables | 94.5993 | 94.7279 | 95.5532 |
| V0 / V1 | V1: MinerU + Paddle formulas | 93.2175 | 90.4345 | 95.7276 |
| V2 | TeleOCR control | 97.4351 | 96.6919 | 98.5996 |
| V2 | V2: TeleOCR + Paddle formulas | 97.0600 | 96.6919 | 97.4793 |
| V3.1 | Original formula-replacement pipeline + original image | 97.6973 | — | — |
| V3.2 OFF / ON | V3.2 OFF | 93.4951 | 86.9556 | 98.2000 |
| V3.2 OFF / ON | V3.2 ON | 93.7208 | 87.3918 | 98.4454 |
| V4 | GBDT input selection + TeleOCR | 97.1719 | 96.0459 | 98.4993 |

V3.1 and V3.2 values were supplied by the researcher on 2026-10-07; raw scoring
files and run revisions have not yet been added. V3.2 includes all 1,651 pages,
with failures retained, and improves Overall by **0.2257 points** over OFF.
V4 also includes all 1,651 pages, including seven timeout predictions kept empty.
V3.1's evaluation denominator was not supplied.

The paired V2 experiment decreased Overall by **0.3751 points**, showing that
specialist replacement can harm a strong parser. V3.2's scores on each arm's own
successful pages are reported separately, since those subsets may differ.
V5 records both repairs and regressions alongside intervention coverage.

An earlier native-table experiment reached **98.7507 Table TEDS** under a
different protocol. This is a component result, not a whole-page Overall score;
see the [table experiments](artifacts/results/README.md).

## Methods

| Version | Method studied | Source |
|---|---|---|
| V0 | Native table-region replacement with a specialist parser | [V0](versions/v0/README.md) |
| V1 | Geometry- and syntax-checked formula replacement on MinerU | [V1](versions/v1/README.md) |
| V2 | Formula expert integration on a TeleOCR backbone | [V2](versions/v2/README.md) |
| V3.1 | Layout/formula integration and original-image input comparisons | [V3.1](versions/v3_1/README.md) |
| V3.2 | OvisOCR2 text-region integration with adoption and fallback rules | [V3.2](versions/v3_2/README.md) |
| V4 | Learned selection between input actions before TeleOCR recognition | [V4 study](research/V4.md) |
| V5 | Original-image input, regional rereading and bounded anomaly recovery | [Completed study](versions/v5/README.md) |

The [earlier table method](LEGACY_NATIVE_TABLES.md) and
[table-structuring training experiments](training/README.md) remain available.
Method descriptions and measured results are linked separately so new evidence
can be added without rewriting earlier experiments.

## Reproducing the research

Start with the [reproduction guide](research/REPRODUCIBILITY.md) to select a
method and its corresponding model, input, and evaluation configuration.
The root package contains the earlier table implementation. Its local checks are:

```bash
git clone https://github.com/DearKarl/borderless-table-structuring-lab.git
cd borderless-table-structuring-lab
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[test]"
python -m pytest -q tests
```

These checks exercise implementation behavior; model inference requires the
method-specific dependencies and separately obtained weights and data.

## Repository structure

| Location | Research content |
|---|---|
| `research/` | Research questions, experiment summaries, protocols, and future work |
| `versions/` | V0–V3.2 and V5 implementations and method-specific instructions |
| `releases/` | Frozen source, models, and results for reproducible experiments |
| `artifacts/` | Machine-readable results, including negative outcomes |
| `btsl/`, `hybrid/`, `training/` | Table structuring, expert integration, and training code |
| `evaluation/` | Evaluation code and metric adapters |
| `tests/`, `examples/`, `scripts/` | Implementation checks and usage examples |

## Attribution

This project builds on OmniDocBench and the upstream parsers named in each
experiment. Cite their original work alongside any result from this repository.
Until a project paper is available, reference the repository commit, method, and
experiment record. See [third-party notices](notices/THIRD_PARTY.md) and
[contribution guidance](CONTRIBUTING.md#attribution-and-publication) for attribution
and licensing information.
