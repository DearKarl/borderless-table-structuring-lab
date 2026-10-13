# V8: large table-specialist LoRA campaign

[Version index](../README.md) · [Project](../../README.md)

**Latest measured version.** Six fits compare ordinary LoRA (V8.1), LoRA-GA initialization (V8.2), and LoRA-GA plus a training-only cell-location head (V8.3), with seeds 0 and 1. Old V7.3.0/V7.3.1/V7.3.2 labels refer to these same six runs.

## Method

The frozen corpus has 20,000 real tables from 11,217 documents, 256 DEV sources and 512 independent confirmation sources. Inputs use predicted native table crops at PDF 200 DPI. All arms train rank-8 attention adapters and the visual merger; other base parameters remain frozen. V8.2/V8.3 share TRAIN-only gradient initialization, and V8.3 adds cell-box supervision used only during training. Each fit completes 3,750 updates and 60,000 exposures. DEV alone selects checkpoints; V8.3/seed0 is the frozen nominee.

## Full benchmark: all eight fields

All policies retain the same 1,651-page universe; table metrics cover 458 scored pages and the paired table supplement contains 665 uniquely identified official table rows. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matched native baseline | 97.539151 | 0.029707 | 98.625832 | 0.049639 | 96.962297 | 98.295915 | 0.034118 | 0.118769 |
| Legacy simple rule | 97.419767 | 0.029707 | 98.625832 | 0.049639 | 96.604146 | 98.164562 | 0.037184 | 0.118769 |
| V8.1/seed0 * | 84.879353 | 0.129939 | 77.333336 | 0.255067 | 90.298645 | 93.212727 | 0.085605 | 0.211800 |
| V8.1/seed1 | 96.316806 | 0.029707 | 98.625832 | 0.049639 | 93.295334 | 96.321028 | 0.060427 | 0.118769 |
| V8.2/seed0 * | 85.094576 | 0.129939 | 77.333336 | 0.255067 | 90.944314 | 93.891982 | 0.081719 | 0.211855 |
| V8.2/seed1 | 95.988052 | 0.029707 | 98.625832 | 0.049639 | 92.309071 | 95.435168 | 0.067085 | 0.118771 |
| V8.3/seed0 | 96.042049 | 0.029707 | 98.625832 | 0.049639 | 92.471062 | 95.439443 | 0.065634 | 0.118771 |
| V8.3/seed1 | 96.108150 | 0.029707 | 98.625832 | 0.049639 | 92.669366 | 95.649207 | 0.064097 | 0.118771 |

### Execution qualifications

* V8.1/seed0 and V8.2/seed0 each retain **161 empty technical-failure outcomes** caused by a PDF assembly defect. Their full-denominator scores, affected contrasts and two-seed summaries describe execution outcomes, not clean training-effect estimates. The original attempts and costs are retained; the versioned correction processed only previously unattempted pages. See the [correction record](results/large-lora-001/EVALUATION_PIPELINE_CORRECTION.json) and [qualified aggregate](results/large-lora-001/benchmark/metrics/AGGREGATE.json).

The other four model runs are unaffected by that defect. All four still trail both matched controls in full-benchmark Table TEDS; the nominee's regression cannot be assigned to the two separate failures. No model is reselected from these results.

## Confirmation and interpretation

On the 512-source confirmation set, nominee TEDS is 0.927999 versus native 0.885459 and legacy 0.899376. Its native contrast is **+4.2541 percentage points**, nominal 97.5% paired percentile interval [2.9812, 5.4708]; legacy contrast is **+2.8623 points**, interval [1.6641, 3.9864]. All 20 missing native inputs remain in the denominator. Bonferroni allocation targets 95% coverage for two fixed primary contrasts; bootstrap coverage is approximate. Four secondary initializer/head contrasts have intervals crossing zero, which does not establish equivalence.

On the full benchmark the same nominee loses **4.4912 Table TEDS points** versus native, with Overall 96.042049 versus 97.539151. The evidence supports cohort-specific improvement but not replacement of the general native parser. Domain, representation and input explanations remain hypotheses. This is a new matched Transformers comparison, not equivalent to historical vLLM scores or an official leaderboard submission.

## Complete results, code and artifacts

- [Full scientific report](results/large-lora-001/REPORT.md): every fit/control, selection, interpretation and limitation.
- [All official metrics CSV](results/large-lora-001/benchmark/metrics/METRICS.csv), [category breakdowns](results/large-lora-001/benchmark/metrics/AGGREGATE.json), and [16 paired benchmark comparisons](results/large-lora-001/benchmark/metrics/PAIRED_OUTCOMES.json).
- [All confirmation endpoints, intervals, gains/harms and strata](results/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json).
- [Dataset design](results/large-lora-001/DATASET_CARD.md), [methods](results/large-lora-001/METHODS.md), [training analysis](results/large-lora-001/TRAINING_ANALYSIS.md), and [19-figure index](results/large-lora-001/FIGURES.md).
- [Published runtime/source guide](source/generator_pilot/LARGE_CAMPAIGN_RUNTIME.md), [six local model-package metadata records](results/large-lora-001/MODEL_PACKAGES.json), and [compute closure](results/large-lora-001/COMPUTE_CLOSURE.json).
- [Lead's table-metric summary (download HTML)](reports/V8_BENCHMARK_RESULTS_AND_NEXT_STEPS_20261011.html) and [full-precision JSON](reports/V8_BENCHMARK_RESULTS_AND_NEXT_STEPS_20261011.json).

Total cost was 263.0598 allocated GPU-hours across 363 closed allocations, including failures and waiting, with shared work counted once. Allocated device time is not elapsed wall time; the detailed report and timing ledger retain that distinction. No weights or raw benchmark payloads are published by this guide.

## Reserved V8.4 and V8.5

V8.4 reserves ordinary LoRA plus full TDATR-style adapted supervision; V8.5 reserves LoRA-GA plus the same intervention, using the same frozen TRAIN/DEV. Both are **reserved/planned, unimplemented and unstarted**. First analyze TRAIN/DEV errors and freeze all components and the training/evaluation protocol. This is an adaptation, not strict paper replication; already exposed test sets cannot become untouched again. See the [reservation record](results/large-lora-001/NEXT_VERSIONS.md).
