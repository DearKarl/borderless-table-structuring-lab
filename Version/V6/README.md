# V6: direct regional scale controllers

[Version index](../README.md) · [Project](../../README.md)

## Method

Four CPU-trained controllers (GBDT/MLP × table/formula) supply six configurations: V6.1.1/V6.1.2/V6.1.3 use GBDT for tables/formulas/both; V6.2.1/V6.2.2/V6.2.3 use MLP for the same tasks. A shared D0 native parse supplies each page, and combined policies reuse component decisions and candidates. Seven outputs are therefore paired shared-computation results, not seven independent fresh runs. The executable package remains `versions.v5_native` for provenance.

## All eight official fields

All seven outputs retain 1,651 pages each, with zero inference failures, empty predictions or replayed pages. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 97.574874 | 0.027098 | 98.602591 | 0.049005 | 96.831865 | 98.167331 | 0.035843 | 0.118297 |
| GBDT_V6.1.1 | 97.475604 | 0.027098 | 98.602591 | 0.049005 | 96.534055 | 97.772406 | 0.038683 | 0.118297 |
| GBDT_V6.1.2 | 97.582948 | 0.027082 | 98.625208 | 0.049068 | 96.831865 | 98.167331 | 0.035843 | 0.118370 |
| GBDT_V6.1.3 | 97.483678 | 0.027082 | 98.625208 | 0.049057 | 96.534055 | 97.772406 | 0.038683 | 0.118370 |
| MLP_V6.2.1 | 97.307734 | 0.027098 | 98.602591 | 0.049005 | 96.030446 | 97.419141 | 0.040142 | 0.118297 |
| MLP_V6.2.2 | 97.592369 | 0.026989 | 98.644101 | 0.048322 | 96.831865 | 98.167331 | 0.035843 | 0.118214 |
| MLP_V6.2.3 | 97.325229 | 0.026989 | 98.644101 | 0.048322 | 96.030446 | 97.419141 | 0.040142 | 0.118214 |

## Interpretation and cost

Both table-only policies and both combined policies regress against shared D0. Formula-only Overall changes are small: +0.008074 for GBDT and +0.017495 for MLP. These are descriptive local differences, not an official rank. GBDT and MLP differ in capacity, optimization and regularization; the comparison cannot isolate continuity as the cause. A matched fixed-scale end-to-end counterfactual is still needed to isolate learned scale selection from rereading itself.

The source-page corpus contains 276 regions from 242 documents, split 180/48/48. Total allocated GPU time is 5.792213 hours, including 4.168781 for shared benchmark inference. Shared costs are counted once; all owned compute exited.

## Implementation and complete evidence

- [Full V6 method and report](report/README.md), [all statistics and missing-data notes](report/STATISTICS.md), and [eight-field CSV](report/statistics/OFFICIAL_METRICS.csv).
- [Aggregate with all category breakdowns](results/AGGREGATE.json), [paired outcomes](results/PAIRED_QUALITY.json), [activation](results/ACTIVATION.json), [costs](results/COSTS.json) and [external validation](results/EXTERNAL_ACTOR_QUALITY.json).
- [Implementation](source/v5_native/README.md) and [frozen model package](archive/v6-v7-progress-20261007/models/v6/PACKAGE_MANIFEST.json).

These controls are not interchangeable with historical V2, V5 original-RGB or V8 Transformers controls; consult the [baseline comparability audit](../V7/results/legacy_selector/BASELINE_COMPARABILITY.md).
