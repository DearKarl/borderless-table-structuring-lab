# Experimental results index

The latest completed study is [V8](versions/V8.md). Results are organized by version, with controls, complete available component scores, missing fields, failure accounting, cost and interpretation in each introduction and its evidence links. No rescoring or cross-version leaderboard is created by this index.

| Version | Complete introduction and evidence | Metric availability |
| --- | --- | --- |
| V0 | [Introduction](versions/V0.md) | Available fields; missing metrics identified explicitly |
| V1 | [Introduction](versions/V1.md) | Available fields; missing metrics identified explicitly |
| V2 | [Introduction](versions/V2.md) | Available fields; missing metrics identified explicitly |
| V3 | [Introduction](versions/V3.md) | Available fields; missing metrics identified explicitly |
| V4 | [Introduction](versions/V4.md) | Available fields; missing metrics identified explicitly |
| V5 | [Introduction](versions/V5.md) | Available fields; missing metrics identified explicitly |
| V6 | [Introduction](versions/V6.md) | Full official eight fields |
| V7 | [Introduction](versions/V7.md) | Table/cell and diagnostic metrics; no generator whole-page benchmark |
| V8 | [Introduction](versions/V8.md) | Full official eight fields |

## Metric conventions

Overall = 100 × ((1 − Text ED) + Formula CDM + Table TEDS) / 3 when component inputs use 0–1. Structure TEDS, Formula ED, Table ED and Reading-order ED do not enter that formula. Individual historical documents use either 0–1 or 0–100 CDM/TEDS; every new version table states its units. Missing values remain missing and are never inferred from rounded Overall.

Use matched input/model/evaluator/failure rules for comparisons. A full page universe is distinct from the official component denominator. Local official-protocol evaluation is not an accepted leaderboard entry. V3 measurements are researcher supplied, and V7 pilot endpoints are not whole-page Overall.

## Current results

[V8 complete report](../artifacts/results/v7_finetuning/large-lora-001/REPORT.md), [all official fields](../artifacts/results/v7_finetuning/large-lora-001/benchmark/metrics/METRICS.csv), [paired outcomes](../artifacts/results/v7_finetuning/large-lora-001/benchmark/metrics/PAIRED_OUTCOMES.json), [confirmation](../artifacts/results/v7_finetuning/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json), and [figures](../artifacts/results/v7_finetuning/large-lora-001/FIGURES.md).

## Historical entry compatibility

These anchors preserve links from the original implementation guides:

<a id="v0-through-v2"></a>
[V0](versions/V0.md), [V1](versions/V1.md) and [V2](versions/V2.md): whole-page controls and expert integration.

<a id="v31-original-image-formula-replacement"></a>
[V3.1 original-image formula replacement](versions/V3.md#v31-original-image-formula-replacement).

<a id="v32-full-set-off-and-on-comparison"></a>
[V3.2 full-set and successful-subset results](versions/V3.md#v32-text-region-replacement-with-ovisocr2).

<a id="v4-learned-input-selection"></a>
[V4 learned input selection](versions/V4.md).

<a id="v5-completed-five-arm-comparison-2026-10-07"></a>
[V5 complete five-arm results](versions/V5.md).

<a id="earlier-table-experiments"></a>
[Earlier table-only experiments](versions/V0.md#earlier-table-only-studies) and their [original aggregate archive](../artifacts/results/README.md) remain separate from whole-page scores.
