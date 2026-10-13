# Experimental results index

The latest completed study is [V8](../Version/V8/README.md). Results are organized by version, with controls, complete available component scores, missing fields, failure accounting, cost and interpretation in each introduction and its evidence links. No rescoring or cross-version leaderboard is created by this index.

| Version | Complete introduction and evidence | Metric availability |
| --- | --- | --- |
| V0 | [Introduction](../Version/V0/README.md) | Available fields; missing metrics identified explicitly |
| V1 | [Introduction](../Version/V1/README.md) | Available fields; missing metrics identified explicitly |
| V2 | [Introduction](../Version/V2/README.md) | Available fields; missing metrics identified explicitly |
| V3 | [Introduction](../Version/V3/README.md) | Available fields; missing metrics identified explicitly |
| V4 | [Introduction](../Version/V4/README.md) | Available fields; missing metrics identified explicitly |
| V5 | [Introduction](../Version/V5/README.md) | Available fields; missing metrics identified explicitly |
| V6 | [Introduction](../Version/V6/README.md) | Full official eight fields |
| V7 | [Introduction](../Version/V7/README.md) | Table/cell and diagnostic metrics; no generator whole-page benchmark |
| V8 | [Introduction](../Version/V8/README.md) | Full official eight fields |

## Metric conventions

Overall = 100 × ((1 − Text ED) + Formula CDM + Table TEDS) / 3 when component inputs use 0–1. Structure TEDS, Formula ED, Table ED and Reading-order ED do not enter that formula. Individual historical documents use either 0–1 or 0–100 CDM/TEDS; every new version table states its units. Missing values remain missing and are never inferred from rounded Overall.

Use matched input/model/evaluator/failure rules for comparisons. A full page universe is distinct from the official component denominator. Local official-protocol evaluation is not an accepted leaderboard entry. V3 measurements are researcher supplied, and V7 pilot endpoints are not whole-page Overall.

## Current results

[V8 complete report](../Version/V8/results/large-lora-001/REPORT.md), [all official fields](../Version/V8/results/large-lora-001/benchmark/metrics/METRICS.csv), [paired outcomes](../Version/V8/results/large-lora-001/benchmark/metrics/PAIRED_OUTCOMES.json), [confirmation](../Version/V8/results/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json), and [figures](../Version/V8/results/large-lora-001/FIGURES.md).

## Historical entry compatibility

These anchors preserve links from the original implementation guides:

<a id="v0-through-v2"></a>
[V0](../Version/V0/README.md), [V1](../Version/V1/README.md) and [V2](../Version/V2/README.md): whole-page controls and expert integration.

<a id="v31-original-image-formula-replacement"></a>
[V3.1 original-image formula replacement](../Version/V3/README.md#v31-original-image-formula-replacement).

<a id="v32-full-set-off-and-on-comparison"></a>
[V3.2 full-set and successful-subset results](../Version/V3/README.md#v32-text-region-replacement-with-ovisocr2).

<a id="v4-learned-input-selection"></a>
[V4 learned input selection](../Version/V4/README.md).

<a id="v5-completed-five-arm-comparison-2026-10-07"></a>
[V5 complete five-arm results](../Version/V5/README.md).

<a id="earlier-table-experiments"></a>
[Earlier table-only experiments](../Version/V0/README.md#earlier-table-only-studies) and their [original aggregate archive](../artifacts/results/README.md) remain separate from whole-page scores.
