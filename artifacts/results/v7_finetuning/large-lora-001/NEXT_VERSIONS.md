# V8.4 and V8.5: reserved versions

Recorded October 12, 2026. **Both versions are reserved/planned. Neither is implemented or trained, and neither has measured results.** This reservation accompanies publication of the completed V8.1-V8.3 study; it does not launch another experiment.

| Version | Initialization | Intended shared supervision | Status |
| --- | --- | --- | --- |
| V8.4 | Ordinary LoRA | Full TDATR-style adapted training and supervision | Reserved / planned; not implemented or started |
| V8.5 | LoRA-GA | The same TDATR-style adapted training and supervision | Reserved / planned; not implemented or started |

The intended comparison keeps the same frozen 20,000-table TRAIN set from 11,217 documents and the same 256-source DEV split. Seeds 0 and 1 are proposed for both versions; the detailed training and evaluation protocol still needs to be frozen. Additional supervision must be derived from existing verified annotations, with missing labels and coverage recorded explicitly. The version names do not authorize silently replacing examples or changing the cohort.

“Full TDATR-style” names the intended scope, not a completed component specification. The implementation mapping must identify every intended task, localization and alignment component, loss, training stage, inference behavior and deviation before execution. Work with the current backbone and dataset is an adaptation; it must not be presented as a strict replication of the paper.

## Work before any new fit

1. Analyze saved TRAIN/DEV errors: native detection and crop coverage, text, row/column relations, spans, empty cells and decoding. Investigate the regression against the matched native baseline. Do not choose design details, loss weights or checkpoints using the exposed benchmark.
2. Freeze component coverage and supervision definitions, including unavailable labels, loss normalization and how the proposed tasks interact.
3. Freeze matched schedules, initialization, seeds, DEV checkpoint selection, training exposures, compute accounting and evaluation rules for V8.4/V8.5. A multi-stage schedule may use different compute or exposures from V8.1-V8.3 even with an unchanged dataset; report those differences.
4. Define the primary endpoint and success criteria prospectively. Table TEDS is proposed as primary, with Structure TEDS, Table Edit Distance, failures and cost retained. Keep the stronger matched native control, the legacy control and all completed V8.1-V8.3 results in view.
5. Keep existing confirmation and full-benchmark exposure explicit. Those sets cannot be described as untouched tests for the next design. Freeze any future evaluation plan before implementation or execution.

Completed controls are reused as recorded. This reservation does not refit V8.1/V8.2, remove their seed0 technical failures, repair exposed evaluations or select a replacement model from benchmark scores. Physical GPU0 remains excluded from any subsequently authorized execution.

## Completed evidence

The [complete V8 report](REPORT.md) retains all six models, both controls, all eight official metrics, confirmation analysis, costs and limitations. The [Research Lead summary](../../../../research/reports/V8_BENCHMARK_RESULTS_AND_NEXT_STEPS_20261011.html) highlights Table TEDS, Structure TEDS and Table Edit Distance; its [JSON](../../../../research/reports/V8_BENCHMARK_RESULTS_AND_NEXT_STEPS_20261011.json) preserves full precision. Download the HTML to view its formatted tables.

The frozen V8.3/seed0 nominee improved confirmation TEDS by 4.2541 percentage points over matched native but lost 4.4912 points on full-benchmark table TEDS. All four technically clean model runs remained below both matched controls on that benchmark metric. Those findings motivate diagnosis; they do not establish the cause of the regression or predict gains from the reserved versions.
