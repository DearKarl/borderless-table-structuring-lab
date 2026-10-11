# Completed V8 result figures

Eight measured final figures; all PNGs visually inspected, all PNG/SVG hashes and SVG XML checked. No inference, scoring or bootstrap analysis was repeated to render them.

| Figure | Raster | Vector |
| --- | --- | --- |
| official metrics | [PNG](benchmark/figures/official_metrics.png) | [SVG](benchmark/figures/official_metrics.svg) |
| matched base differences | [PNG](benchmark/figures/matched_base_differences.png) | [SVG](benchmark/figures/matched_base_differences.svg) |
| table seed variability | [PNG](benchmark/figures/table_seed_variability.png) | [SVG](benchmark/figures/table_seed_variability.svg) |
| allocation costs | [PNG](benchmark/figures/allocation_costs.png) | [SVG](benchmark/figures/allocation_costs.svg) |
| confirmation primary | [PNG](confirmation/figures/confirmation_primary.png) | [SVG](confirmation/figures/confirmation_primary.svg) |
| confirmation all models | [PNG](confirmation/figures/confirmation_all_models.png) | [SVG](confirmation/figures/confirmation_all_models.svg) |
| confirmation gains harms | [PNG](confirmation/figures/confirmation_gains_harms.png) | [SVG](confirmation/figures/confirmation_gains_harms.svg) |
| confirmation detail | [PNG](confirmation/figures/confirmation_detail.png) | [SVG](confirmation/figures/confirmation_detail.svg) |

The confirmation detail plot uses an explicitly expanded vertical axis; its method intervals are descriptive secondary results. The full benchmark marks both PDF-failure-affected seed0 runs and their two-seed summaries. Two-seed ranges are not confidence intervals.

The existing [four dataset figures](DATASET_FIGURES.md) and [seven training figures](TRAINING_ANALYSIS.md) remain available, for 19 campaign figures in total. Machine-readable inputs are in [benchmark metrics](benchmark/metrics), [confirmation metrics](confirmation/metrics), and their JSON/CSV files linked in the [report](REPORT.md).
