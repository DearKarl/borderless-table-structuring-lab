# V8: table-specialist LoRA

[Versions](../README.md) · [Project](../../README.md)

| Version | Method |
| --- | --- |
| V8.1 | Ordinary rank-8 attention LoRA with a trainable visual merger |
| V8.2 | LoRA-GA initialization with the same adapter and merger training |
| V8.3 | LoRA-GA with an additional training-only cell-location head |
| V8.4 | Ordinary LoRA with TDATR-style detail and fusion supervision |
| V8.5 | LoRA-GA with the same TDATR-style detail and fusion supervision |

## Training and inputs

The corpus contains 20,000 real tables from 11,217 TRAIN documents, with 256 DEV
sources and 512 confirmation sources. Inputs use predicted native crops at PDF
200 DPI. V8.1–V8.3 complete 3,750 optimizer updates and 60,000 exposures per fit.
V8.4/V8.5 complete three detail passes and three fusion passes: 7,500 updates and
120,000 exposures per fit. DEV TEDS selects each checkpoint. Both new selected
checkpoints are fusion epoch 2, at 6,250 updates and 100,000 exposures.

V8.3's auxiliary head is training-only. V8.4/V8.5 use a native VLM localization
branch with detail-task supervision and OTSL fusion. Localization replays the
model's own OTSL in an extra forward; its boxes do not rewrite official table text
or HTML. This is a TDATR-style adaptation, not exact paper reproduction.

## Full benchmark results

Each entry retains all 1,651 benchmark pages. Table metrics cover 458 scored pages
and 665 official table rows. Overall, Formula CDM and TEDS use 0–100 in this table;
edit distances use 0–1. The linked CSV preserves full precision.

| Version | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V8.1 | 96.316806 | 0.029707 | 98.625832 | 0.049639 | 93.295334 | 96.321028 | 0.060427 | 0.118769 |
| V8.2 | 95.988052 | 0.029707 | 98.625832 | 0.049639 | 92.309071 | 95.435168 | 0.067085 | 0.118771 |
| V8.3 | 96.108150 | 0.029707 | 98.625832 | 0.049639 | 92.669366 | 95.649207 | 0.064097 | 0.118771 |
| V8.4 | 95.785166 | 0.029707 | 98.625832 | 0.049639 | 91.700415 | 94.916362 | 0.070172 | 0.118769 |
| V8.5 | 95.779698 | 0.029707 | 98.625832 | 0.049639 | 91.684010 | 94.861013 | 0.070156 | 0.118769 |

## Scientific outputs and source

- [Full-precision benchmark metrics](results/tdatr-v8-45-001/METRICS.csv) and [V8.1–V8.3 category breakdowns](results/large-lora-001/benchmark/metrics/AGGREGATE.json).
- [Confirmation endpoints](results/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json); all 512 sources are retained, including 20 missing inputs per entry.
- [Training records](results/large-lora-001/training/metrics/training_scalars.csv), [DEV checkpoints](results/large-lora-001/training/metrics/dev_epochs.csv) and [figures](results/large-lora-001/FIGURES.md).
- [Dataset](results/large-lora-001/DATASET_CARD.md), [methods](results/large-lora-001/METHODS.md), [model-package identities](results/large-lora-001/MODEL_PACKAGES.json) and [reproduction source](source/generator_pilot/LARGE_CAMPAIGN_RUNTIME.md).

These are local official-protocol measurements, not an accepted leaderboard entry.

## V8.4 and V8.5 outputs and reproduction

- [Full numerical results, DEV selection and training statistics](results/tdatr-v8-45-001/RESULTS.json).
- [Dataset and training parameters](results/tdatr-v8-45-001/DATASET_AND_PROTOCOL.md), [input summary](results/tdatr-v8-45-001/DATASET_SUMMARY.json), [model-package identities](results/tdatr-v8-45-001/MODEL_PACKAGES.json) and [source](source/tdatr_adaptation/README.md).
- [V8.4 training history](results/tdatr-v8-45-001/training_curves/V8.4.csv) and [V8.5 training history](results/tdatr-v8-45-001/training_curves/V8.5.csv), each containing all 7,500 optimizer updates.
- [Five-version benchmark figure](results/tdatr-v8-45-001/figures/official_metrics.png), [confirmation figure](results/tdatr-v8-45-001/figures/confirmation_metrics.png), [training loss](results/tdatr-v8-45-001/figures/generation_loss_curves.png) and [localization loss](results/tdatr-v8-45-001/figures/localization_loss_components.png).
