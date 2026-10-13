# V8: table-specialist LoRA

[Versions](../README.md) · [Project](../../README.md)

| Version | Method |
| --- | --- |
| V8.1 | Ordinary rank-8 attention LoRA with a trainable visual merger |
| V8.2 | LoRA-GA initialization with the same adapter and merger training |
| V8.3 | LoRA-GA with an additional training-only cell-location head |

## Training and inputs

The corpus contains 20,000 real tables from 11,217 TRAIN documents, with 256 DEV
sources and 512 confirmation sources. Inputs use predicted native crops at PDF
200 DPI. Each run completes 3,750 optimizer updates and 60,000 exposures. DEV TEDS
selects the checkpoint. V8.3's auxiliary cell head is not used during inference.

## Full benchmark results

Each entry retains all 1,651 benchmark pages. Table metrics cover 458 scored pages
and 665 official table rows. Overall, Formula CDM and TEDS use 0–100 in this table;
edit distances use 0–1. The linked CSV preserves full precision.

| Version | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V8.1 | 96.316806 | 0.029707 | 98.625832 | 0.049639 | 93.295334 | 96.321028 | 0.060427 | 0.118769 |
| V8.2 | 95.988052 | 0.029707 | 98.625832 | 0.049639 | 92.309071 | 95.435168 | 0.067085 | 0.118771 |
| V8.3 | 96.108150 | 0.029707 | 98.625832 | 0.049639 | 92.669366 | 95.649207 | 0.064097 | 0.118771 |

## Scientific outputs and source

- [Full-precision benchmark metrics](results/large-lora-001/benchmark/metrics/METRICS.csv) and [category breakdowns](results/large-lora-001/benchmark/metrics/AGGREGATE.json).
- [Confirmation endpoints](results/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json); all 512 sources are retained, including 20 missing inputs per entry.
- [Training records](results/large-lora-001/training/metrics/training_scalars.csv), [DEV checkpoints](results/large-lora-001/training/metrics/dev_epochs.csv) and [figures](results/large-lora-001/FIGURES.md).
- [Dataset](results/large-lora-001/DATASET_CARD.md), [methods](results/large-lora-001/METHODS.md), [model-package identities](results/large-lora-001/MODEL_PACKAGES.json) and [reproduction source](source/generator_pilot/LARGE_CAMPAIGN_RUNTIME.md).

These are local official-protocol measurements, not an accepted leaderboard entry.
