# Borderless Table Structuring Lab

Research on reliable document parsing: how input processing, specialist models and learned adaptation can improve text, formulas and table structure while preserving correct content. The repository retains methods, complete available measurements and negative results across successive studies.

[Research Question](#research-question) · [Latest Version](#latest-version) · [History Version](#history-version) · [Results index](research/RESULTS.md) · [Implementation index](versions/README.md) · [Reproduction](research/REPRODUCIBILITY.md) · [Contributing](CONTRIBUTING.md)

## Research Question

How can a document parser correct errors without damaging reliable native predictions, and do improvements transfer from isolated regions to complete documents? We study input choice, specialist integration and generator adaptation under matched controls. The research objective is an officially recognized OmniDocBench improvement and a generalizable contribution; no accepted leaderboard position or paper acceptance is claimed.

## Latest Version

### V8: table-specialist LoRA

V8.1 uses ordinary LoRA, V8.2 adds LoRA-GA initialization, and V8.3 adds a training-only cell-location head. Each has seeds 0 and 1, trained on the same 20,000 real tables from 11,217 documents with default native 200-DPI inputs. DEV-selected checkpoints are fixed before independent evaluation.

All eight policies below retain a 1,651-page evaluation universe. Table metrics use the official component denominators; these are not 1,651 separate tables.

| Policy | Table TEDS ↑ /100 | Structure TEDS ↑ /100 | Table Edit Distance ↓ /1 | Overall ↑ /100 |
| --- | --- | --- | --- | --- |
| Matched native baseline | 96.962 | 98.296 | 0.034118 | 97.539 |
| Legacy simple rule | 96.604 | 98.165 | 0.037184 | 97.420 |
| V8.1/seed0 * | 90.299 | 93.213 | 0.085605 | 84.879 |
| V8.1/seed1 | 93.295 | 96.321 | 0.060427 | 96.317 |
| V8.2/seed0 * | 90.944 | 93.892 | 0.081719 | 85.095 |
| V8.2/seed1 | 92.309 | 95.435 | 0.067085 | 95.988 |
| V8.3/seed0 | 92.471 | 95.439 | 0.065634 | 96.042 |
| V8.3/seed1 | 92.669 | 95.649 | 0.064097 | 96.108 |

\* Execution qualifications: [see full results](research/versions/V8.md#execution-qualifications).

The prespecified V8.3/seed0 nominee gains 4.2541 TEDS percentage points over matched native on 512 confirmation sources, but loses 4.4912 points on full-benchmark Table TEDS. The evidence does not support general parser replacement. These are matched Transformers results, not historical vLLM comparisons or an external leaderboard submission.

[V8 methods and complete results](research/versions/V8.md) · [All eight official metrics](artifacts/results/v7_finetuning/large-lora-001/benchmark/metrics/METRICS.csv) · [Confirmation analysis](artifacts/results/v7_finetuning/large-lora-001/confirmation/metrics/CONFIRMATION_AGGREGATE.json) · [Figures](artifacts/results/v7_finetuning/large-lora-001/FIGURES.md)

**V8.4 / V8.5 are reserved, not implemented or started:** ordinary LoRA / LoRA-GA with the same full TDATR-style adapted supervision. The same TRAIN/DEV set is retained; diagnosis and a complete protocol freeze precede execution. [Version reservations](artifacts/results/v7_finetuning/large-lora-001/NEXT_VERSIONS.md).

## History Version

Each introduction explains the method, available results, interpretation, implementation and original evidence. Different backbones and protocols are not a single controlled ranking. V3 has one entry with V3.1/V3.2 subsections; V7 separates its historical naming collisions.

| Version | Methods | Source |
| --- | --- | --- |
| V0 | MinerU + NaviDC table replacement | [Introduction](research/versions/V0.md) |
| V1 | MinerU + Paddle formula replacement | [Introduction](research/versions/V1.md) |
| V2 | TeleOCR + Paddle formula replacement | [Introduction](research/versions/V2.md) |
| V3 | Layout/formula integration (V3.1); text expert (V3.2) | [Introduction](research/versions/V3.md) |
| V4 | GBDT selection of native input actions | [Introduction](research/versions/V4.md) |
| V5 | Original RGB input and regional rereading | [Introduction](research/versions/V5.md) |
| V6 | GBDT/MLP regional scale controllers | [Introduction](research/versions/V6.md) |
| V7 | Generator LoRA pilots; separate legacy keep/replace selector | [Introduction](research/versions/V7.md) |

The [version index](research/versions/README.md) keeps stable introductions as new studies become Latest Version. Original source, releases and measured artifacts remain at their existing paths. See [third-party notices](notices/THIRD_PARTY.md) for attribution and licensing.
