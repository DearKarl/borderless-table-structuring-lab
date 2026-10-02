# Borderless Table Structuring Lab

Research code for document parsing that combines a page backbone with specialist table, formula, layout and text models. This repository archives the V0–V3.2 series, measured whole-page results through V2, and the earlier independent Training and native-table routes.

## Measured whole-page results

These are **our historical local measurements on 1,651 OmniDocBench pages**, not a claim of a verified public leaderboard position. Higher Overall is better. All measured rows below use the later official evaluation protocol; model versions and output handling matter. See the [result record](artifacts/whole-page-results.json) for exact scores, component metrics, source hashes and protocol details.

| Model / system | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | Table TEDS-S ↑ | Reading-order ED ↓ |
|---|---:|---:|---:|---:|---:|---:|
| MinerU2.5-Pro-2605-1.2B (V1 control) | 93.167994 | 0.064838 | 95.553206 | 90.434549 | 93.101331 | 0.152900 |
| PaddleOCR-VL-1.6 (V1 control) | 95.211472 | 0.052914 | 96.780383 | 94.145464 | 96.513127 | 0.140712 |
| TeleOCR (V2 control) | 97.435109 | 0.029862 | 98.599622 | 96.691890 | 98.036939 | 0.119176 |
| Hybrid V0 (MinerU + NaviDC tables) | 94.599252 | 0.064833 | 95.553206 | 94.727857 | 96.599701 | 0.154207 |
| Hybrid V1 (MinerU + Paddle formulas) | 93.217499 | 0.065097 | 95.727598 | 90.434549 | 93.101331 | 0.154055 |
| Hybrid V2 (TeleOCR + Paddle formulas) | 97.060019 | 0.029912 | 97.479332 | 96.691890 | 98.036939 | 0.119262 |
| Hybrid V3.1 | Pending | Pending | Pending | Pending | Pending | Pending |
| Hybrid V3.2 | Pending | Pending | Pending | Pending | Pending | Pending |

Overall, CDM, TEDS and TEDS-S use a 0–100 scale; edit distances use 0–1. Overall = (100 × (1 − Text ED) + Formula CDM + Table TEDS) / 3. Reading order and TEDS-S do not enter Overall. Values are calculated at full precision, then displayed to six decimal places. These archived runs are contextual comparisons, not one unified paired experiment.

The paired V2 hybrid is **0.3751 Overall points below** its TeleOCR raw control. Expert integration did not improve that measured run. V3.1/V3.2 have bounded engineering checks, but no completed full-benchmark result is reported here.

MinerU/Paddle and V1 use empty primary output for truncated generations; the V2 pair retains TeleOCR native truncated output. Cross-group scores provide context and are not a controlled single-variable comparison. The earlier Paddle development result, 95.14238890128128, is distinct from the fresh control above. CPU fixtures and small smoke checks are not accuracy results.

## Versions and source

| Version | Implementation | What it adds |
|---|---|---|
| [V0](versions/v0/README.md) | Root `btsl/`, `hybrid/native_tables/`, CUDA reproduction templates | Replaces native table regions using NaviDC while preserving page content |
| [V1](versions/v1/README.md) | MinerU/Paddle page assembly and native controllers | Geometry- and syntax-checked formula replacement |
| [V2](versions/v2/README.md) | Tele-base full-run and specialist controllers | Formula expert integration, fallback and strict run evidence |
| [V3.1](versions/v3_1/README.md) | Conservative layout/formula integration | Fresh recognition after accepted geometry changes; explicit asset binding |
| [V3.2](versions/v3_2/README.md) | OvisOCR2 text-slot integration | Text expert dispatch with strict completion and fallback checks |

See the [version archive](versions/README.md) for entry points and scope. Run each archived version from its own directory; package names are deliberately preserved. These are research source snapshots with explicit external runtime prerequisites, not bundled model environments. TeleOCR source is obtained separately from its pinned upstream and checked against fixed member hashes. Models, datasets, predictions and private deployment records are not included.

## Earlier native-table and Training routes

The earlier MinerU 2604/MLX + NaviDC/MPS route achieved **98.75068667701048 full Table TEDS** and **99.2913812392524 structure TEDS** under the older evaluation protocol. These are table metrics, **not Overall**, and do not belong in the whole-page table above. The raw table baseline was 93.0862668980718, a gain of 5.664419778938679 Table TEDS points.

The [preserved native-table guide](LEGACY_NATIVE_TABLES.md) contains installation, model pins, CLI usage, historical aggregates and reproduction limits. The separate [Training route](training/README.md) remains available with its original checkpoint and loading contract. Neither route is relabeled as a new V3 result.

## Reproduction and attribution

Published source/package locks identify the public archive bytes. Publication changed documentation, example paths, container label names and external-source preparation; it did not rerun historical model inference or scoring. Historical measurements therefore do not certify a new clone, hardware platform or prepared environment.

Ground truth is evaluator-only. Inference uses original page inputs and declared model assets. Failed or incomplete runs cannot be silently promoted to measured results. The [contribution rules](CONTRIBUTING.md), [third-party notices](notices/THIRD_PARTY.md) and version-specific environment contracts describe the boundaries. Necessary original third-party license text, frozen evaluator text and Chinese recognition/filename test strings are retained.
