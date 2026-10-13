# V1: formula replacement on MinerU

[Version index](../README.md) · [Project](../../README.md)

## Method

V1 uses MinerU as the page backbone and PaddleOCR-VL1.6 as a formula expert. Strict one-to-one geometry and syntax checks control formula replacement while preserving other regions. The internal formula_v0 package name is an assembly prototype, not the measured V0 table route. Fresh inference and verified-cache development are distinct modes.

## Results and interpretation

All 1,651 pages remain in each whole-page denominator. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MinerU control | 93.167994 | 0.064838 | 95.553206 | — | 90.434549 | 93.101331 | — | 0.152900 |
| Paddle fresh control | 95.211472 | 0.052914 | 96.780383 | — | 94.145464 | 96.513127 | — | 0.140712 |
| V1 | 93.217499 | 0.065097 | 95.727598 | — | 90.434549 | 93.101331 | — | 0.154055 |

Overall improves only slightly over the MinerU control, from 93.167994 to 93.217499; Formula CDM rises while Table TEDS is unchanged. The standalone Paddle control is context, not evidence that its integration must improve the backbone. MinerU/V1 retain 27 truncated generations as empty primary outputs; the Paddle control records 13 failed and 11 truncated outcomes.

Formula ED and Table ED were not supplied in this public historical aggregate. Its additional text-accuracy field equals 100 × (1 − Text ED), rather than a separate ninth official endpoint. Full-precision values, input/model/evaluator pins and original aggregate hashes are in the [whole-page record](../../artifacts/whole-page-results.json). These are local measurements, not accepted leaderboard entries.

## Implementation and evidence

- [Archived V1 implementation and reproduction notes](source/v1/README.md).
- [Whole-page aggregate and original-evidence hashes](../../artifacts/whole-page-results.json).
- [Reproduction guide](../../research/REPRODUCIBILITY.md).
