# V2: formula replacement on TeleOCR

[Version index](../README.md) · [Project](../../README.md)

## Method

V2 transfers the formula-expert route to a TeleOCR page backbone, using PaddleOCR-VL1.6 region recognition, strict assembly and bounded recovery. It is the Tele-base route, not the stopped Paddle-plus-NaviDC prototype. The original native BF16 configuration contained mixed parameter dtypes.

## Results and interpretation

All 1,651 pages remain in each whole-page denominator. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matched TeleOCR control | 97.435109 | 0.029862 | 98.599622 | — | 96.691890 | 98.036939 | — | 0.119176 |
| V2 | 97.060019 | 0.029912 | 97.479332 | — | 96.691890 | 98.036939 | — | 0.119262 |

Overall falls by 0.375090 points against the paired TeleOCR control. Formula CDM declines from 98.599622 to 97.479332 while table metrics remain unchanged. Replacing a specialist component can damage a strong parser. Each arm retains one timed-out page empty. Native truncated TeleOCR content is retained, unlike the earlier MinerU/Paddle truncation rule.

Formula ED and Table ED were not supplied in this public historical aggregate. Its additional text-accuracy field equals 100 × (1 − Text ED), rather than a separate ninth official endpoint. Full-precision values, input/model/evaluator pins and original aggregate hashes are in the [whole-page record](../../artifacts/whole-page-results.json). These are local measurements, not accepted leaderboard entries.

## Implementation and evidence

- [Archived V2 implementation and reproduction notes](source/v2/README.md).
- [Whole-page aggregate and original-evidence hashes](../../artifacts/whole-page-results.json).
- [Reproduction guide](../../research/REPRODUCIBILITY.md).
