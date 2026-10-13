# V4: learned input selection

[Version index](../README.md) · [Project](../../README.md)

## Method

A GBDT predicts a raster input action before frozen TeleOCR recognition. Training uses 30 PDF groups and 90 action records, with geometry, glyph-height, reliability, blur, contrast, whitespace, density and processor-grid features. Action A keeps the native image; B applies up to 1.5× bicubic scaling under the 3,500-pixel long-side limit. The measured policy uses margin 0 and fallback B, with at most one recognition pass per page.

## Results and interpretation

Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4 | 97.171890 | 0.030295 | 98.499335 | — | 96.045863 | 97.382499 | — | 0.121037 |

All 1,651 pages are retained: 1,644 completed and seven timeouts kept empty. A/B choices were 241/1,410, including fallback. The public result records do not supply Formula ED or Table ED. This completed system evaluation does not isolate the learned selector's benefit over matched fixed-input policies. Neither historical V2 TeleOCR nor a public leaderboard row is a paired control.

## Implementation and full evidence

- [Detailed method](METHOD.md) and [frozen source/model release](release/README.md).
- [Full evaluation record](release/results/REPORT.json) and [all available metric fields](release/results/METRICS.json).
- [Selector source](release/runtime/gpu/code/hybrid/v4_input_selector/selector_input.py). The archived default disables learning; the documented measured entry enables it explicitly.
