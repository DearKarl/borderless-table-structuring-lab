# V0: specialist table replacement

[Version index](../README.md) · [Project](../../README.md)

## Method

V0 keeps the frozen MinerU2.5-Pro-2605-1.2B page backbone and replaces eligible native table regions using fresh NaviDC recognition. Assembly checks preserve the original page when a replacement cannot be adopted. The measured CUDA reproduction uses the released btsl 0.2.0 source snapshot.

## Results and interpretation

All 1,651 pages remain in each whole-page denominator. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MinerU control | 93.167994 | 0.064838 | 95.553206 | — | 90.434549 | 93.101331 | — | 0.152900 |
| V0 | 94.599252 | 0.064833 | 95.553206 | — | 94.727857 | 96.599701 | — | 0.154207 |

Table TEDS rises from 90.434549 to 94.727857 and Overall from 93.167994 to 94.599252. This supports the measured table-replacement system under its recorded protocol, not a cross-version ranking. Assembly accepted 1,247 pages and retained raw content on 404 pages (393 assembly abstentions and 11 native abstentions).

Formula ED and Table ED were not supplied in this public historical aggregate. Its additional text-accuracy field equals 100 × (1 − Text ED), rather than a separate ninth official endpoint. Full-precision values, input/model/evaluator pins and original aggregate hashes are in the [whole-page record](../../artifacts/whole-page-results.json). These are local measurements, not accepted leaderboard entries.

## Implementation and evidence

- [Archived V0 implementation and reproduction notes](source/v0/README.md).
- [Whole-page aggregate and original-evidence hashes](../../artifacts/whole-page-results.json).
- [Reproduction guide](../../research/REPRODUCIBILITY.md).

## Earlier table-only studies

An earlier native-table route, with a different MinerU revision and evaluator protocol, scored **98.750687 Table TEDS** and **99.291381 Structure TEDS**, against raw Table TEDS **93.086267**. It is a separate component study, not V0 Overall. The Explicit-v2 Training checkpoint adopted zero edits and matched raw. The [table archive](../../artifacts/results/README.md) retains the Training, OCR sidecar, complete-cell-cycle and native-table results and failure records; see [training implementation](../../training/README.md) and [native table implementation](../../hybrid/native_tables/README.md).
