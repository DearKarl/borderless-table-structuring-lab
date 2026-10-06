# Experimental results

The project evaluates input processing and specialist integration for document
parsing. Results below are grouped by experiment; a higher score across groups
does not by itself identify a method improvement. Overall, CDM, TEDS and TEDS-S
are expressed on a 0–100 scale, and edit distances on 0–1. Missing values are
shown as dashes and remain null in the supplied-result record.

## V0 through V2

These historical experiments evaluate all 1,651 pages. See the
[original aggregate record](../artifacts/whole-page-results.json) for model,
input and evaluator identities and failure handling.

| Method | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | TEDS-S ↑ | Reading-order ED ↓ |
|---|---:|---:|---:|---:|---:|---:|
| MinerU control | 93.167994 | 0.064838 | 95.553206 | 90.434549 | 93.101331 | 0.152900 |
| Paddle control | 95.211472 | 0.052914 | 96.780383 | 94.145464 | 96.513127 | 0.140712 |
| V0 | 94.599252 | 0.064833 | 95.553206 | 94.727857 | 96.599701 | 0.154207 |
| V1 | 93.217499 | 0.065097 | 95.727598 | 90.434549 | 93.101331 | 0.154055 |
| TeleOCR V2 control | 97.435109 | 0.029862 | 98.599622 | 96.691890 | 98.036939 | 0.119176 |
| V2 | 97.060019 | 0.029912 | 97.479332 | 96.691890 | 98.036939 | 0.119262 |

V2 is 0.375090 Overall points below its paired TeleOCR control. Its formula
replacement reduced CDM while table metrics remained unchanged. Truncation
handling differs between the earlier MinerU/Paddle group and the TeleOCR pair;
the groups should not be interpreted as one controlled experiment.

## V3.1 original-image formula replacement

The researcher supplied and identified this project's V3.1 result on 2026-10-07.
The method retains the original formula-replacement pipeline while bypassing
PDF transcoding and resampling. Only Overall was supplied.

| Method | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | TEDS-S ↑ | Reading-order ED ↓ |
|---|---:|---:|---:|---:|---:|---:|
| V3.1 | 97.6973 | — | — | — | — | — |

No project baseline was supplied for this result, so no paired gain is reported.
The page count, failure handling, exact run revisions and raw evaluator files
remain unspecified. The [archived implementation](../versions/v3_1/README.md)
is available; its exact binding to the scored run still needs documentation.

## V3.2 full-set OFF and ON comparison

The researcher reports a completed 1,651-page evaluation for each arm using the
same official evaluator and ground truth. Failed pages remain in the denominator.
The supplied values are transcribed below; they were not independently rescored.

| Arm | Pages | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | TEDS-S ↑ | Reading-order ED ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | 1651 | 93.4951 | — | 98.2000 | 86.9556 | — | — |
| ON | 1651 | 93.7208 | — | 98.4454 | 87.3918 | — | — |
| ON minus OFF | — | +0.2257 | — | +0.2454 | +0.4362 | — | — |

Text quality reportedly regressed slightly; no numeric text metric was supplied.
Missing metrics are not inferred from the rounded Overall and component scores.
The official evaluator revision, GT hash, per-arm failure counts and scored code
revisions are not yet included in the supplied evidence.

### Supplementary successful-page statistics

| Subset | Overall ↑ | Page count |
|---|---:|---:|
| OFF's own successful pages | 97.6657 | — |
| ON's own successful pages | 97.5607 | — |

The two subsets are not confirmed to contain identical pages. These statistics
cannot establish a paired accuracy regression or replace the full-set result.
Assessing reliability versus conditional quality requires aligned page statuses,
a common-success comparison, and the complete-denominator scores above. Changes
in formula or table metrics do not alone demonstrate direct formula/table edits
by the text expert.

Source for both V3 sections:
[researcher-supplied result record](../artifacts/results/v3-reported-20261007.json).

## V4 learned input selection

V4 evaluates a GBDT input-action selector with TeleOCR on raster inputs. All
1,651 pages are counted: 1,644 completed and seven timeout outputs remained empty.

| Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | TEDS-S ↑ | Reading-order ED ↓ |
|---:|---:|---:|---:|---:|---:|
| 97.171890 | 0.030295 | 98.499335 | 96.045863 | 97.382499 | 0.121037 |

See the [method description](V4.md) and
[full-precision evaluation record](../releases/v4-gbdt-evaluation-20261006/results/REPORT.json).
This run does not establish selector benefit over a matched fixed-input policy.

## Earlier table experiments

The earlier native-table route achieved 98.750687 full TEDS and 99.291381 TEDS-S
under an older protocol. The corresponding raw baseline was 93.086267 full TEDS.
These are table-only measurements and cannot be combined with another run's text
and formula scores to create an Overall result. The separate Training checkpoint
adopted zero edits and matched its raw baseline. See the
[table experiment archive](../artifacts/results/README.md).

## Evaluation conventions

Overall = (100 × (1 − Text ED) + Formula CDM + Table TEDS) / 3.
Reading-order ED and TEDS-S do not enter Overall. Scores above are local results,
not accepted leaderboard entries. Compare methods using identical input versions,
model settings, evaluator/runtime, failure rules and compute accounting. Preserve
each experiment's original records when additional evidence becomes available.
