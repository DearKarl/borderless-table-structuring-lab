# Local result archive

This directory publishes the available aggregate results for the current
Training/Hybrid handoff, including unsuccessful outcomes. It is not the entire
private research archive, a dataset release or evidence of public leaderboard
acceptance. No new evaluation was run to create this archive.

| Artifact | Full Table TEDS | Status |
|---|---:|---|
| `training-original/AGGREGATE.json` | 93.0862668980718 | Best verified Training checkpoint, equal to fixed Raw; zero adopted edits |
| `ocr-sidecar/PAIRED_TABLE_SUMMARY.json` | 93.08808257262936 | Historical correction system; small gain, known crop defect |
| `complete-cell-cycle-001/PAIRED_TABLE_SUMMARY.json` | 93.08512970738767 | Below Raw; retained, not promoted |
| `native-tables/PAIRED_TABLE_SUMMARY.json` | 98.75068667701048 | Completed 2026-09-17; highest project system aggregate; local scored target met |
| `native-tables/AGGREGATE.json` | 98.75068667701048 | Sanitized completed-result projection, not a replacement for the original paired receipt |
| `native-tables/PROGRESS_2026-09-16.json` | No score | Historical 633/1651-page and 890-cache handoff snapshot; no longer current |
| `ocr-sidecar-first-launch/FAILURE.json` | No score | Evaluator process exited 125; failed root preserved, not treated as a model result |

The three paired summaries and their READY markers, plus the failed-launch
receipt, are **byte-identical copies** of local aggregate receipts. Each READY
binds the corresponding summary SHA256. The Training file is explicitly a
sanitized scalar projection with the original source hash, not a full original
receipt. The native AGGREGATE is also a sanitized projection. The retained native
PROGRESS file is a dated historical projection, not a metric estimate or current status.

All completed scores above use 1651 input pages and page-macro Table TEDS from
the frozen OmniDocBench evaluator at commit
`147cd5ac9472002f5751221d390bf00abdbc0d2f`. The paired summaries record the fixed
458 GT-table-page denominator and 665 matched evaluation samples. The 764
Training detected records describe edit adoption, not either metric denominator.
Structure TEDS is not the success criterion. The completed native-table Hybrid
achieved full TEDS **98.75068667701048**, exceeding 95; the older results did not.

The native run completed **1651 pages and 2296 successful canonical generation
calls**. Against Raw full **93.0862668980718** / structure
**95.70961193860337**, it scored structure **99.2913812392524**, with full gain
**5.664419778938679** points, structure gain **3.5817693006490288** points and
structure-minus-full gap **0.5406945622419244** points. Baseline and candidate
each have 665 matched samples, zero evaluation errors and zero timeouts.

Native paired-summary SHA256:
`5ccecac587dc99528e6e0881453c682dac7fc72089ed2d9901f18b9b064b1542`.
Native READY SHA256:
`15069ed43f1d89dd562fa307e07f95c36e68efa403383304f48436d086ad5e4d`.
This is the highest completed **system** result in the project. The best
**trained checkpoint** is still Explicit-v2 Original, equal to Raw. The scored
local target is met, not verified public leaderboard acceptance, global SOTA,
or a production guarantee. Repeated aggregate benchmark exposure remains disclosed.

See `../observed-results.json` for model identities and provenance. Public
architecture material is under `../../training/` and
`../../hybrid/native_tables/`. Native weights are pinned upstream; earlier
Training and OCR Hybrid model assets have separate GitHub Releases.

Private page predictions, inference caches, document images, Gold, credentials,
historical execution contracts and the device-migration handoff are intentionally
not included. A Git clone alone cannot recreate the completed private pipeline,
predictions or execution archive. The portable assembler001 subset lacks the
current assembler003's Official consumer-format gate and is not an end-to-end
runner. Reproduction on another device requires private assets and a separately
verified runtime binding, not a false identity claim or blind cache reuse.

The completed native result retains mixed MPS execution epochs, including
full-K/V query chunking and final Q1 synchronization/unused allocator-cache
release. That final resource step did not change model or generation settings;
no bit-identity or causal quality benefit is claimed. The upstream author's
97.05 is not our reproduction, and the separately downloaded pinned NaviDC
weights do not by themselves reproduce this result. Historical failed and
below-Raw outcomes remain preserved.
