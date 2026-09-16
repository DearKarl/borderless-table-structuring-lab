# Local result archive

This directory publishes the available aggregate results for the current
Training/Hybrid handoff, including unsuccessful outcomes. It is not the entire
private research archive, a dataset release or evidence of public leaderboard
acceptance. No new evaluation was run to create this archive.

| Artifact | Full Table TEDS | Status |
|---|---:|---|
| `training-original/AGGREGATE.json` | 93.0862668980718 | Best verified Training checkpoint, equal to fixed Raw; zero adopted edits |
| `ocr-sidecar/PAIRED_TABLE_SUMMARY.json` | 93.08808257262936 | Highest completed correction-system aggregate; small gain, known crop defect |
| `complete-cell-cycle-001/PAIRED_TABLE_SUMMARY.json` | 93.08512970738767 | Below Raw; retained, not promoted |
| `native-tables/PROGRESS_2026-09-16.json` | Pending | 633/1651 committed pages and 890 cached generations at the dated handoff boundary |
| `ocr-sidecar-first-launch/FAILURE.json` | No score | Evaluator process exited 125; failed root preserved, not treated as a model result |

The two paired summaries and their READY markers, plus the failed-launch
receipt, are **byte-identical copies** of local aggregate receipts. Each READY
binds the corresponding summary SHA256. The Training file is explicitly a
sanitized scalar projection with the original source hash, not a full original
receipt. The native file is a dated progress projection, not a metric estimate.

All completed scores above use 1651 input pages and page-macro Table TEDS from
the frozen OmniDocBench evaluator at commit
`147cd5ac9472002f5751221d390bf00abdbc0d2f`. The paired summaries record the fixed
458 GT-table-page denominator and 665 matched evaluation samples. The 764
Training detected records describe edit adoption, not either metric denominator.
Structure TEDS is not the success criterion. None has achieved full TEDS >95.

See `../observed-results.json` for model identities and provenance. Public
architecture material is under `../../training/` and
`../../hybrid/native_tables/`. Native weights are pinned upstream; earlier
Training and OCR Hybrid model assets have separate GitHub Releases.

Private page predictions, inference caches, document images, Gold, credentials,
historical execution contracts and the device-migration handoff are intentionally
not included. A Git clone alone cannot resume the old machine's 633-page run.
The owner must privately transfer the bound execution archive and verify the
new device/runtime before any continuation. Do not replace existing evidence
or fabricate completion for missing work.

When the current native run genuinely completes, add its immutable aggregate
and summary hash here regardless of whether it beats 95. Do not replace Pending
with an upstream author's score, a partial-page score or the old sidecar score.
