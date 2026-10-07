# V6 and V7 progress archive

Preserved on2026-10-07 after the researcher paused all engineering for discussion.
V6 completed; V7 trained and packaged a CPU keep/replace controller whose selected
margin is infinity. V7 independent confirmation and full evaluation did not run.
No generator post-training model or official leaderboard rank is claimed.

Read [current state](../../research/CURRENT_STATE.md),
[V6 results](../../versions/v5_native/README.md),
[V7 methods](../../versions/v7/README.md), and
[baseline audit](../../artifacts/results/v7/BASELINE_COMPARABILITY.md).

## Preserved models and provenance

`models/v6/` contains four first-party controller checkpoints, six configs and
historical package/evaluation manifests. `models/v7/` contains one final GBDT,
four configs and package/training-interface provenance. Third-party OCR weights
and source datasets are not redistributed. Model checkpoints remain byte-for-byte
unchanged. Historical manifests preserve original hashes and status text.

V7 configuration/protocol copies replace the private GPU UUID with a placeholder.
The original exclusion of physicalGPU0 remains mandatory. Historical hashes for
those original files therefore do not equal public-copy hashes. Use
`PUBLICATION_MANIFEST.json` for the explicit original-to-public mapping and
`SHA256SUMS.txt` for published bytes. Public copies are archival templates, not
live job configurations; no script should be launched solely from this snapshot.

V6 frozen runtime source is retained in its original archive with its source
manifest; current analysis code under`versions/v5_native/` may be newer than the
executed runtime. No historical result was rerun or checkpoint refitted for this
publication. V7 gate amendment is documented separately from its original protocol.

## Publication boundary

All eight V6 official metric fields, category breakdowns, failures, costs,
controller-call counts, input/output change counts and paired quality aggregates
are preserved under`artifacts/results/v6/`. V7 CPU training, development results,
comparison audit and stopped-state summary are preserved under`artifacts/results/v7/`.
Raw GT, source images/PDFs, predictions, per-page identity ledgers, private
deployment bindings and operational conversations remain in the local workspace.
The local preservation snapshot retains original prepublication bytes.

The literature scan and two distinct unexecuted proposals are retained in
`research/`. Their presence does not approve training or increase compute budgets.
