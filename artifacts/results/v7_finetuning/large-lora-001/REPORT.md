# V8 large LoRA campaign: dataset freeze milestone

October 9, 2026: the common dataset is frozen at **20,000 TRAIN tables from
11,217 documents**, with 256 DEV and 512 confirmation sources. All 21,535 source
pages completed native processing. All 23,619 static-eligible targets have input
QA dispositions; 22,973 have usable native inputs before final TRAIN selection.
The 200 visually reviewed examples all remain in TRAIN. Full-corpus runtime
validation is in progress; no formal training or accuracy result is claimed by
this snapshot. See the [dataset card](DATASET_CARD.md),
[aggregate counts](DATASET_FROZEN_SUMMARY.json) and [measured dataset figures](DATASET_FIGURES.md).

Naming amendment, October 9, 2026: V8.1 is ordinary LoRA (formerly V7.3.0), V8.2
is LoRA-GA (formerly V7.3.1), and V8.3 is LoRA-GA with cell-location supervision
(formerly V7.3.2). Each retains seeds 0 and 1. These are the same six fits, with
unchanged scientific protocol and original evidence paths. See the explicit
[version mapping](VERSION_MAPPING.json).

## Historical preparation snapshot

Snapshot: 2026-10-08 18:30 UTC. Data preparation is running. **No formal training
fit, confirmation score or full-benchmark result exists for this campaign yet.**

This campaign asks whether source-verified table supervision improves TeleOCR,
whether gradient initialization improves over ordinary LoRA, and whether cell
geometry adds a further free-generation benefit. See the
[dataset card](DATASET_CARD.md) and [prospective methods](METHODS.md).

## Observed progress

The full official annotation archive was downloaded and checksum verified. Indexing
scanned 401,733 source documents in 2,047 seconds. After original-split, known-source
and annotation checks,310,702 documents were indexed;639,419 TRAIN-partition table
annotations were eligible for further checking. These are annotation candidates,
not accepted native training inputs.

At the snapshot,1,528 original source PDFs had been retrieved. The fixed first
candidate tranche contained 2,688 tables passing static source text/structure checks.
Native coverage, token/cache checks, duplicate review and the accepted-sample visual
audit remain to be completed. The first 463 pages were dispatched to four queued
native processing jobs; actual concurrency follows fresh resource availability.

Four source-feasibility pages produced five predicted table crops. Three of four
annotated tables passed source-label checks; one was excluded for corrupt original
text encoding. Batch-four detection reproduced the individual layout outputs exactly
and took 22.43 seconds of detection time for those four pages. This small measurement
is not a forecast of corpus throughput.

The actual-model interface check passed for 174 mapped cells, including blank and
merged origins. LoRA-GA initialization and fresh-base reload produced exactly matching
evaluation logits. The visual merger, adapters and training-only geometry head all
received nonzero finite gradients. No optimizer updates were performed. Technical
implementation checks do not establish recognition improvement or generalization.

## Failures and limitations retained

- A local/server PDF rendering mismatch led to retaining exact server-native crop
  bytes instead of reconstructing pixels locally.
- Natural BPE tokens crossed cell-marker character boundaries. The prospective
  anchor rule was corrected before formal training, with target text unchanged.
- Two numerical gradient-check attempts failed under fused BF16 SDPA. A controlled
  math-backend equality diagnostic passed without relaxing its tolerance; production
  SDPA remains unchanged and its repeat-gradient variation is disclosed.
- The first bulk acquisition process lacked network access and failed DNS before
  downloading bytes. It was stopped and the same ordered candidates were retried
  with network access in a new attempt directory. Those failures are infrastructure
  events, not evidence that the corresponding source documents are unsuitable.
- Financial-domain source reconstruction remains unverified. No financial, language
  or corpus-size coverage is inferred from a dataset name or a target count.

The next deliverables are the complete 20,000-table common dataset freeze, six matched
fits with scalar telemetry, DEV-only model nomination, confirmation diagnostics and
the unconditional full benchmark of the nominated seed 0 model and matched controls.
Training curves and accuracy claims will be published only when measured.
