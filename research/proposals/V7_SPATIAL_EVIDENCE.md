# Visual evidence feasibility and proposed learning route

Status: concrete design proposal frozen 2026-10-07T20:32:42.443754+00:00. This document does not launch acquisition, feature-model fitting, extra OCR or V7.5. All engineering is now paused for research discussion. This is an archived spatial-feature GBDT proposal, not the later generator fine-tuning proposal.

## What the 112 TRAIN assets actually contain

The training-only asset audit retained locally verified all 112 native crop file/pixel hashes and all 112 source-PDF hashes. Both native and fixed-candidate HTML are present for all 112. Saved selected table outputs contain **no independent cell/word positions**; the only inference geometry is the table region itself. Native and candidate have equal row counts in 101 cases and equal maximum logical column counts in 111.

The pinned reference release contains 6560 nonempty annotated cells across all 112 sources, each with an in-bounds four-coordinate prefix. Its bbox arrays append a fifth value whose semantics have not yet been verified. These are reference-crop coordinates, not native-page coordinates. They can support a training-only correspondence audit once their transform is verified; they are forbidden as deployment features.

The initial schema check required exactly four bbox values and incorrectly counted zero boxes. That initial receipt is preserved as `TRAINING_ASSET_FEASIBILITY_initial_strict_bbox4.json`; the corrected audit explicitly validates the first-four coordinates of four/five-value records against the reference-image bounds. No score/model was changed.

A deterministic native-pixel projection diagnostic took0.1991 CPU seconds across 112 crops (p950.00536s/crop, excluding image loading). Its ink-band count matched the parsed native row count in **0/112** cases. Multiline cells, rules and headings prevent interpreting raw text-line bands as table rows. This diagnostic establishes cheap access to pixels, not useful grounding or model improvement.

## Proposed deployable evidence

Keep the existing 34 scalar features and add six spatial features, for 40 total: candidate-minus-native row-cut ink cost, column-cut ink cost, empty-cell/ink conflict, cell text-length versus component-area discrepancy, span-boundary/component conflict, and minimum localization support. The exact definitions and proposed deterministic extractor are in [expansion design](V7_TRAINING_EXPANSION.json).

The extractor uses only captured default-native pixels and the two parsed structures. Per-image Otsu foreground and connected components provide independent spatial observations. A deterministic grid fit with the same objective and tie-breaking for each structure proposes cell support; spans are respected. An ambiguous fit, insufficient support, malformed structure, memory cap or 20-second CPU timeout retains native. All failures stay in training/evaluation dispositions.

Character counts versus foreground mass are only a weak content proxy. They cannot verify whether a particular word appears in an image. No independent recognized word locations are currently saved, and embedded source-PDF text is unavailable in raster-only deployment. A future lexical-localization/OCR branch would require its own pinned extractor and cost proposal; it is not silently added here.

Before collecting new documents, audit the reference-to-native transform on TRAIN only, inspect alignment overlays with deterministic sample selection, and measure cell IoU/foreground coverage against the training annotations. Require the geometric coverage and CPU criteria in the JSON plan. Check image sensitivity using controlled occlusion/substitution while keeping structure inputs fixed. These are feasibility checks, not held-out accuracy claims. If they fail, stop this route rather than presenting global density as image-to-structure evidence.

## Controls and learning curves

Compare native, fixed, frozen V7.3, a geometry rule, scalar 34 GBDT and augmented 40 GBDT. The two GBDTs use the same labels/documents/unit weights/five folds/parameters/training-OOF margin grid, including infinity. The geometry rule requires V7.3 plus nonworsening spatial costs with at least one strict improvement. Record all-keep outcomes.

Plan 300 independent TRAIN documents: existing 112 plus 188 new. Add disjoint 64 DEV and 128 future-confirmation documents, requiring 380 newly usable documents in total. Use one table/document. All old validation/test groups, all 640 current confirmation candidates and known benchmark overlaps are excluded. The present 128 confirmation documents never become training data.

The nested TRAIN sizes are 56,112,168,224,300 with three predeclared source-order seeds 0/1/2. Use identical prefixes for both feature arms; report their old/new source composition. Full 300 membership is identical across seeds, so duplicate full fits are omitted: 26 pipelines and at most 156 fold/final CPU fits. The 64 DEV set describes the learning curves; it cannot select a favorable size, seed or retuned model. The single proposed future nominee is the augmented model at 300 documents, frozen before opening the new 128 confirmation outcomes.

The plan specifies a separate future gate for review; it does not alter the current V7.3 gate.300 is a feasibility target, not a power calculation or promised improvement. Scientific-paper sources still limit domain coverage.

## Resource feasibility and stopping

No additional resources are allocated by this design. A separately approved expansion would cap 960 source candidates, 380 fixed regional rereads, 4 allocated GPU-hours, 2 allowed GPUs, 8 CPU-core-hours and 3 GiB new downloads. The current V7 budget retains 16 total / 4 pilot / 12 conditional-full GPU-hours and its original 48-hour deadline.

Historical native collection consumed5.697 allocated GPU-seconds/page across 299 pages. Historical mixed regional actors consumed1.738 GPU-seconds/physical request. At 960 native pages plus 380 rereads, this gives a planning proxy of1.703 GPU-hours, or3.405 with a2× allowance. These mixed historical jobs are not a fresh V7 ETA. Current TRAIN PDFs average1363980bytes;960 PDFs at that mean would use1.219GiB, and at the observed 90th-percentile size2.163GiB. Retrieval failures and heavy tails remain possible; stop at the hard download cap.

Use already downloaded unused-source shards only after the acquisition protocol is accepted. Freeze source order/split/correspondence before quality, preserve every attempted-source disposition, and keep failures/natural ties. Do not fill missing splits with current confirmation documents, rerun failed pages for better results, or expand automatically to 1000 TRAIN documents. Log actual native parser requests, candidate OCR calls, feature latency, model calls and all startup/failure costs separately.
