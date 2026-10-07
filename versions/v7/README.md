# V7 fixed table reread and keep/replace policies

**Paused by the researcher on 2026-10-07 at 21:40 BST.** All engineering stopped;
local experiment processes exited and no V7 GPU or remote job was submitted.
The preserved work contains CPU training, packaging, development results and
local audit/design deliverables. There is no independent-confirmation or full
V7 benchmark result. See the [stopped-state summary](../../artifacts/results/v7/STOPPED_STATE.json).

The public configuration/protocol copies remove a private GPU UUID; their original
and publication hashes are distinguished in the
[publication manifest](../../releases/v6-v7-progress-20261007/PUBLICATION_MANIFEST.json).
Checkpoint weights are unchanged. The original protocol records the historical
V7.4 gate; the prospective amendment described below takes precedence.

One new net-TEDS-gain GBDT and four load-verified configurations are frozen in the
[package manifest](../../releases/v6-v7-progress-20261007/models/v7/PACKAGE_MANIFEST.json). Five grouped
training-only OOF fits selected an infinite acceptance margin, so V7.4 retains
every native result. This permitted negative outcome is preserved without forcing
activation. Fresh source confirmation is pending; no V7 benchmark result exists.

| Configuration | Policy | Newly trained deployed model |
|---|---|---|
| V7.1 | Keep default native TeleOCR output | None |
| V7.2 | Accept the shared fixed reread after common structural checks | None |
| V7.3 | Accept the same candidate under the frozen simple structure rule | None |
| V7.4 | Accept the same candidate if predicted net TEDS gain exceeds the OOF margin | One GBDT |

All configurations share one native parse and one fixed table candidate per
eligible region. The fixed scale is 0.535299427146522. Default native rendering
remains 200 DPI with the unchanged 3500-pixel cap. Crops come from the native
captured image. There is no original-RGB shortcut, source rerender for OCR,
formula intervention, scale search, or backbone training. Post-OCR rejection
does not save an already executed candidate OCR request.

## Frozen model and controls

The [protocol](PROTOCOL.json) and [features](features.py) freeze 34 scalar features:
native geometry/density, native/candidate table dimensions and cell statistics,
span fractions, canonical lengths, feature differences, and canonical content
disagreement. The strict parser preserves rowspan/colspan occupancy, decodes
entities, normalizes cell text to NFC with collapsed whitespace, and rejects
unknown or malformed boundaries. Unknown required features retain native.
The rule accepts equal row/maximum-logical-column counts with no increase in
empty-cell or adjacent nonempty duplicate-cell fractions, tolerance 1e-12.

The primary data contains exactly 112 prescribed V6 training table regions from
112 documents, one fixed-candidate row per region with unit weight. Targets are
candidate retained TEDS minus native TEDS. Candidate, input PNG and native pixel
hashes, source/split identities, feature vectors and scored-probe bindings were
checked. Other scales are not additional training rows. Old test and benchmark
outputs are excluded from fitting and model selection.

HistGradientBoostingRegressor uses squared error, learning rate 0.05, 100
iterations, seven leaves, minimum ten samples/leaf, L2=1, random_state=0 and no
early stopping. There is no fitted preprocessing. Source groups sorted by SHA256
are assigned round-robin to five folds. Training-only OOF scores select among
0, 0.0025, 0.005, 0.01, 0.02 and infinity. Ties within 1e-6 prefer fewer actual
replacements, then a larger margin. One final model is fitted on all 112 rows.

[Training evidence](../../artifacts/results/v7/TRAINING.json) retains all margin
choices. Native/all-keep OOF TEDS was 0.905307; the best finite-margin value was
0.904448. The six CPU fit timers sum to approximately 0.187 seconds; this excludes
startup, feature preparation and packaging. Exported final predictions match the
in-memory model exactly. Five fold models remain private diagnostics; only one
final model is deployed.

A NumPy integer serialization error interrupted OOF recording after the five
fold fits. The repair converted counts to JSON integers and recovered OOF
predictions from the five unchanged saved models. No completed fold was refitted;
the final model was fitted once. The [implementation receipt](IMPLEMENTATION_RECEIPT.json)
records original and repaired source hashes. Protocol, features and scientific
selection rules remained unchanged. Six contract tests passed.

## Existing development results

These 29 validation tables were already used to select the V6 fixed scale.
They provide development evidence, without an unbiased generalization claim.
Model, features, rule and margin were frozen before this comparison, and no
refit followed it. [Full development record](../../artifacts/results/v7/DEVELOPMENT.json).

| Configuration | Region-macro TEDS |
|---|---:|
| V7.1 | 0.891977 |
| V7.2 | 0.899530 |
| V7.3 | 0.907530 |
| V7.4 | 0.891977 |

V7.4 is equal to V7.1 by construction under the selected infinite margin. It
therefore cannot exceed the highest of the three controls by the preregistered
0.005 threshold on any matched confirmation set. This is a property of the
frozen policy, not an observed fresh-set score or permission to select a new
threshold. The independent confirmation remains separately recorded.

## Independent confirmation and resource boundaries

The authorized confirmation targets 128 new documents, at most one table per
document, from the pinned audited scientific-paper source release. At most 640
source candidates and 5 GiB of new downloads are allowed. Source selection is
independent of recognition quality and model decisions, with old inspected
documents and known benchmark identities excluded. Failed retrieval, alignment,
inference and reference dispositions are retained. Quality labels are offline.

Before any new confirmation quality was generated or inspected, the nominated
candidate changed prospectively to V7.3. The frozen amended gate requires all128
complete documents, V7.3 mean TEDS at least0.005 above max(V7.1,V7.2), and strictly
positive paired95% document-bootstrap lower bounds against both controls
(2000 shared resamples, seed0). V7.4 remains an all-keep negative comparator;
its original gate did not pass. The [analysis protocol](../../artifacts/results/v7/CONFIRMATION_ANALYSIS_PROTOCOL.json)
and [implementation](confirmation_gate.py) retain the amended rule. Four synthetic
gate tests passed, without opening fresh quality outcomes. These are operational
criteria, not a power or novelty claim. The user pause supersedes execution:
no confirmation or full run resumes without a new decision.

Physical GPU0 is forbidden. Every allocation must verify a fresh physical
index/UUID map, exclude both current index0 and the known historical GPU0 UUID,
and bind explicit idle allowed UUIDs from indices1-7. The pilot allows at most
two simultaneous GPUs and four allocated GPU-hours; the entire task allows
16 GPU-hours and 48 hours from dispatch. CPU-only fitting used no GPU. Current
remote runtime verification is unavailable; no V7 GPU work was launched. At stop,
77 source documents had been attempted,76 PDFs retrieved and one download
interrupted. Four reference-alignment attempts yielded one template match and
three unresolved/failed cases; native predicted-region correspondences and
fresh quality outcomes remained zero. Downloaded PDFs are not completed
confirmation data. Partial files were preserved. No generator model was trained.
