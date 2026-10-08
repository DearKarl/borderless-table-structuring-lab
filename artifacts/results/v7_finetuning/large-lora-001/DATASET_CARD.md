# Large LoRA campaign: prospective dataset card

Protocol status: 2026-10-08. Acquisition and native input construction are active.
The final membership, accepted counts, sequence distribution and visual audit are
not complete. This document describes a frozen design, not a delivered dataset.

## Units and targets

The target is 20,000 distinct real training tables from at least 5,000 source
documents, with at most four training tables per document. A separate development
cohort contains 256 documents and confirmation contains 512 documents, each with
one prespecified table. A document, page, table, cell, token and training exposure
are different units. Three passes would provide 60,000 exposures per fit without
increasing the number of distinct training tables. These targets are not a power
calculation. Synthetic tables are not part of this campaign's primary data.

## Sources and permissions

The source currently being acquired is the original public TRAIN partition of
[PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m), revision
`35b1c097807e0b07ec5313879b85956b7b3890db`. The PDF annotation archive has SHA256
`6c7f9b0aa56c9690b4ba4a54d054dd8b510b35198c2c30d45f661cb11cf79932`.
Annotations are distributed under
[CDLA-Permissive 2.0](https://cdla.dev/permissive-2-0/). Dataset preparation code
is documented by the [table-transformer authors](https://github.com/microsoft/table-transformer).
Each original source PDF additionally requires verified source identity and an
accepted reuse license. Current accepted source categories are CC BY, CC BY-SA
and CC0; original license strings and source metadata are retained privately.
Manuscripts, retractions, historical OCR sources, unverified licenses and changed
source pages are excluded. No original public validation or test partition enters
training.

[FinTabNet.c](https://huggingface.co/datasets/bsmock/FinTabNet.c), revision
`e5673a90b98d02c4832f9e836d72762f0e8933a0`, was inspected as a possible financial
source. Original-PDF retrieval has not been established, so no financial coverage
is claimed. Pre-cropped images at another DPI are not substituted. Actual source,
domain and language coverage will be reported after construction. Unknown overlap
with the base model's pretraining data prevents a claim of pretraining independence.

## Splits and selection

Known benchmark sources and all exposed project sources are excluded. Remaining
source IDs are ordered by SHA256 of `large-lora-001-source-v1:` followed by the
source ID. The integer represented by the first eight hexadecimal digits modulo
100 assigns buckets 0–89 to training, 90–93 to development and 94–99 to confirmation.
Tables within sources use the analogous `large-lora-001-table-v1:` hash order.

Training candidates are drawn in deterministic round-robin order across eight
observed strata: merged column header present/absent, at least 25 rows or eight
columns present/absent, and explicit blank cell present/absent. The four-table
document cap applies across strata. Exhausted strata redistribute their slots;
examples are not duplicated to fill quotas. Ruledness, dimensions, sequence length,
table area and text/numeric mix are measured and reported, including coverage gaps.

Held-out sources pass static provenance, license and label checks before a target
is frozen. Membership is fixed before native detection or recognition quality.
Subsequent missing detections and failed generations stay in the denominator.
No performance-based replacement is permitted. Source, normalized content and exact
and near-duplicate image checks are required before quality evaluation. Final
overlap checks and membership hashes remain pending.

## Labels and native inputs

Published cell rows, columns, contiguous spans, explicit blank cells and PDF-frame
boxes are retained with the original annotations. Training text uses published
`pdf_text_content`, NFC normalization and collapsed whitespace. Replacement,
private-use, surrogate and unsupported control characters are rejected. No guessed
character repairs or implicit blank cells are inserted. Complete rectangular grids
must round-trip exactly through the pinned TeleOCR OTSL decoder. Original scoring
targets are retained separately; training normalization does not rewrite official
evaluation ground truth.

Original PDF pages pass through the pinned default 200 DPI native renderer and
base-model layout detector. Training receives predicted native crops only. GT
boxes never create model inputs. The exact server-generated crop bytes are retained:
local and server PDF rendering differed in an early check, so locally reconstructed
pixels are not an admissible substitute. Four-page checks established identical
layout outputs between individual processing and a fixed batch size of four.

Cell boxes are mapped through recorded PDF/render dimensions, the predicted crop,
native padding and resizing, and processor patch/grid dimensions. Complete crop
coverage allows at most two native pixels at a boundary; any clipping is recorded.
Unsupported rotations, ambiguous matches and crops containing another complete
source table are rejected for training. Native image, pixel, processor tensor/grid,
prompt, target, mask and cache hashes bind each accepted example.

Assistant text through EOS receives generation loss. Prompt, image placeholders,
padding and template suffix are masked. The total sequence ceiling is 16,384 tokens;
silent truncation is forbidden. A cell anchor is the first full token that consumes
its origin `<fcel>` or `<ecel>` marker. Natural BPE tokens can cross the marker's
character boundary, so token spans and character spill are recorded. Anchors must
be unique and monotonic; blank and merged origin cells are included. Continuation
markers do not create additional cell boxes. This prospective mapping correction
was made before formal training, using three feasibility examples excluded from
all campaign cohorts.

## Quality requirements and remaining work

Every candidate receives provenance, encoding, source text correspondence, grid,
crop, coordinate, token and duplicate checks. Acquisition/infrastructure failures
are distinguished from source or label exclusions. All flagged cases require a
recorded disposition. At least 200 randomly stratified accepted examples require
visual review before the main freeze. Review by one assistant is not independent
human annotation validation. The first 2,000 accepted inputs are a data milestone,
not a separate training experiment. Final training cannot begin on an evolving
partial dataset.

No source PDFs, images, reference-label bodies, per-example predictions or private
deployment details are published here. Final aggregate composition and rejection
counts will be added after acceptance.
