# Distilling V5 interventions into native recognition

> Superseded input assumption, 2026-10-07: the researcher now requires the default
> TeleOCR input path, including native-rendered regional crops, for future work.
> See [the revised adaptive-scale plan](ADAPTIVE_NATIVE_SCALE_PLAN.md). This
> document retains the earlier proposal; its original-RGB baseline and teacher
> must not be executed as the current experiment. No training ran under this plan.


Date: 2026-10-07. Status: proposed experiment protocol; no training launched.
The researcher supports this direction. Dataset availability, training support,
the final launch manifest and confirmatory margins remain to be verified.

## Question and immediate objective

Can a student initialized from TeleOCR absorb verified benefits of V5.5 regional
interventions while retaining accurate predictions, using the ordinary native
recognition path without additional formula/table rereads or guard retries?

The immediate target is teacher-level quality at lower inference cost, not a
mandatory score above V5.5. The ultimate official leaderboard objective remains.
The existing 98.347514 score belongs to the untrained V5.5 pipeline and must never
be attributed to a future student. Matching it is an aspiration, not a forecast.

## Existing evidence to reuse

- [V5 report](../versions/v5/README.md) and
  [aggregate](../artifacts/results/v5/AGGREGATE.json): V5.2 Overall 98.173146,
  V5.5 Overall 98.347514, all 1651 pages retained in each arm.
- [Protocol](../versions/v5/protocol.json): original-image input, native prompts,
  decoding, regional operations and evaluator identities.
- Saved native/intermediate predictions support diagnosis only. Independent
  V5 arms have different native outputs, so do not label their cross-arm
  differences as the teacher's verified repairs. Compare stages within the
  same teacher transaction when constructing intervention evidence.
- V2 motivates protection against regression; V4 contributes input/feature
  instrumentation, without requiring the old GBDT in the new student.

No automatic rerun of V5.1-V5.5 is needed to design this study.

## Teacher, student and inference contract

Teacher T0 is the frozen V5.5 pipeline. Baseline B0 is unchanged TeleOCR with the
V5.2 original-image path. Both use the actual hash-verified inference assets,
not an unverified model-repository revision substituted for those files.

The proposed student starts from the same TeleOCR weights. Initially train LoRA
on supported language-decoder modules, keeping the vision encoder frozen.
Resolve trainable module names, image-token handling and checkpoint conversion
from the actual model implementation; do not assume an inference plugin also
supports training. No new tokenizer or resized vision input is proposed.

First scope: table and display-formula content recognition. Layout, reading-order
assembly and ordinary text recognition retain the baseline path. Task-specific
adapter activation must be demonstrated: layout/text requests must use the
unchanged base weights, with no added recognition requests. If the runtime cannot
support this contract, revise the protocol before training, rather than silently
changing every task. Record the serving memory and adapter-switching cost.

Student inputs are the original native content-request crops and prompts, not
the teacher's enhanced views, final predictions, reference answers or quality
scores. Teacher targets must map to the same region and content task. Reject
ambiguous mappings. Native parsing already has multiple stages and regional
requests; this study does not claim one model call per page.

Teacher information unavailable in the student's view may be impossible to
distill. Audit legibility and effective pixel grids. Do not enlarge student
inputs silently to obtain the desired score.

## Data preparation and frozen splits

Use external labeled training data. Candidate sources are
[PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) for table image/HTML pairs and
[UniMERNet resources](https://github.com/opendatalab/UniMERNet) for mathematical
expression recognition. These are candidates, not downloaded or approved asset
manifests. Check the particular dataset release, sample terms, source identifiers,
annotation conversion and accessibility before fixing either source.

Pilot target: 4000 training regions (2000 tables, 2000 formulas), 500 validation
regions (250 each), and 1000 locked external test regions (500 each). These are
feasibility sizes, not a power calculation or promised available counts.

Split by source document and available template/family identifiers before target
generation. Keep variants of a document in one split. Check exact hashes,
near-duplicate images and normalized content against benchmark inputs and
between splits. Missing source identifiers are a limitation, not evidence of
independence. Synthetic variants inherit the original split.

Training may enrich real errors and verified correct examples, recording sampling
probabilities and counts. Validation/test populations must be fixed before
inspecting their teacher gains, retain natural error prevalence, and include
failures. Do not fill a quota by inventing repairs. Bound the initial candidate
pool at 20000 regions; report insufficient repair yield instead of continuing
unbounded generation. Freeze one training population for all main students.

Crop datasets support content-model feasibility, not full V5.5 reproduction:
the whole-page layout and guard context are absent. Call their producer the
V5-derived regional teacher. Separately assemble a source-held-out, labeled
full-page external set, initially 100 pages if accessible, to test integration
and teacher/student cost. Full-pipeline or generalization claims remain pending
if this full-page set is unavailable.

Exclude OmniDocBench images, answers, predictions and generated variants from
student training and validation. It remains an exposed final benchmark, not a
blind generalization test. Base-model pretraining overlap may remain unknown;
distinguish that limitation from the new training-data audit.

## Training targets and proposed mechanism

For each external training region, save image and source identity, native input,
baseline output, teacher intermediate/final outputs, ground truth, validity,
quality and intervention lineage. Measure tables with TEDS/TEDS-S and formulas
with CDM plus normalized edit distance. Check annotation conversions and
equivalent markup before labeling apparent differences as errors.

The candidate mechanism is repair-aware sequence distillation with preservation
examples. It uses output sequences; teacher logits are not required.

1. Verified repair: the teacher improves on its own native output and meets the
   predefined target-quality rule. Train on that verified teacher sequence.
2. Verified keep: the native output is correct and the teacher does not improve
   it or damages it. Train on the native sequence to preserve that capability.
3. Neither output is trustworthy: use a validated ground-truth target when its
   format is compatible; otherwise exclude the ambiguous target for every arm.

Quality thresholds are task-specific and frozen using training examples and
annotation audits, before the comparison. They are not selected on OmniDocBench.
Sampling and target decisions use external training labels only.

The proposed objective combines token cross-entropy on selected targets with an
additional loss weight on verified-keep examples. Start preservation weight at
1 relative to the main loss; report the effective weights and sample exposures.
This is a provisional design, not a new-algorithm claim. More elaborate DPO,
reinforcement learning and learned routing are outside the first pilot.

## Controls

| ID | Configuration | Purpose |
|---|---|---|
| B0 | Unchanged TeleOCR, V5.2 input | Baseline ability without new training |
| T0 | Frozen V5.5; regional teacher explicitly distinguished on crop data | Teacher quality and cost reference |
| S1 | LoRA supervised fine-tuning on external GT | Benefit from ordinary supervised training |
| S2 | LoRA on all usable teacher outputs, without quality selection | Naive sequence-distillation baseline |
| S3 | LoRA with verified repair/keep target selection and preservation weighting | Proposed training method |

S1-S3 share initialization, native inputs, trainable modules, source population,
optimizer, sequence policy, token/update budgets and checkpoint selection rule.
The quality-selection method uses GT supervision; it must not be described as
label-free or compared only against a teacher-only baseline. S1 is essential.
Record differing loss weights and target lengths rather than asserting identical
effective supervision. Align target normalization across arms.

Only if S3 shows useful validation behavior, add two attribution controls:
remove target selection while retaining the keep-example weighting, and retain
selection while removing the extra keep weight. Use the same examples and update
budget. A benefit over S2 alone does not establish benefit over ordinary SFT.

Provisional pilot defaults: BF16, LoRA rank 16/alpha 32/dropout 0.05, AdamW,
learning rate 2e-5, effective batch 32 regions, seed 0, at most 3 epochs or 1000
updates, whichever comes first. Start from native image/context settings and
audit long examples; no silent truncation of supervised targets. A training-only
smoke may change memory settings, then all main arms use the frozen settings.
Do not tune only S3. Confirmatory runs use seeds 0, 1 and 2 after pilot review.

## Stages and resource envelope

| Stage | Work | Evidence required to advance |
|---|---|---|
| P0 | Inspect model training support; 32-example overfit and adapter roundtrip | Correct masked loss, changed adapter weights, reproducible load/serve and unchanged non-target path |
| P1 | Freeze external splits; generate and audit teacher/native pairs | Verified provenance, valid region alignment, quantified repair/keep/damage yield |
| P2 | Train S1-S3; evaluate on external validation data | Actual quality gains and damage counts, not merely declining training loss |
| P3 | Run attribution controls and repeated seeds if justified | Contribution survives matched controls and variation; frozen candidate selection |
| P4 | Locked external tests, full-page integration, then one frozen OmniDocBench evaluation per retained final arm | Complete denominators, quality/cost comparison and explicit scope of conclusions |

Proposed initial ceiling: 96 allocated A100 GPU-hours, divided into 8 for P0,
20 for teacher data, 48 for S1-S3 training (16 each), and 20 for validation and
integration. These are ceilings, not throughput forecasts. P3 and confirmation
need a separately frozen budget based on pilot measurements. Include adapter
conversion, teacher generation, failed startup and evaluation in cost records.

Use up to six available GPUs for three two-GPU training jobs only if the runtime
benefits from two GPUs; otherwise prefer one GPU per job. Remaining available
cards may serve data generation/evaluation. These are proposed roles, not physical
GPU assignments. Recheck occupancy before launch and never displace other work.

Stop on exhausted limits, nonfinite loss, broken label/image alignment, changed
data identity or systematic assembly damage. Preserve failed runs; do not quietly
restart or overwrite them. Low repair yield triggers a data/teacher diagnosis,
not more blind training. Do not use an OmniDocBench score to choose pilot
checkpoints or loss weights.

## Metrics, candidate selection and claim criteria

Report per-task TEDS/TEDS-S, CDM/edit distance, repair/damage counts and malformed
output rate. For full pages, report official Overall, all components, failed and
empty pages, and reading order. Every system uses the same declared denominator.
Do not invent an Overall for unrelated crop datasets.

Validation checkpoint rule: minimize the equally task-weighted mean of
`1 - TEDS` and `1 - CDM` on the fixed validation split, with a predeclared
invalid-output guard and no inspection of test results. The same rule applies
to S1-S3. Freeze exact evaluation intervals before training.

For confirmation, predeclare a maximum acceptable teacher-student quality gap
and a minimum useful measured inference-cost reduction based on the pilot and
research priorities. They remain unset here; no arbitrary tolerance is claimed
to make an observed score a pass. Keep task regression limits alongside Overall.
Use paired intervals with source-document grouping; average crop scores and
official page aggregates are different estimands. Include training-seed variation
and flag inconclusive intervals rather than asserting equivalence.

Cost comparison must use matched hardware, concurrency, inputs and serving
settings, including preprocessing, adapter handling, request counts, output
length, warm/cold time and memory. The original V5 allocated GPU-hour difference
is not a clean steady-state saving estimate. Trainable weights alone do not prove
efficiency; measure the saving. Report amortization of teacher/training cost
separately from per-document inference cost.

Possible outcomes: a useful trained student; a training gain without evidence
for S3-specific novelty; an efficiency gain without the desired quality; or a
negative distillation result. All remain publishable experiment records, but
none guarantees a venue or official leaderboard acceptance.

## Research contribution and deliverables

Distillation itself is established; see the
[PaddleOCR documentation](https://www.paddleocr.ai/v3.0.3/en/version2.x/ppocr/model_compress/knowledge_distillation.html).
Compare the proposed mechanism with hard-example training, selective distillation,
preservation/continual-learning objectives and document post-training before
claiming novelty. Prior selective correction work includes
[DEC](https://arxiv.org/abs/2608.09842). A full novelty review is still pending.

Required artifacts are a pinned training/data protocol, split and overlap audit,
training source and tests, dataset lineage, adapter/checkpoint hashes, exact
commands, measured resource logs, and complete comparative results. Operational
records and raw benchmark payloads remain outside public research files.

Exact installation and launch commands are intentionally not invented before
model training support and source assets are inspected. The first executable
deliverable is a validated training smoke recipe plus a frozen manifest, not a
claim that the current inference scripts already implement distillation.
