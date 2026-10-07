# Learned selection within the V5 candidate pipeline

> Superseded input assumption, 2026-10-07: the researcher now requires the default
> TeleOCR input path, including native-rendered regional crops, for future work.
> See [the revised adaptive-scale plan](ADAPTIVE_NATIVE_SCALE_PLAN.md). This
> document retains the earlier proposal; its original-RGB baseline and teacher
> must not be executed as the current experiment. No training ran under this plan.


Date: 2026-10-07. Status: proposed experiment, with a completed saved-event audit.
No new model has been trained or integrated by this document.

## Researcher direction and relationship to distillation

The researcher requests a project-trained component, such as GBDT, in the
combined candidate pipeline, with evidence that it is invoked and genuinely
effective. The immediate candidate is a newly trained regional result selector.
The [distillation plan](DISTILLATION_PLAN.md) remains a subsequent option for
transferring verified improvements into native recognition weights.

The first experiment tests selection quality after candidates are generated.
It does not claim to avoid their inference cost. A trained selector is a project
artifact; using GBDT alone is not a novel research contribution.

## Evidence and the failure mode to avoid

The completed V4 record contains 1476 estimator prediction calls. All 241 A
selections had identical A/B candidate-image fingerprints; the 55 pages with
distinct candidates selected B. Thus those decisions produced no selected
input different from fixed B at that boundary. Calls alone did not establish
effective intervention. This does not assert deterministic OCR-output equality.

A new audit of all 1651 saved V5.5 event files found:

| Stage | Accepted, content-distinct events | Pages with such events |
|---|---:|---:|
| Formula 1.25x | 793 | 238 |
| Guard | 6 | 6 |
| Table 1x | 14 | 14 |

There are 813 unique page/type/bbox regions on 256 unique pages across these
events. Counts establish byte-level candidate diversity, not semantic differences
or accuracy gains. They do not measure learned selections: no selector ran here.
All other pages must remain in full evaluation. Source:
[V5 aggregate](../artifacts/results/v5/AGGREGATE.json) and the saved event archive
whose SHA256 is `873fbbc0f26565ecb1f3204d07491e4b96a60815d85cf7219e9156b377bb38d8`.

## Proposed implementation contract

1. Freeze one original-image native parse per page and generate a V5.5 candidate
   bank on that lineage. Preserve native and every accepted regional alternative.
   Do not combine outputs of independently run V5 arms as a shared baseline.
2. Preserve exact region/page identity and dependencies between stages. Begin
   with unambiguously aligned formula/table regions. For overlapping, missing or
   conflicting regions, retain the fixed V5.5 outcome and report exclusion.
   Text guard behavior remains fixed in the first experiment.
3. Deduplicate identical candidate strings before selection. Also report safe
   task-specific normalization/tree equivalence; different whitespace alone is
   not evidence of a useful intervention. Never collapse meaning-changing edits.
4. Fit a new GBDT regressor to predict candidate quality gain relative to the
   native candidate. Retain native as the no-change action. Pick the predicted
   best candidate only above a validation-frozen margin; otherwise keep native.
   Do not reuse V4 weights with a different feature/target interface.
5. Apply all chosen candidates once to the shared block tree, then assemble and
   score complete pages. Keep unrelated spans and ordering untouched. Record
   validity rejection, identity problems and fallback separately.

Proposed features available at inference include image/glyph geometry, blur and
contrast, task type, candidate scale, output length and length change, repeated
patterns, formula delimiter balance, table dimensions/span indicators, and
candidate edit distance. A missing feature has an explicit flag. Token confidence
is excluded unless the runtime actually records it consistently. No reference
answer, evaluation score, filename, page ID or dataset identity is a feature.

## Training and controls

Use external labeled regions with source-document separated train/validation/test
splits and overlap checks. Reuse the candidate dataset preparation described in
the distillation plan, with native and candidate results additionally retained.
Initial target: 2000 table and 2000 formula training regions, 500 validation
regions and 1000 held-out test regions, subject to verified data availability.
These are pilot counts, not a power calculation. Never train on OmniDocBench
inputs, answers, predictions or per-page intervention labels.

Training targets are within-region quality differences using external labels
(table TEDS and formula CDM under frozen annotation conversion). Equalize
task/source weighting and audit malformed or unscorable examples. The initial
regression target is quality, not a hidden mixture of cost and accuracy.

| Control | What it tests |
|---|---|
| Shared native output | Baseline before interventions |
| Fixed V5.5 selection on the same candidate bank | Strong existing combination |
| Validation-frozen simple rules / linear gain model | Whether a complex learned selector is needed |
| Newly trained GBDT selector | Proposed learned candidate adoption |
| Shuffled-label GBDT on training data | Whether informative supervision matters, rather than arbitrary output changes |

Keep candidate generation, inputs, parser weights, assembly and evaluator fixed.
Report offline GT-best candidate quality only on development data as a diagnostic
ceiling; never use it to select evaluation predictions or as a submitted score.
If the candidate bank offers no useful room over fixed V5.5, stop expanding the
selector and investigate better candidates or the distillation route instead.

## Required acceptance evidence

Report these separately, with counts and denominators:

1. Model load, checkpoint hash and actual estimator predict calls.
2. Regions with more than one valid, distinct candidate.
3. Learned choices differing from the fixed V5.5 choice.
4. Final region/page outputs actually changed by those choices.
5. Paired improvements, regressions and ties against both native and fixed V5.5,
   plus full-page Overall and component metrics on the full population.
6. Measured generation, feature, selection and total serving costs.

Disabling the selector must reproduce fixed V5.5 from the same candidate bank.
Add synthetic integration fixtures where candidate differences and model-driven
choices are known. Such tests establish wiring only, not benchmark effectiveness.
Use source-grouped paired intervals for quality and repeat fitting seeds on the
external pilot. Freeze margins and model choice before final evaluation.

Zero changed decisions is a negative result, not an excuse to force random
adoptions or lower the threshold after looking at test scores. Nonzero changes
without quality benefit establish activity, not usefulness. Meeting all three
goals of invocation, effective intervention and quality contribution is an
experimental question, not a guaranteed outcome.

## Next bounded work

The immediate executable preparation is candidate-bank extraction, immutable
lineage checks, external-data availability verification, and a data-free selector
wiring test. Model training follows only after feature/label/split contracts are
frozen. Preserve completed V5 outputs and the original V4 model unchanged.
No GPU round, new checkpoint, official submission or publication is claimed here.
