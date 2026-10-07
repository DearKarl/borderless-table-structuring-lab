# V5 research proposal

Date: 2026-10-07. Status: five-arm inference and scoring complete.

This document preserves the study rationale and original execution plan. The
[completed V5 report](../versions/v5/README.md) is authoritative for actual
configuration, GPU allocation, measured results and deviations. Later proposed
research directions below have not been executed.

The primary performance objective remains an officially recognized OmniDocBench
Overall score above the leader. The researcher has additionally requested a
research contribution suitable for submission to a CCF A venue. A leaderboard
result and a publishable contribution require different evidence; neither has
been established. Architecture, per-experiment budgets and submission date remain open.

## Lessons from completed work

| Evidence | Implication for V5 |
|---|---|
| Historical V0 Overall 94.5993 versus its MinerU control 93.1680 | Composition can help a weaker backbone; it does not establish benefit on TeleOCR. |
| Paired V2 Overall 97.0600 versus raw TeleOCR 97.4351; table scores unchanged and formula CDM lower | Broad replacement can damage a strong backbone. Measure repairs and regressions separately. |
| Early Training adopted zero edits; its output matched Raw | Report actual intervention coverage. A trained module can contribute nothing to the final system. |
| Earlier native-table TEDS 98.7507 under an older protocol | Investigate reusable crop, inference and assembly behavior. Do not combine this component with later text/formula scores. |
| Researcher reports V3.1 Overall 97.6973 and V3.2 OFF/ON 93.4951/93.7208 | Preserve their distinct protocols. V3.2's +0.2257 full-set gain and differing successful-page subsets motivate aligned failure analysis. |
| V4 Overall 97.17189, with seven timeouts retained empty | Retain the selector and instrumentation as controls; separate failure handling, input protocol and learned selection effects. |

Sources: [whole-page records](../artifacts/whole-page-results.json),
[table archive](../artifacts/results/README.md),
[V4 release](../releases/v4-gbdt-evaluation-20261006/README.md).
These are not all mutually comparable experiments. V4 evaluated raster inputs,
not a demonstrated adaptive rasterization method on original vector PDFs.

The public [ki-OCR release](https://github.com/yushuosun/ki-OCR-v1/tree/ac1ab4f5ded8b025f47a7683f58e65db4ac43a2b)
motivates strong original-image, typed-reread and recovery controls. Its
98.38731 result is a historical recovered composite, not a fresh reproduction.
Its public source/preparation release is not a complete independently runnable
R3 installation. A partial recreation must be named as such and must not inherit
the historical score. Private group materials are not part of this public plan.

## Consolidated first-round plan

The researcher approved A-E as V5.1-V5.5 and requested all five full-benchmark
experiments with overlapping GPU execution. This supersedes the earlier plan to
promote only pilot winners. Necessary runtime checks precede launch, but pilot
accuracy was not a gate for whether an arm ran. All five runs and scoring
completed; use the completed report for observed outcomes.
The immediate hypothesis is that original-image input plus targeted regional
rereading and bounded recovery can improve a strong reproducible TeleOCR system
without widespread damage to already correct content.

### What is already available

All V0-V4 implementation archives are present locally. See the
[source inventory](SOURCE_INVENTORY.md) for concrete entry points and inspection
limits. V3.1 and V3.2 are implemented packages, not README placeholders.
Source availability, external runtime readiness and the identity of a particular
scored run are separate questions.

A concrete V3.1 discrepancy needs resolution: the archived
[native entry](../versions/v3_1/hybrid_v3_full_eval_v1/native.py) calls
read_fn followed by do_parse, while the reported 97.6973 method bypasses PDF
transcoding/resampling. Locate the scored variant or its patch in historical
artifacts; do not silently call the current archive that exact original-image run.
This does not invalidate the reported score or establish that the variant is lost.

### How the evidence changes the design

| Evidence | V5 decision |
|---|---|
| V0 table integration improved its control; older native-table TEDS reached 98.7507 | Prioritize table crops, structural validation and failure recovery. Reuse source knowledge without importing the old append-tables assembly, which did not preserve interleaving. |
| V1 formula integration produced only a small Overall change; V2 broad Paddle replacement reduced Overall by 0.3751 | Do not default to broad cross-model formula replacement. Retain native correct content and measure regressions. |
| Reported V3.1 original-image variant reached 97.6973 | Reconcile its source/input variant and saved outputs. Use it as a historical candidate; do not claim a paired gain over B without compatible evidence. |
| V3.2 is an Ovis text expert, with +0.2257 full-set Overall and a reported text regression | Reuse dispatch, completion and fallback instrumentation. Do not assume formula/table metric changes prove direct formula/table improvement by the expert. Do not add wholesale Ovis substitution to the first round. |
| V4 has feature/scale instrumentation but no demonstrated matched selector gain | Keep scale observations and failure records; GBDT is optional, not the compulsory V5 backbone. |
| Public ki-OCR uses original images, TeleOCR formula rereads, guard retries and table rereads | Keep A-E to test these distinct mechanisms. Same-TeleOCR formula rereading is not the rejected V2 Paddle intervention. |
| Public ki-OCR Table150 is a failed candidate relative to its historical composite | Start table rereading at 1x; larger crops are not presumed better. This historical comparison is not an isolated causal scale ablation. |
| Early Training accepted zero edits | Report accepted interventions and net repairs, not merely model training or execution success. |

The colleague's 98.38731 is a historical recovered self-test, not an official
score or a matched control for our runs. Review of the companion study also
informs prioritization; its non-public source details remain outside this public
document. No component scores from different systems are spliced into an Overall.

### Comparison matrix

| Arm | Configuration | Role and reuse |
|---|---|---|
| V5.1 (A) | Pinned TeleOCR default input path | Establish the default-route control. Reuse compatible historical native predictions; regenerate only if identity or required settings differ. |
| V5.2 (B) | Same TeleOCR/backend/decoding, direct original-image input | Isolate bypassing image-to-PDF rendering. Reuse a verified direct-image native cache if available; the formula-enabled V3.1 score alone is not B. |
| V5.3 (C) | B plus TeleOCR table-crop rereading at 1x | First-priority regional candidate. Preserve unrelated slots and evaluate full-page assembly. |
| V5.4 (D) | B plus TeleOCR display-formula rereading at 1.25x | Test the colleague's different, plausible formula mechanism. Track both repairs and damage; do not equate it with Paddle replacement. |
| V5.5 (E) | B -> formula reread -> bounded anomaly guard -> table reread | Test the combined pipeline and interactions against B, C and D. Save every intermediate stage. |

Declare crop geometry, rotation, padding, scale caps, prompts, acceptance rules
and the actual recognition calls. A nominal 1x reread still adds inference and
can differ from native crop preparation. Direct-image input still allows the
model's internal preprocessing. For guard equation retries, record effective
pixel factors: the reviewed reference includes an additional 200/72 multiplier;
the labels 1.0/0.75/1.5 alone do not specify its actual views.

E is an inspired comparison, not an exact recreation of the recovered ki-OCR
score. Do not claim the missing external Paddle producer has been reproduced.
Use inference-only fixed triggers for empty/malformed/repetitive output, a
bounded retry sequence and a no-change fallback. Syntax validity alone is not a
semantic quality guarantee. Never use benchmark answers to select final patches.

E changes both composition and guard behavior. If it wins and a component claim
is needed, add an E-without-guard comparison on the same development pages.
D is not necessarily reusable as a later state if preceding stages differ.
Cache only identical requests with identical crop/model/prompt/decoding identity;
do not manufacture E by pasting arbitrary independently scored outputs.

### Approved execution and evaluation

1. **Prepare and freeze.** Reuse existing code and locate pinned model/data
   assets. Freeze one backend, model revision, decoding setup, dataset manifest,
   official evaluator, renderer and failure rules across all arms. The reported
   V3.1 score remains a historical reference until its exact variant is bound.
2. **Check all five implementations.** Use a fixed 16-page mixed-content smoke
   set to check input identity, output assembly, failure handling and throughput.
   Fix execution defects; do not tune rules to the smoke scores.
3. **Run all five full sets concurrently.** Each arm covers all 1,651 pages.
   Initial scheduling assigns one GPU to V5.1, one to V5.2, two independent page
   shards to V5.3, one GPU to V5.4 and three shards to V5.5, subject to observed
   availability. Prefer fresh end-to-end execution for every arm; preserve any
   shared-cache lineage and distinguish diagnostic cached runs from fresh runs.
4. **Evaluate together.** Score the five frozen outputs under the same evaluator.
   Retain failures in the denominator and record actual stage interventions.
   Compare B-A, C-B, D-B and E against B/C/D, without treating E as a guard-only
   ablation. Do not adjust thresholds between arms based on benchmark scores.
5. **Choose the submission candidate.** Select the strongest complete,
   reproducible system. V5.5 is not presumed best. If needed, independently
   replay the frozen selected configuration before official retesting. New
   methods or sweeps are outside this five-arm round.

The previous 192-page accuracy-gated pilot is not a prerequisite for the approved
full runs. Intermediate partial scores, if inspected, remain exploratory and
must not change the fixed inference protocol. OmniDocBench is an exposed
benchmark for this project; these runs do not create a blind test.

Eight A100 80GB GPUs are available. Set exact per-page deadlines, retry limits,
wall-time and GPU-hour ceilings in the launch manifest using smoke throughput.
Initial operational ceilings are 72 hours from the first full-set launch,
576 allocated GPU-hours total and 600 seconds per page including regional work.
These are maximum guardrails, not promised duration or expected cost. Never
evict unrelated workloads; checkpoint and report incomplete arms if a ceiling
prevents completion.

Measure full-denominator Overall, components, page failures, changed slots,
repair/regression counts, wall time, GPU seconds and peak memory. Charge baseline
work to composed systems even when cached. Save source/configuration hashes,
prediction manifests, progress and exit status separately for every arm.

### Which results warrant a leaderboard submission?

V5.1 is a control and should retain the upstream model identity. V5.2 establishes
whether an input adaptation itself provides an improvement. V5.3-V5.5 are
pipeline candidates; their value depends on net full-page gain and reproducibility.
A high score alone does not justify renaming upstream weights as a new trained
model. Attribute the backbone and reference mechanisms.

The [maintainer's response](https://github.com/opendatalab/OmniDocBench/issues/245#issuecomment-4910652298)
allows improved projects built on third-party parsing models to submit repository
links, self-test scores and Markdown predictions. Eligibility does not establish
acceptance or rank. Submit the strongest verified candidate after the official
requirements are satisfied, not every control or the most elaborate pipeline.
The leaderboard submission itself is a subsequent action, not part of launching
this experiment round.

### Fast-result and paper deliverables

The first deliverable is one reproducible strong system, one matched comparison,
and an explanation of which changes helped. The paper question follows the
remaining failure mechanism: for example, learning when to accept or reject
additional regional evidence if candidate quality exceeds selection quality.
A-E establish the system and diagnosis; assembling these known stages alone is
not claimed as sufficient novelty for a CCF A paper. Use external development
data and independent source-held-out evaluation for the subsequent research claim.

## Longer term research question

Can a parser decide whether, where and how to acquire additional visual evidence,
and whether to retain an update, using predicted repair benefit, regression risk
and measured cost, rather than a fixed image scale or unconditional replacement?

This is a candidate question, not a novelty claim or an approved architecture.
Start with table errors because they are a prominent gap in current results;
verify their importance again after establishing a stronger matched baseline.
Keep text and formula quality protected and evaluate the assembled full page.
Input scale is one possible action, not a permanent boundary on V5.

An initial bounded action set could include no change, a typed crop reread and
one alternative crop scale. Add structure-aware splitting or an alternative
parser only if failure analysis shows a missing capability. Record which
observation is available before and after each paid action. Do not use future
recognition outputs as supposedly free selection features.

## Novelty assessment before method development

The initial source screening already found close prior work:

- [DEC and TableParseMap](https://arxiv.org/abs/2608.09842) describe decomposition,
  enhanced views, visual-consistency gating, candidate ranking and rollback for
  frozen table parsers. Generic selective rereading plus a visual verifier is
  therefore insufficient as a novelty claim.
- [PaddleOCR-VL-1.6](https://arxiv.org/abs/2606.03264) describes targeted optimization
  of weak regions and progressive post-training. Targeting hard examples alone
  is also insufficient as a novelty claim.

These are an initial screen, not a completed literature review. Read complete
methods and available code, then compare decision timing, supervision, action
space, regression control, cost accounting and cross-source generalization.
Also cover adaptive resolution, coarse-to-fine parsing, selective prediction
and inference-time compute allocation. A possible distinction is learning and
calibrating action-specific incremental benefit and damage under a budget, but
its novelty and feasibility remain unverified. If it duplicates prior work,
revise the question before implementing a large system.

## Milestones and decision gates

| Stage | Work and deliverable | Gate |
|---|---|---|
| 0. Reconcile evidence | A comparison manifest for inputs, pixels, model revisions, backend, prompts, evaluator, renderer, timeouts and costs; a literature difference matrix | Separate comparable runs from historical context and identify a defensible research gap. |
| 1. Recover a strong control | Reuse verified saved predictions and run identities; select one recoverable strong baseline and fill only the evidence gaps needed for the next intervention | No automatic full baseline sweep. Add input-scale controls only if making an input-scale claim. |
| 2. Diagnose remaining errors | A small, source-stratified development pilot, initially 128–256 pages subject to resources; saved candidates, repair/regression counts and failure taxonomy | Determine whether existing actions can repair enough errors to justify routing. This pilot is diagnostic, not confirmatory. |
| 3. Test the method | Compare the simplest useful selector or verifier with fixed, random-at-matched-budget and simple-rule controls; ablate added components | Require meaningful net benefit over the strongest practical control, not merely over a weak default. |
| 4. Confirm generalization | Frozen evaluation on a held-out source set, a second compatible benchmark and at least two backbones as resources permit; full-page and component scores, costs and uncertainty | The main claim must survive new sources and matched controls. Limit portability claims if retraining is required. |
| 5. Release and submit | Fresh frozen full-set inference, independent replay, accessible dependencies, prediction manifests, official retest request and paper evidence package | Report official score only after acceptance; choose the venue after the contribution and evidence are clear. |

Stages are dependencies, not a promised calendar. Planned compute is eight
NVIDIA A100 80GB PCIe GPUs. Use cards for necessary independent shards and bounded ablations after evidence
reuse; hardware availability does not require a new baseline sweep. Set per-run GPU-hour
limits and stopping conditions, and verify the model environment before execution.
The target submission window remains open. Freeze a minimum meaningful accuracy effect, a cost limit
and a tolerated noninferiority margin using pilot variability and research
priorities before confirmatory testing. Do not invent a universal acceptance
threshold or treat a small pilot's confidence interval as sufficient power.

For true PDF rasterization, compare fixed 200/300 DPI and the selected rendering
policy from identical PDFs. For benchmark rasters, compare original pixels and
declared resampling actions; do not relabel interpolation as recovered source
detail. Record clipping, rotation, resolution caps and final visual grids.

## Separate candidate quality from decision quality

On development data, ground truth may measure an offline best-candidate upper
bound. It is a diagnostic only and must never select deployment/test predictions.
If even the candidate upper bound has little useful gain, a better router cannot
solve the problem: improve the available action or stop that route. If good
candidates exist but selection destroys the benefit, focus on selection and
verification. Full-page assembly can introduce additional damage and must be
scored after intervention.

Measure overall and component accuracy, repaired and damaged cases, abstention
and intervention coverage, timeout/empty counts, wall time, GPU seconds, peak
memory and actual model work. Include feature extraction, rendering, extra
recognition, verification and recovery in system cost. Separate cold-start from
steady-state measurements and keep hardware/concurrency comparable. Use paired
analysis with document/source grouping rather than treating repeated pages from
one document as independent evidence.

## Data and claim boundaries

OmniDocBench has already informed project and peer development. Mark it as an
exposed benchmark; a new split of previously inspected pages is not a blind test.
Train and tune on appropriately licensed external data, split by source document
and template, check overlap, and freeze the final independent test before use.
Use synthetic data as a supplement and validate on natural documents. Do not
train on OmniDocBench answers to manufacture the claimed generalization result.

No-change must remain a valid action. A purported safe verifier that accepts no
edits has not established a useful repair method. A confidence estimate does not
provide a universal guarantee under distribution shift. A faster method may
support an efficiency paper while failing the separate leaderboard objective,
because OmniDocBench Overall does not include inference cost.

## Repository deliverables

Preserve V0–V4 releases unchanged. Use `research/` for public scientific questions,
protocols, evidence maps and limitations; existing code roots for executable
implementation; `artifacts/` for publishable aggregate results; and `releases/`
for frozen reproducible candidates. Add new method code only when its scope is
settled. Avoid creating empty framework directories as a substitute for research.

Every experiment should bind a question, immutable configuration, model/data
identity, complete denominator, failure policy, cost record and analysis. Retain
negative runs. Public method documentation must contain the scientific details
needed for reproduction. Codex instructions, private reference copies, operational
scripts and chat records remain in ignored `codex/`. Group contributions require
clear attribution and publication permission before incorporation.

The next concrete deliverables are the baseline comparison manifest, the
prior-work difference matrix and a bounded diagnostic protocol. A paper outline
should follow a supported finding rather than predetermine the result.
