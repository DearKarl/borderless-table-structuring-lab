# V5 research proposal

Date: 2026-10-07. Status: proposal for discussion; no V5 experiment has run.

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

## Research question to investigate

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
| 1. Establish controls | Same-backbone default 200-DPI route, original-image route, strongest fixed candidate scale and existing V4 selector; add a documented simple reread control | Select a strong reproducible baseline. Record actual prepared pixels and processor grids, not DPI labels alone. |
| 2. Diagnose remaining errors | A small, source-stratified development pilot, initially 128–256 pages subject to resources; saved candidates, repair/regression counts and failure taxonomy | Determine whether existing actions can repair enough errors to justify routing. This pilot is diagnostic, not confirmatory. |
| 3. Test the method | Compare the simplest useful selector or verifier with fixed, random-at-matched-budget and simple-rule controls; ablate added components | Require meaningful net benefit over the strongest practical control, not merely over a weak default. |
| 4. Confirm generalization | Frozen evaluation on a held-out source set, a second compatible benchmark and at least two backbones as resources permit; full-page and component scores, costs and uncertainty | The main claim must survive new sources and matched controls. Limit portability claims if retraining is required. |
| 5. Release and submit | Fresh frozen full-set inference, independent replay, accessible dependencies, prediction manifests, official retest request and paper evidence package | Report official score only after acceptance; choose the venue after the contribution and evidence are clear. |

Stages are dependencies, not a promised calendar. Planned compute is eight
NVIDIA A100 80GB PCIe GPUs. Prioritize independent baseline and ablation jobs
across cards before adding distributed-training complexity. Set per-run GPU-hour
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
