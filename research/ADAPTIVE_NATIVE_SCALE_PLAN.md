# Learned regional scale control on the default TeleOCR input path

Date: 2026-10-07. Status: design revision; no new experiment executed.

## New constraint and scope

The researcher relayed the supervisor's requirement to remove direct original-RGB
input from the next study and use TeleOCR's default input path and native output.
This supersedes original-image assumptions in the earlier candidate-selection
and distillation proposals for future work. Historical V5 source and results
remain unchanged.

Use the existing V5.1 `read_fn` / `do_parse` route, its pinned 200-DPI rendering,
3500-pixel long-edge cap, model files, prompts and native settings as the
reference. Initial regional interventions must crop the actual native-rendered
page image, with audited coordinates. Do not read the original PNG again for
regional crops or enhanced detail. Do not change PDF rendering DPI, raise caps,
add super-resolution or introduce source-PDF rereading in this first study.

The strict interpretation is a default native parse followed by optional
adaptive regional rereading. The controller is post-native but pre-reread; it is
not a claim of choosing a page's resolution before any recognition. The original
input must of course still enter the default loader; bypassing that loader is
what is removed.

## What is learned

Replace the three-action scale menu with a model of region-specific quality
response to a numeric scale or target pixel budget. Start with GBDT as a tractable
baseline; do not claim that the algorithm itself is new.

Let x contain features available after the default native parse: native region
geometry, estimated glyph height/reliability, image density, blur, contrast,
native output length/repetition/structure, and the processor geometry achievable
at a proposed scale. Fit g(x, log(s)) to measured regional quality gain relative
to retaining the native output. Separate formula and table quality targets or
models so their calibration is explicit. Exclude reference answers, page/source
identities and unobserved reread results from deployment features.

At inference, numerically maximize predicted gain within a frozen feasible scale
range and visual-token budget. Retain the native output when no feasible scale
clears a validation-frozen gain margin. Otherwise perform one additional regional
recognition at the selected scale. Compare resulting inputs with the native
recognition input when available and skip equivalent actions. Syntax-invalid
outputs retain native content. Apply the same acceptance rule in all controls.

This can select numeric values beyond hand-written recipe labels. However, GBDT
is piecewise constant, bounded numeric search is finite, image sizes are integers,
and visual grids are quantized. Describe it as learned numerical scale control,
not an infinitely continuous optimum or a guarantee of new visual information.
Optimizing the surrogate is not the same as finding the true OCR optimum.

An alternative baseline directly regresses log(scale) from region features.
Do not assume this is superior: averaging several good but separated training
scales can produce a poor intermediate scale. Surrogate response modeling makes
the measured scale-quality relationship explicit and is the preferred first test.

## Training evidence

Use source-separated external labeled data and the default TeleOCR input chain
for all new baseline/teacher generation. Old V5.5 original-RGB predictions are
historical diagnostic evidence, not the matched teacher or new training labels.
Never train or select parameters on OmniDocBench images, predictions or answers.

On a bounded external development population, sample scales broadly within
geometry-derived feasible limits and log the realized pixels, processor tensors
or hashes, grid, quality and cost. Include native/no-change and fixed control
scales as anchors; use log-space random or stratified samples beyond those
anchors to avoid turning the dataset into only a three-label classifier.

Training-time sampling is necessary supervision, not the deployed action menu.
Generalization between sampled scales is a hypothesis to test, not a property
guaranteed by GBDT. Reserve unseen regions/documents and some numerical scale
settings for evaluating prediction error, action quality and calibration.
Deduplicate exact effective inputs before spending recognition calls.

First diagnostic target: 128 external regions balanced between formulas and
tables, sampled without selecting for favorable intervention results. A ceiling
of eight distinct realized views per region bounds this mapping probe at 1024
regional requests, plus the separately counted baseline requests. This is a
feasibility probe, not enough evidence for a generalization claim. Source/data
availability and range bounds must be established before execution.

If quality differs meaningfully across realized views and region-dependent
choices have room over the strongest fixed policy, expand the external training
pool. If most views alias, first resolve the native preprocessing boundary. If
the development best-candidate ceiling is weak, report limited headroom instead
of forcing selector activity. Training costs and expansion budgets follow the
measured probe, not an assumed throughput.

## Controls and causal attribution

| ID | Default-path configuration | Purpose |
|---|---|---|
| N0 | Unmodified default TeleOCR | Required native baseline |
| N1 | Fixed regional combination, with crops from native-rendered pages | Re-establish a same-input composition reference; not historical V5.5 |
| N2 | Strongest validation-selected fixed scale under the same intervention/request budget | Test whether adaptation is necessary |
| N3 | GBDT selecting from a small fixed scale menu | Separate learned selection from numeric action freedom |
| N4 | Learned numerical scale controller | Proposed adaptive method |

For N2-N4, hold region eligibility, candidate acceptance, native parse, processor
limits, reread allowance and assembly constant. Where no-change selection
changes actual spend, report both request-capped quality and matched-cost
comparisons. Add random/shuffled-policy diagnostics if needed; do not claim
learning benefit from a comparison with N0 alone. N1 has a different operation
sequence and must be reported as an end-to-end system reference.

No hard-coded formula-only 200/72 multiplier is silently inherited by N4.
Any conversion required for coordinates must be independent of the learned
resize operation. The old fixed multipliers may appear in the explicit N1
reference only, with their actual pixel effects recorded.

## Required evidence and interpretation

Record estimator calls, requested scales, realized dimensions, image hashes,
processor tensor identity where feasible, visual grids, cap/alias frequency,
actual extra requests, changed outputs and paired quality gains/regressions.
A new scale value or grid size alone does not prove useful information or gain.
Unchanged grid dimensions alone also do not prove identical image/tensor content.
Counts must flow through to the final assembled prediction and full denominator.

Repeat checks of native variability and use shared native outputs for diagnostic
comparisons. Freeze final settings before full-page confirmation; record failed
and empty pages, cost, all components and source-grouped uncertainty. The final
system requires a fresh reproducible end-to-end confirmation after cached
diagnostics. It has not run yet.

Historical measurements: V5.1 default input scored 97.568231; V5.2 original RGB
scored 98.173146; V5.5 original RGB plus interventions scored 98.347514.
The latter is 0.779284 points above the default-path reference. This arithmetic
gap is not a forecast for N4. Do not subtract the historical 0.604916 input-path
difference from V5.5 and present the result as a measured native combination.
Scale/intervention interactions and independent native variability matter.

Working expectation: some regional scales may improve perception, but simple
upsampling cannot recover details removed by native preprocessing. A score near
98.35 is uncertain under the new constraint. First test gains over N0 and the
strongest matched fixed policy. Quality and efficiency may trade off; keep the
leaderboard objective distinct from efficiency findings.

## Prior work and relation to earlier plans

[ResAdapt](https://arxiv.org/abs/2603.28610) already learns pre-encoding visual
budget allocation with an unchanged backbone for multimodal tasks.
[FocusVTC](https://arxiv.org/abs/2609.36651) studies learned selective resolution
enhancement for visual text compression. These are adjacent tasks, not direct
OmniDocBench replications, but make generic adaptive-resolution novelty claims
insufficient. A detailed method comparison remains required.

The output-selection and distillation plans remain archived design options.
If resumed, their training and test inputs must be rebuilt under this default-path
constraint. The immediate study concerns learned pre-reread scale control,
with a project-trained policy and effective-input auditing. No new model,
GPU launch, leaderboard submission or GitHub publication is claimed here.
