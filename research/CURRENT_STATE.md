# Current research state

Updated 2026-10-07, Europe/London. **All engineering and scheduled monitoring are
paused at the researcher's request.** Preserve the completed evidence and discuss
the next study before any new implementation, acquisition, training or evaluation.
Repository publication does not restart work or allocate compute.

## Completed and incomplete studies

| Study | Status | Main finding |
|---|---|---|
| V0 to V4 | Historical results preserved | Different backbones and protocols; not a single controlled ranking |
| V5 | Five outputs, each 1651 pages, scored | Original RGB input explained most observed gain; independent native variation limits causal attribution |
| V6 | Six configurations plus shared D0, each 1651 pages; all eight metrics | Both table-only and combined arms regress; formula-only differences are small |
| V7 | One CPU GBDT, five OOF fits plus one final fit; four configs packaged | OOF selects infinity margin, so V7.4 keeps every native output |
| V7 independent confirmation | Paused before fresh OCR and quality evaluation | No independent V7 score, full benchmark score, or passed gate |
| Generator post-training | Discussion only | No chosen trainable checkpoint, frozen training protocol, model or allocated budget |

## V6 evidence

D0 Overall is97.574874. GBDT table/formula/combined are97.475604/97.582948/97.483678;
MLP counterparts are97.307734/97.592369/97.325229. All seven outputs retain1651pages,
zero inference failures and zero replays. Four CPU controllers serve six configs.
The source-page dataset contains276regions from242documents, split180/48/48.
Task cost was5.792213 allocated GPU-hours, including4.168781 benchmark GPU-hours.
All task-owned V6 compute exited. These are local measurements, not a rank or a
statistically established learned-over-fixed advantage.

See [the V6 report](../versions/v5_native/README.md),
[all metrics](../artifacts/results/v6/AGGREGATE.json),
[activation and input changes](../artifacts/results/v6/ACTIVATION.json),
[paired quality](../artifacts/results/v6/PAIRED_QUALITY.json), and
[costs and failures](../artifacts/results/v6/COSTS.json).

The [saved-candidate diagnosis](SAVED_CANDIDATE_DIAGNOSTIC.json) preserves the
training/development-only rationale for V7, including optimistic GT-assisted
envelopes, harm counts and source grouping. These are not deployable results.

## V7 evidence and prospective amendment

The112training tables represent112independent source documents. The34-feature
model's OOF native TEDS is0.905307; the best finite margin yields0.904448. The
infinite margin is a valid negative result, not a reason to force replacements.
Six CPU fit timers total0.187seconds, excluding preparation and packaging.
An integer-serialization error was recovered from the five saved fold models;
no completed fold was refitted. Four configurations share one final new model.

The already inspected29-table development set gives V7.1/V7.2/V7.3/V7.4 TEDS
0.891977/0.899530/0.907530/0.891977. It previously informed fixed-scale selection,
so it does not establish generalization. V7.2/3/4 use the same fixed candidate at
scale0.535299427146522; decisions occur after OCR and do not save its call.

Before fresh confirmation quality was generated or read, V7.3 replaced V7.4 as
the sole nominated candidate. The preserved gate requires128complete new source
documents, mean V7.3 gain>=0.005 over max(V7.1,V7.2), and positive paired95%
bootstrap lower bounds against both controls, with2000draws and seed0. This
amendment did not mean the original V7.4 gate passed. Work is now paused; neither
gate authorizes continuation without a new researcher decision.

See [V7 methods and negative result](../versions/v7/README.md),
[training](../artifacts/results/v7/TRAINING.json),
[development](../artifacts/results/v7/DEVELOPMENT.json), and
[frozen confirmation analysis](../artifacts/results/v7/CONFIRMATION_ANALYSIS_PROTOCOL.json).

## Baseline comparability

Local D0 97.574874 is not a verified reproduction of the public96.91 TeleOCR row.
Formula CDM explains almost all the arithmetic difference; its cause remains
unknown. Model, inputs and evaluator execution identity are not fully bound for
the public row. No local-minus-public difference is claimed as our contribution.
The [audit](../artifacts/results/v7/BASELINE_COMPARABILITY.md) freezes the matched
local comparator and preserves all eight fields and component denominators.

## Next direction remains under discussion

The earlier [spatial evidence design](proposals/V7_SPATIAL_EVIDENCE.md) adds six
geometric features to the34-feature GBDT. It is archived, unimplemented, and
different from the later generator proposal. Its [300-training-document plan](proposals/V7_TRAINING_EXPANSION.json)
also requires64development and128new confirmation documents; none of that work
was launched. No budget is allocated by retaining these designs.

The later proposal would directly post-train a pretrained image-to-table
generator on controlled image/structure pairs, comparing ordinary versus targeted
supervision at matched sample and compute budgets. Compatibility, novelty,
resource requirements and real held-out gains remain unverified. See the
[literature evidence and proposed controls](LITERATURE_SCAN_20261007.json).
Generic visual grounding was already tried in Explicit-v2; it is not a new
contribution merely because it is renamed. Neither LoRA nor synthesis alone
establishes novelty. The next discussion must choose a concrete falsifiable
mechanism and feasible training protocol before execution.

Future constraints remain200DPI, default native inputs, no original-RGB bypass,
test isolation and exclusion of physicalGPU0. Regional pixel scale, actual model
input, final output changes and measured quality gains must remain distinct.

## Historical evidence

[Results](RESULTS.md), [V5](../versions/v5/README.md), [V4](V4.md),
[Explicit-v2](../training/README.md), and [sidecar](../hybrid/README.md) preserve
earlier results and their limitations. Historical proposals are retained as
design history, not current execution instructions. Raw datasets, predictions,
private runtime bindings and operational records remain local; public source,
first-party controller weights and aggregate evidence are archived separately.
