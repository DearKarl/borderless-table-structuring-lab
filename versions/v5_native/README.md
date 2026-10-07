# V6 direct regional scale controllers

Four CPU-trained controllers and six load-verified configurations are available in [the V6 package](../../releases/v6-v7-progress-20261007/models/v6/PACKAGE_MANIFEST.json). External controller-selected OCR validation and all seven full benchmark outputs are complete. Each output retains 1651 pages and all eight official-protocol metrics. Results establish measured local differences, without an official rank or novelty claim.

| Configuration | Controller | Intervention |
|---|---|---|
| GBDT_V6.1.1 | GBDT | Tables |
| GBDT_V6.1.2 | GBDT | Formulas |
| GBDT_V6.1.3 | GBDT | Tables and formulas |
| MLP_V6.2.1 | MLP | Tables |
| MLP_V6.2.2 | MLP | Formulas |
| MLP_V6.2.3 | MLP | Tables and formulas |

The configurations share four task/family checkpoints. D0 is a shared default TeleOCR baseline. Each benchmark page has one native parse; the six derived outputs reuse it. Combined configurations reuse their family's component decisions and candidates. These are paired shared-computation outputs, not seven independent fresh runs.

## Complete benchmark results

The [aggregate](../../artifacts/results/v6/AGGREGATE.json) retains exact values, all official category breakdowns, diagnostics and deltas from the shared D0. Overall is on 0-100; all other metrics below are on 0-1. Overall = 100 * ((1 - Text Edit) + Formula CDM + Table TEDS) / 3. Lower edit distance is better; higher CDM/TEDS is better. No metric is missing.

| Output | Overall | Delta vs D0 | Text Edit | Formula CDM | Table TEDS |
|---|---:|---:|---:|---:|---:|
| D0 | 97.574874 | +0.000000 | 0.027098 | 0.986026 | 0.968319 |
| GBDT_V6.1.1 | 97.475604 | -0.099270 | 0.027098 | 0.986026 | 0.965341 |
| GBDT_V6.1.2 | 97.582948 | +0.008074 | 0.027082 | 0.986252 | 0.968319 |
| GBDT_V6.1.3 | 97.483678 | -0.091196 | 0.027082 | 0.986252 | 0.965341 |
| MLP_V6.2.1 | 97.307734 | -0.267140 | 0.027098 | 0.986026 | 0.960304 |
| MLP_V6.2.2 | 97.592369 | +0.017495 | 0.026989 | 0.986441 | 0.968319 |
| MLP_V6.2.3 | 97.325229 | -0.249645 | 0.026989 | 0.986441 | 0.960304 |

| Output | Formula Edit | Table TEDS-S | Table Edit | Reading Order Edit |
|---|---:|---:|---:|---:|
| D0 | 0.049005 | 0.981673 | 0.035843 | 0.118297 |
| GBDT_V6.1.1 | 0.049005 | 0.977724 | 0.038683 | 0.118297 |
| GBDT_V6.1.2 | 0.049068 | 0.981673 | 0.035843 | 0.118370 |
| GBDT_V6.1.3 | 0.049057 | 0.977724 | 0.038683 | 0.118370 |
| MLP_V6.2.1 | 0.049005 | 0.974191 | 0.040142 | 0.118297 |
| MLP_V6.2.2 | 0.048322 | 0.981673 | 0.035843 | 0.118214 |
| MLP_V6.2.3 | 0.048322 | 0.974191 | 0.040142 | 0.118214 |

MLP_V6.2.2 has the highest observed Overall, 97.592369, a +0.017495-point difference from D0. Both table-only and both combined configurations regress. The formula-only differences are small. These results do not establish a learned-scale advantage over a matched fixed reread, statistical significance, or a continuity effect. No arm was refitted or selected for further benchmark execution.

## Output edits and observed quality

The [paired quality report](../../artifacts/results/v6/PAIRED_QUALITY.json) retains complete official-sample, page, all-eligible-region and applied-edit outcomes. The following funnel describes only applied canonical output edits. Each cell gives **regions / unique pages**. Table changes use TEDS and formula changes use CDM; their combined counts are diagnostic and are not an Overall metric. A page can contain multiple outcome types, so page columns overlap and must not be summed.

| Version | Edited regions | Improved | Worsened | Tied | Unmatchable |
|---|---:|---:|---:|---:|---:|
| GBDT_V6.1.1 | 191 | 56 / 51 | 112 / 87 | 20 / 20 | 3 / 3 |
| GBDT_V6.1.2 | 339 | 22 / 20 | 45 / 34 | 208 / 121 | 64 / 40 |
| GBDT_V6.1.3 | 529 | 78 / 70 | 157 / 121 | 227 / 140 | 67 / 43 |
| MLP_V6.2.1 | 185 | 55 / 50 | 105 / 77 | 23 / 22 | 2 / 2 |
| MLP_V6.2.2 | 368 | 28 / 24 | 42 / 34 | 206 / 125 | 92 / 55 |
| MLP_V6.2.3 | 553 | 83 / 74 | 147 / 111 | 229 / 147 | 94 / 57 |

Region correspondence is conservative: a unique canonical native prediction must match a D0 scored unit, the version must preserve the same exact GT identity, and its final text must equal the expected retained region. Ambiguous, split or mixed units stay unmatchable. Ties use tolerance 1e-6. These counts do not determine the official mean: effect magnitudes and official page weighting also matter. No-match coverage is retained rather than treated as success or dropped silently.

All page-level outcome tables retain the 1651-page denominator. Component-absent or unpairable pages are explicitly unmatchable. Diagnostic per-page CDM/TEDS are means of emitted sample metrics; no per-page Overall is fabricated. The frozen benchmark metadata has no authoritative source-document grouping, so no document-cluster confidence interval is claimed and no independence-assuming page bootstrap is substituted. External source-document bootstrap intervals are reported separately.

The [evaluation receipt](../../releases/v6-v7-progress-20261007/models/v6/EVALUATION_RECEIPT.json) connects the completed results to unchanged checkpoint, configuration and runtime-source hashes. The deployed configs and package manifest retain their original pre-evaluation status text as frozen provenance.

## Input and prediction contract

[native.py](native.py) captures the image returned by unchanged TeleOCR `read_fn`/`do_parse`, with 200-DPI rendering and the native 3500-pixel cap (renderer rounding can produce 3501 pixels). Crops use those captured pixels and the native integer coordinate frame. There is no original RGB shortcut, source rerendering, cap increase, or formula 200/72 multiplier. Rendering DPI stays 200; regional pixel resize scale is learned.

[direct_controller.py](direct_controller.py) predicts one log-scale from the same 17 native-region features for both families, clips log-scale to log([0.5, 3]), exponentiates, and rounds pixel dimensions. There is no deployed scale menu, candidate loop, snapping, gain head, or OCR search. Nonfinite features/predictions or more than 64 million requested pixels retain native content. Each region receives at most one additional OCR request per family. Equal actual inputs can share OCR when pixel tensor, image-grid and prompt identities match.

[actor.py](actor.py) checks preview preprocessing against actual OCR tensors. The change reference is the same region's 1x preprocessing; equality with its earlier native page extraction input is not assumed. Unknown comparisons remain unknown. A changed nominal scale or output is not proof of improved recognition.

[assembly.py](assembly.py) derives each output from immutable native content. Accepted changes compete by table priority, larger area, then native order; overlap suppressions are recorded. Structural rejection retains native. The old multi-attempt guard is inactive. Failures retain each output's last atomic stage and remain in denominators. Completed or failed pages are not replayed.

## Frozen external data

The pool contains 276 reliably matched native predicted regions from 242 source documents: 110 formulas and 166 tables. It was built from 590 selected public examples, retaining a smaller reliable pool than the initial 3072-region target. Source-PDF correspondence and labels are offline; GT boxes/text never select inference regions or features.

| Split | Formulas | Tables | Regions | Documents |
|---|---:|---:|---:|---:|
| Train | 68 | 112 | 180 | 159 |
| Validation | 19 | 29 | 48 | 42 |
| Test | 23 | 25 | 48 | 41 |

Document groups are disjoint. Pilot split roles were preserved; new groups used a frozen source-hash assignment. Contribution was capped at four regions/source (actual maximum three). No duplicate native pixels or native units were found. Crop hash/near-duplicate and known benchmark source-ID checks do not prove contamination freedom.

Sources are [Image2Struct equations](https://huggingface.co/datasets/stanford-crfm/i2s-latex), revision `d889f4a4bed5d6394a5966e35f37686fae4070a2`, and [Docling PubTabNet-OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL), revision `c6b519724c09c81dc4797b742826681ebf5c7dde`. Equations use the 300-record published validation split, repartitioned by document as disclosed here; published test data was not used. Tables were sampled across shards 0, 17 and 34, using original HTML/cell labels. These scientific-paper sources do not establish multilingual, scanned, financial or general-document coverage. Images, labels, raw predictions and deployment bindings remain outside public artifacts.

At most eight offline realized views supplied one best valid measured log-scale target per region, using TEDS for tables and CDM for formulas. Differences within 1e-6 tie; choose the measured scale closest to 1, then smaller. This is a sampled empirical optimum, not a certified continuous/global optimum. All 182 non-beneficial examples remain, with one training weight each. In 222 regions, multiple distinct measured inputs tied. All 276 regions obtained valid targets. Of 2208 planned views, 2207 completed with verified actual tensor identities; one over-context view failed without replay. Its other valid views and failure record were retained.

Fixed external controls were selected by common-coverage validation macro quality over the same offline scales: table 0.535299427146522, formula 1.25. Test quality did not select them. The 96 validation/test regions received 288 actual learned/fixed actions: 192 controller predictions, 286 physical OCR requests and two identical-input reuses, with all actual tensor identities verified and no failed actions.

The [external quality report](../../artifacts/results/v6/EXTERNAL_ACTOR_QUALITY.json) preserves these held-out results (0-1 scale):

| Test subset | Native | GBDT | MLP | Validation-selected fixed |
|---|---:|---:|---:|---:|
| 25 tables, TEDS | 0.888638 | 0.883552 | 0.892418 | 0.909553 |
| 23 formulas, CDM | 0.987609 | 0.988130 | 0.983000 | 0.990957 |

Fixed-scale means exceeded both learned families. Most paired 95% source-document bootstrap intervals included zero; the MLP-minus-fixed formula interval was negative. These are small, exploratory comparisons without multiplicity correction. The report uses 2000 source-document resamples, seed 0, separately by split and task. No model, fixed scale, benchmark version or stopping decision was selected from these test results.

## Training and use

[direct_train.py](direct_train.py) fits log-scale squared error on matched rows, weights, features and training-only normalization. GBDT uses constant-leaf HistGradientBoosting: 100 iterations, maximum 15 leaves, minimum 20 samples/leaf, learning rate 0.05, L2=1, seed 0, no early stopping. The MLP has 64/32 ReLU layers and a scalar linear head; NumPy float64 CPU Adam, learning rate 0.001, weight decay 0.0001, batch 128, seed 0, maximum 200 epochs and validation patience 20. Target normalization uses training only.

The table MLP selected epoch 9 and stopped at 29; the formula MLP selected epoch 6 and stopped at 26. The measured training process took about 0.19 seconds on this small dataset; its four per-fit timers sum to 0.14 seconds. Startup and packaging are excluded. No OCR weights were trained. Exported checkpoints exactly reproduced in-memory validation predictions; this validates loading, not OCR quality.

The [package manifest](../../releases/v6-v7-progress-20261007/models/v6/PACKAGE_MANIFEST.json) records checkpoint/data hashes, split counts and load checks. Each named configuration includes source hashes, feature schema, dependencies and invocation. A JSON array of 17 native features can be evaluated with:

```text
python -m versions.v5_native.predict --package releases/v6-v7-progress-20261007/models/v6/GBDT_V6.1.1.json --kind table --features native_features.json --width 100 --height 40
```

[shared_worker.py](shared_worker.py) is the bounded OCR execution entry. [external_actor_worker.py](external_actor_worker.py) runs actual learned/fixed actions on saved native validation/test regions. Historical response-model code in `controller.py`, `train.py` and parts of `matched_train.py` remains inactive for V6 deployment; the latter's tested optimizer/export helpers support direct training.

The [frozen runtime source archive](../../releases/v6-v7-progress-20261007/models/v6/FROZEN_RUNTIME_SOURCE.tar.gz) preserves the exact deployed source bytes; its [inventory](../../releases/v6-v7-progress-20261007/models/v6/FROZEN_RUNTIME_SOURCE.json) matches every source hash in the six configurations. Later reporting-only changes are separate from that frozen inference implementation.

## Measurement and limits

The full comparison completed 1651 pages per output: 9906 adaptive arm-page outputs, or 11557 including D0, from 1651 distinct pages. All pages completed without failure or replay. Sixteen deterministic technical-smoke pages contribute unchanged outputs; only the remaining 1635 pages were subsequently processed. Smoke acceptance checked runtime, lineage and assembly without a quality gate. All 9906 adaptive assemblies were subsequently verified against saved native content and frozen candidates without additional OCR or GT access.

[evaluate.py](evaluate.py) retains Overall, Text Edit, Formula CDM, Formula Edit, Table TEDS, Table TEDS-S, Table Edit and Reading Order Edit where the pinned official evaluator emits them, plus category breakdowns. Missing fields are null with reasons. Activation reporting separates predict API calls/rows, nominal scale, requested pixels, verified tensors, physical/reused OCR, accepted edits and offline quality effects. Actual shared costs and estimated standalone costs are labeled separately.

The [activation report](../../artifacts/results/v6/ACTIVATION.json) distinguishes regional counts from unique page unions. All page counts below use 1651 as the denominator. Input changes compare actual regional preprocessing against the same crop at 1x; they do not assert equality to an earlier native page recognition input.

| Version | Evaluated regions | Eligible pages | Changed-input regions / pages | Applied canonical edits / pages |
|---|---:|---:|---:|---:|
| GBDT_V6.1.1 | 670 | 462 | 666 / 461 | 191 / 150 |
| GBDT_V6.1.2 | 2371 | 324 | 2242 / 318 | 339 / 172 |
| GBDT_V6.1.3 | 3041 | 757 | 2908 / 750 | 529 / 319 |
| MLP_V6.2.1 | 670 | 462 | 667 / 462 | 185 / 144 |
| MLP_V6.2.2 | 2371 | 324 | 2343 / 323 | 368 / 183 |
| MLP_V6.2.3 | 3041 | 757 | 3010 / 757 | 553 / 327 |

Each family physically predicted 3041 rows in 3041 one-row API calls. These are not summed again across its three configurations. GBDT executed 3041 regional OCR requests; MLP executed 3026 and reused 15 identical inputs from GBDT. All 6082 actions have verified actual tensor identities, including reused actions, and all passed structural checks. D0 had 1651 native page parses with 36793 internal model requests. Native detection recall is not established by eligible-region counts.

The full per-page input/output identity ledger is preserved locally; its hash and publication boundary are recorded in the public archive. The published activation aggregates preserve actual-change counts and comparison limits. There were no boundary-clipped actions. Realized native image edges reached 3501 pixels because of renderer rounding around the unchanged 3500 cap; rendering stayed at 200 DPI for all pages.

Shared benchmark inference, including retained smoke and engine startup, consumed 4.168781 allocated GPU-hours. Total task GPU usage including preparation, source collection, scale measurements and external actions was 5.792213 GPU-hours. Maximum concurrent benchmark GPUs: six A100 80GB PCIe cards, BF16. Successful page latency was 6.310 seconds median, 15.329 seconds p90 and 20.717 seconds p95; there were no failed-page latency observations. The sum of native stage times was 9175.422 seconds across workers, and controller prediction totals were 0.880 seconds for GBDT and 1.629 seconds for MLP. These overlapping totals are not elapsed wall time. All seven official CPU scorers completed successfully. [Costs](../../artifacts/results/v6/COSTS.json) records each scorer's start, finish and measured duration; at most three ran concurrently, each limited to eight CPUs and 32 GiB. All task-owned compute has exited.

TeleOCR source is pinned to `9921cffe380efe4e2fa010258b3d0c3cb70bab2d`; [runtime provenance](../../artifacts/results/v6/RUNTIME_PROVENANCE.json) records the 14 actual OCR asset hashes, dependency versions, prompts, sampling and engine settings verified across eight workers. The requested upstream model revision remains unverified. Structural checks/reference mechanisms derive from [ki-OCR-v1](https://github.com/yushuosun/ki-OCR-v1), commit `ac1ab4f5ded8b025f47a7683f58e65db4ac43a2b`, through preserved V5 code. The official evaluator is pinned to `147cd5ac9472002f5751221d390bf00abdbc0d2f`, configuration hash `56db52659910273376223a5646eabfaff21b25dc30f434e1c90aab1a233d335a`.

GBDT and MLP differ in capacity, optimization and regularization as well as continuity. This comparison cannot isolate continuity as the cause of a difference; routing, pixel rounding and OCR are outside any continuity claim. A fixed-scale end-to-end counterfactual would be needed to isolate learned-scale benefit from rereading alone. Existing external fixed controls provide source-domain evidence only. Official-protocol local scoring is not an official leaderboard submission.
