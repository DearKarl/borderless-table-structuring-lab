# V7: generator pilots and the legacy selector

[Version index](../README.md) · [Project](../../README.md)

Two different research lines historically used the V7 name. They are separated here; matching labels do not mean matching models or repeated evidence. The later large campaign's old **V7.3.0/V7.3.1/V7.3.2** names are aliases for [V8.1/V8.2/V8.3](../V8/README.md), not additional V7 runs.

## Generator V7.1 / V7.2: ordinary and targeted LoRA

Two rank-8 attention-LoRA fits start independently from the same TeleOCR base. Both use 34 real TRAIN sources, eight synthetic donors forming 32 controlled table prototypes, 132 exposures and 33 optimizer updates. Ordinary SFT repeats one relation state; targeted SFT sees both states. The base and vision stay frozen. Eight DEV sources were reserved; fixed final-update checkpoints were evaluated on 64 independent confirmation sources using matched default Transformers processing. Real inputs are predicted native crops at PDF 200 DPI; synthetic inputs use complete rendered tables and default recognition preprocessing.

Every policy retains 64 sources. Metrics below use 0–1 except counts:

| Policy | TEDS ↑ | Structure TEDS ↑ | Typography-stripped TEDS ↑ | Cell exact ↑ | Character distance ↓ | Malformed | Content damage vs native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matched native baseline | 0.891100 | 0.933341 | 0.894532 | 0.784049 | 0.033078 | 0 | 0 |
| Legacy simple rule | 0.902616 | 0.931999 | 0.909881 | 0.839004 | 0.020279 | 0 | 13 |
| V7.1 ordinary LoRA | 0.890519 | 0.932313 | 0.894783 | 0.787277 | 0.032666 | 0 | 5 |
| V7.2 targeted LoRA | 0.890484 | 0.932313 | 0.894739 | 0.787169 | 0.032886 | 0 | 4 |

The prespecified gate failed. Targeted minus ordinary mean TEDS is -0.00003515, with paired percentile 95% interval [-0.00011661, +0.00001289] from 2,000 source resamples (seed 0); gains/harms/ties are 1/2/61. Both trained arms trail native and the fixed legacy rule in mean TEDS. These intervals are descriptive, not an equivalence claim. Total allocation was 2.384904 GPU-hours. No full-page Overall or other official whole-document score was measured for these pilots.

[Full-precision four-policy metrics, all five paired contrasts and gate](generator/RESULTS.json) and [complete pilot report](generator/APPENDIX.md#pilot) retain the remaining inference, training and cost measurements.

## Follow-up diagnostics and V7.2.1 / V7.2.2

| Study | What changed or was checked | Result and interpretation | Complete reported metrics |
| --- | --- | --- | --- |
| Post-pilot diagnosis | Adapter loading, matched conditional losses, image swaps and greedy structures | Targeted/ordinary structures identical on 64/64 confirmation sources; all three checkpoints 0/32 exact prototype structures; lower loss did not solve free generation | [Diagnosis](generator/APPENDIX.md#diagnosis) |
| V7.2.1 | Increase synthetic empty/span-token weighting; same 33 updates and 132 exposures | 0/16 both-state pairs and 0/32 exact structures; branch NLL falls but fixed gate fails | [All endpoint and NLL tables](generator/APPENDIX.md#structural-loss) |
| Identifiability audit | Representation round-trips and bounded rendering alternatives | 74/74 canonical targets pass interface checks; 0/32 conflicting duplicate-image labels; padding-aware alternatives show convention dependence, not a proven optimization cause | [Audit and representation metrics](generator/APPENDIX.md#identifiability) |
| V7.2.2 original input gate | Eight ruled and eight paired borderless fixtures | Detector finds 8/8 ruled and 0/8 borderless tables; this stage stops before fitting, not a failed learning score | [Input accounting](generator/APPENDIX.md#native-input-gate) |
| V7.2.2 ruled-only continuation | One 200-update fit, 800 exposures on eight valid native crops | Base and checkpoints 25/50/100/200 all score 8/8 exact and TEDS 1; final endpoint passes with zero learned gain | [All five checkpoints and conditional losses](generator/APPENDIX.md#ruled-only) |
| Real-TRAIN inventory | Original-base generation and visual review on 34 sources | 8/34 canonical exact, 25/34 structure exact; one repair candidate, eight retention cases, 25 excluded for supervision uncertainty; no new fit | [Scores, dispositions and cost](generator/APPENDIX.md#real-TRAIN-inventory) |

Seen-fixture success does not establish borderless transfer. Borderless recognizer outputs in V7.2.2 remain unavailable, not zero-accuracy predictions. All losses, exactness counts, control results, costs and original stage limitations remain accessible in the linked appendix. No follow-up supplies a 1,651-page generator benchmark.

## Legacy V7 selector: a separate keep/replace study

This earlier line trained one CPU GBDT on 112 table regions from 112 documents. Thirty-four features describe geometry, structure and native/candidate differences. Five grouped OOF fits select the acceptance margin. The fixed reread scale is 0.535299427146522; candidate rejection occurs after OCR and does not save its call.

| Legacy configuration | Policy | Already-exposed 29-table DEV TEDS |
| --- | --- | --- |
| V7.1 | Keep native | 0.891977 |
| V7.2 | Common-check fixed reread | 0.899530 |
| V7.3 | Simple structure-rule acceptance | 0.907530 |
| V7.4 | GBDT acceptance with selected infinite margin | 0.891977 |

OOF native TEDS was 0.905307 versus 0.904448 at the best finite margin. The selected infinite margin keeps every native output. Six fit timers total about 0.187 CPU seconds. The prospective nominee later changed to the simple rule before fresh quality; confirmation paused with no fresh quality scores or full benchmark. Download counts are not completed confirmation samples. DEV values do not establish unbiased generalization.

Implementation and evidence: [legacy selector method/source](source/v7/README.md), [training/OOF results](results/legacy_selector/TRAINING.json), [complete DEV results](results/legacy_selector/DEVELOPMENT.json), [stopped state](results/legacy_selector/STOPPED_STATE.json) and [frozen model package](../V6/archive/v6-v7-progress-20261007/models/v7/PACKAGE_MANIFEST.json).

## Source availability and naming

The historical pilot controllers and adapter packages are retained locally and are not newly published by this documentation task. The public [shared supervision implementation](../V8/source/generator_pilot/supervision.py) is available; it is not presented as the complete frozen pilot runtime. The appendix and JSON bind the original report/result hashes and exclude prediction bodies, images, labels and private infrastructure. Current large-model code is documented separately in [V8](../V8/README.md).
