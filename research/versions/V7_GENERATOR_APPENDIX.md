# V7 generator: historical result appendix

This documentation-only consolidation preserves the complete numeric tables and interpretation of seven existing reports, in study order. Historical stop/planning statements describe their original stage; the ruled-only continuation follows the earlier native-input blocker. No inference, training or scoring was repeated.

[Concise V7 introduction](V7.md) · [Full-precision pilot results](V7_GENERATOR_RESULTS.json)

Source report hashes below bind the original local artifacts. Links to already published evidence remain active; supporting records not in the public snapshot are marked as retained locally. No private data, images, labels, predictions or deployment files are included.

<a id="pilot"></a>

Source report SHA-256: `ad9c5cf468d828bdc7f58d9421a2314d0d992b470f722e148e30cf933b9731de`.

## V7 generator fine-tuning pilot

Completed 64 independent source documents per policy. The prespecified pilot gate **did not pass**. Targeted minus ordinary SFT mean TEDS was -0.000035 (paired bootstrap 95% CI [-0.000117, +0.000013]).

This is a new matched Transformers comparison on real native table crops. Its numbers are not equivalent to historical vLLM benchmark scores. No full 1,651-page evaluation, leaderboard rank or novelty claim follows.

### Four-policy results

All metrics are on a 0–1 scale. Higher TEDS and cell exact match are better; lower character distance is better. Every policy retains all 64 sources. Missing or malformed outputs receive zero TEDS and cell exact match, and character distance 1.

| Policy | TEDS | Structure TEDS | Typography-stripped TEDS | Cell exact match | Character distance | Malformed | Content damage vs native |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 0.891100 | 0.933341 | 0.894532 | 0.784049 | 0.033078 | 0 | 0 |
| Legacy simple rule | 0.902616 | 0.931999 | 0.909881 | 0.839004 | 0.020279 | 0 | 13 |
| V7.1 ordinary SFT | 0.890519 | 0.932313 | 0.894783 | 0.787277 | 0.032666 | 0 | 5 |
| V7.2 targeted SFT | 0.890484 | 0.932313 | 0.894739 | 0.787169 | 0.032886 | 0 | 4 |

### Paired comparisons

Equal source weights; 2,000 paired source bootstrap resamples, seed 0, percentile 95% intervals. Gain/harm/tie uses a TEDS tolerance of 1e-9.

| Left minus right | Mean TEDS difference | 95% CI | Gain / harm / tie |
|---|---:|---|---:|
| V7.2 targeted SFT − V7.1 ordinary SFT | -0.000035 | [-0.000117, +0.000013] | 1 / 2 / 61 |
| V7.2 targeted SFT − Native | -0.000615 | [-0.008792, +0.007633] | 7 / 4 / 53 |
| V7.2 targeted SFT − Legacy simple rule | -0.012132 | [-0.027149, +0.000880] | 16 / 21 / 27 |
| V7.1 ordinary SFT − Native | -0.000580 | [-0.008763, +0.007637] | 9 / 5 / 50 |
| V7.1 ordinary SFT − Legacy simple rule | -0.012097 | [-0.027122, +0.000908] | 17 / 21 / 26 |

### Prespecified gate

| Requirement | Met |
|---|---|
| All 64 sources complete in all inference arms | Yes |
| Targeted minus ordinary mean TEDS at least 0.005 | No |
| Targeted has a positive lower confidence bound against each of three controls | No |
| Targeted malformed count does not exceed any control | Yes |
| Targeted mean character distance does not exceed native | Yes |

### Training and data

Both arms start independently from the same pinned TeleOCR custom Qwen2.5-VL base and identical initial adapters. They use the same 34 real TRAIN sources, eight synthetic donor sources, real-example order and appearance frequencies. Eight DEV sources remain reserved and were not used for tuning or checkpoint selection.

There are 32 controlled prototypes: eight donors × two relation states × two appearances. Ordinary SFT repeats relation state 0 twice per appearance; targeted SFT sees each relation state once per appearance. Each arm uses 68 real and 64 synthetic exposures over two epochs, with 33 optimizer updates. Synthetic examples do not add independent real documents.

Rank-8, alpha-16, dropout-0.05 additive LoRA trains 2,293,760 float32 parameters in 112 language attention projections. The base and vision weights remain bfloat16 and frozen. AdamW uses learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01 and gradient clipping 1. Microbatch size is 1, with accumulation 4. Seed is 0. Updates 1–2 warm up linearly; updates 3–33 use `1e-4 * 0.5 * (1 + cos(pi * (update - 3) / 31))`. Only complete assistant OTSL and EOS tokens contribute to loss; no sequence truncation, quantization or gradient checkpointing is used. Final update 33 is the fixed checkpoint.

| Arm | Sequence tokens | Target tokens | Image tokens | Allocated training seconds | Fresh-base reload exact |
|---|---:|---:|---:|---:|---|
| V7.1 ordinary SFT | 132764 | 40164 | 87188 | 48.599 | Yes |
| V7.2 targeted SFT | 132768 | 40168 | 87188 | 49.357 | Yes |

A frozen 640-candidate order excluded historical source groups. The acquisition prefix contained 192 candidates: 190 PDFs were downloaded (261,186,007 bytes). Of 111 aligned pages, 74 passed the frozen native-crop geometry requirements; the first 64 formed the confirmation set. All 64 crops were visually reviewed before recognition. Selection did not use recognition quality. Source IDs are disjoint from TRAIN, DEV and historical exclusions; exact and near-identical image checks found no duplicates. Shared publisher formatting may still recur.

Real inputs are unchanged layout-model crops from 200-DPI source PDF rendering. Synthetic inputs use the complete table PDF at 200 DPI and default recognition preprocessing, as explicitly selected for this pilot. Local and server PDF pixels differed, so the actual server crop PNGs were retrieved and hash-verified. Ground-truth boxes were used only for eligibility checks, never as recognition crops.

### Inference, validation and cost

All four policies use the same pinned default Transformers generation path, prompt and decoder: greedy generation, maximum length 16,384, no-repeat n-gram size 100 and repetition penalty 1. The Transformers path ignores the presence/frequency penalties applied by historical vLLM. The frozen legacy rule adds a LANCZOS reread at scale 0.535299427146522 and selects between that candidate and shared native output.

256/256 table model calls completed, in 256 batches of one. Native and both fine-tuned arms received identical processed input tensors for 64 and 64 sources respectively. The fixed reread changed actual tensors on 64/64 sources. The legacy rule accepted 59 candidates and changed 38 final outputs. Ordinary and targeted SFT changed 14 and 11 outputs from native.

Total task allocation was **2.384904 GPU-hours**, including technical attempts, layout diagnostics, source layout, both fits, loading and recognition. Shared native work is charged once. One inference admission attempt found all idle cards already leased; it performed zero model calls and allocated zero GPU time. The previously unstarted shard was admitted after a card was released. All task-owned workers exited and leases were released; physical GPU0 was excluded throughout.

Exact frozen-base parameter checks, adapter changes, fresh-base reload logits/loss agreement, and 21 supervision/target/output contract checks passed. Malformed outputs are counted separately from runtime failures. Original rich HTML is the primary TEDS target; symmetric typography stripping is a separate diagnostic. Cell exact match uses the union of (row, column, rowspan, colspan) keys. Character distance uses row-major text, explicit cell separators and maximum string length.

The pilot is small, covers one eligible table per source, and does not establish borderless-only generalization or a full-page overall score. Intervals are descriptive and carry no power guarantee. Descriptive DEV metrics were not run. Runtime versions and source hashes were recorded, but a complete binary environment snapshot was not captured.

| Recognition path | Calls | Prompt tokens | Image tokens | Generated tokens including EOS | EOS not reached | Decoder errors |
|---|---:|---:|---:|---:|---:|---:|
| Native | 64 | 61973 | 59477 | 41777 | 0 | 0 |
| Fixed reread candidate | 64 | 19739 | 17243 | 41914 | 0 | 0 |
| V7.1 ordinary SFT | 64 | 61973 | 59477 | 41806 | 0 | 0 |
| V7.2 targeted SFT | 64 | 61973 | 59477 | 41809 | 0 | 0 |

### Evidence and adapters

- [All metrics and paired intervals](V7_GENERATOR_RESULTS.json)
- Training and optimizer traces (retained locally)
- Data selection and input contract (retained locally)
- Inference calls, changes and timing (retained locally)
- Allocation costs and failures (retained locally)
- V7.1 adapter configuration (retained locally)
- V7.2 adapter configuration (retained locally)
- Executable source and usage (retained locally)

Frozen protocol SHA256: `6cb1c011102a73f698e66544dcf4b685897cd176ef835ff7140cb3ebf6651b66`.
TEDS scorer SHA256: `5694f070d87ffb0a919b326ec293abb70fd9eae00b3a1dc00983dd4e8b20216e`.

<a id="diagnosis"></a>

Source report SHA-256: `882c4b05305562cd3495583dccfaa2a2052541c9bcc14d3770b3cecd07464b2d`.

## V7 post-pilot diagnosis

**The adapters loaded and learned, but most probability changes did not change generation decisions. The controlled empty-cell/span task itself remained unsolved in free generation.** The strongest supported diagnosis combines a narrow shared training signal with a persistent structural decision failure; this audit does not identify one unique underlying cause.

The original failed pilot gate, all 64 confirmation results, both checkpoints and all original evidence remain unchanged. This work performed no training and no new confirmation inference.

### Output equality is now measured directly

| Comparison | Raw OTSL exact | Canonical grid exact | Structure exact | TEDS ties |
|---|---:|---:|---:|---:|
| targeted_sft - ordinary_sft | 61/64 | 61/64 | 64/64 | 61/64 |
| ordinary_sft - native | 50/64 | 50/64 | 60/64 | 50/64 |
| targeted_sft - native | 53/64 | 53/64 | 60/64 | 53/64 |

The 61 targeted-versus-ordinary TEDS ties happen to be exact raw-output matches; that was verified from saved strings, not inferred from scores. The other three differences are in cell text; all 64 structures match. All saved confirmation generations reached EOS.

### Ranked findings

1. **The adapters are active but remain close at the discrete decisions used for generation.** Verified loading; nonzero effective updates and logit/KL shifts; matched real TRAIN/DEV losses improve. Ordinary-versus-targeted teacher-forced top1 agreement is about 99.98%, and all eight DEV greedy outputs match exactly. Directly observed functional proximity; a proximate explanation of output similarity, not identification of a unique optimization cause.

2. **The controlled structural task was not mastered in free generation, including its seen training prototypes.** All three checkpoints have 0/32 exact prototype structures. Targeted first structural-tag mismatch is an expected empty-cell tag replaced by a content/span tag in 26/32 cases. Targeted relation NLL improves under correct prefixes without correcting the full greedy table. Directly observed failure mode; does not isolate data ambiguity from loss weighting, capacity or optimization.

3. **The experimental contrast is real but narrow and shares a dominant training signal.** 100/132 exposure slots have identical actual image tensors and target token IDs. One 2x4 template family has two relation states and eight text donors. Changed-exposure relation-edit coefficients are about 1.76% and 1.82% of the equal-example objective. Direct schedule/token evidence; coefficient is not gradient attribution or a demonstrated causal effect.

4. **bfloat16 residual rounding can damp small updates, but total numerical erasure is ruled out.** No sampled nonzero update casts to zero. About 46–60% of sampled residual additions leave base output values unchanged; the remaining values change and downstream logits/losses differ. All 112 tested projections are bias-free. Illustrative two-sample activation audit; relative contribution needs a precision counterfactual and is not identified here.

### Matched forward and generation evidence

The frozen probe uses every existing 34 real TRAIN source, eight DEV sources and 32 prototypes. Three frozen checkpoints receive 222 correctly paired teacher-forced calls, 96 opposite-relation image calls, and 120 default greedy generation calls on all prototypes and DEV inputs. Image swaps preserve the target, teacher-forced prefix, text donor and appearance. All actual image tensors, tokenizer targets and loss masks were checked before GPU execution.

| Group | Base mean NLL | Ordinary mean NLL | Targeted mean NLL |
|---|---:|---:|---:|
| real_train | 0.019160 | 0.017268 | 0.017342 |
| development | 0.026906 | 0.023871 | 0.024411 |
| prototype | 0.116628 | 0.062126 | 0.064248 |

NLL is averaged equally over examples and conditions on the complete correct preceding tokens. These are fixed-checkpoint comparisons on matched samples, not a sequential training-loss curve or a free-generation accuracy estimate.

| Checkpoint | Prototype structure exact | DEV structure exact | DEV canonical exact | Prototype relation-token NLL | Wrong-image minus correct-image relation NLL |
|---|---:|---:|---:|---:|---:|
| base | 0/32 | 5/8 | 3/8 | 0.342706 | +0.727069 |
| ordinary_sft | 0/32 | 5/8 | 3/8 | 0.361034 | +0.653603 |
| targeted_sft | 0/32 | 5/8 | 3/8 | 0.288303 | +0.664818 |

All 32 image swaps worsen relation-token loss for each checkpoint. Targeted greedy structure changes on 10/16 paired images, but neither state pair is fully correct. Image conditioning exists; it is not sufficient for the prescribed structure. Ordinary has one malformed prototype output; base and targeted have none. All three have valid outputs and EOS on all eight DEV examples. Ordinary and targeted DEV strings are identical on 8/8.

The most frequent prototype error is omission or misplacement of the required empty cell. Targeted first structural-tag mismatch replaces the expected `<ecel>` with `<fcel>` in 14 cases and `<lcel>` in 12. Twenty-six outputs collapse the first row to two cells. This is a descriptive sequence audit; the exact conditional logit at the first erroneous greedy branch was not separately retained.

| Teacher-forced group | Targeted–ordinary RMS logit difference | KL(ordinary || targeted) | Top-1 agreement |
|---|---:|---:|---:|
| real_train | 0.050280 | 0.00002675 | 99.9848% |
| development | 0.056153 | 0.00001739 | 99.9821% |
| prototype | 0.075878 | 0.00246052 | 99.3880% |

These agreement rates concern next-token choices under correct prefixes. They do not measure real-table recognition accuracy. Aggregate margins cannot establish the behavior at every rare structural decision.

### Supervision and numerical checks

Exactly 100/132 paired exposure slots have the same processed image tensor and target token IDs; 32 differ. Eight text donors share one 2×4 grid template with two states: a first-row colspan of two moves from the left to the right. The row-major text sequence and a deliberately empty cell are preserved. The processor keeps distinct image tensors with the same grid `[1, 28, 86]`. Appearance variants change font/background, not structural coverage.

The rendered text centers visibly move relative to four stable lower-row columns. However, there are no rules or visible empty-cell markers: the assigned spans are a plausible centering convention, not a uniquely determined interpretation of the glyphs. Mechanical target/image consistency therefore does not prove visual identifiability.

| Objective quantity | Ordinary | Targeted |
|---|---:|---:|
| Pure structure coefficient | 48.7136% | 48.7383% |
| Structure plus mixed coefficient | 52.8150% | 52.8369% |
| All prototype relation-edit coefficient | 3.5200% | 3.5761% |
| Relation-edit coefficient in differing exposure slots | 1.7600% | 1.8161% |

Coefficients follow the actual per-example mean loss and equal example weighting. Mixed boundary tokens are reported separately. The relation-edit subset comes from token-sequence alignment, not a complete causal attribution of structural learning; coefficient mass is not gradient mass.

The effective updates use `deltaW = 2 * B @ A`. Their global cosine is 0.884049; the arm-to-arm difference norm is 0.4735 times the ordinary update norm. Update/base Frobenius ratios are 0.002220 (ordinary) and 0.001794 (targeted). All 112 module digests match their frozen adapters. Both models use float32 LoRA factors and bfloat16 base/residual arithmetic, with dropout disabled for evaluation.

On two fixed activation samples, no nonzero delta casts to zero. Residual additions leave 46–60% of sampled base values unchanged after bfloat16 rounding. This shows partial damping, not complete erasure. All 112 projections are bias-free; the probe uses the general bias-aware expression regardless. No float32 counterfactual was run.

Both training runs record 132 exposures and 33 updates, finite nonzero gradients, frozen base weights, different final adapters and exact fresh-base reload logits/loss. Recorded update time sums to 30.046 and 31.114 seconds; worker elapsed time is 45.149 and 45.985 seconds. No skipped-step evidence was found. Saved batch losses mix changing examples and cannot be interpreted as a fixed validation learning curve. Neither short runtime nor 33 updates alone identifies the failure cause.

### One minimal next-training proposal — not executed

**Test one loss-only structural-branch weighting change on the original targeted schedule.** On synthetic examples only, weight tokens overlapping <ecel>, <lcel>, <ucel>, or <xcel> tags by 1+4*f, where f is the fraction of token-character offsets inside these tags; other assistant/EOS tokens have weight 1. Use sum(weight*cross_entropy)/sum(weight) per example. Keep real-example loss unchanged. Same original base, seed0, rank8/alpha16/dropout0.05, optimizer, precision, batch/accumulation, final-update selection, 34 real sources, 32 prototypes and targeted 132-slot schedule. One fixed multiplier 5, no sweep; do not initialize from a trained adapter.

Frozen original targeted_sft checkpoint under the same default Transformers inference, with ordinary_sft/base retained as descriptive references. Predeclare success as both relation states structurally correct on at least 12/16 existing image pairs, including correct empty cells, without increased malformed outputs. Report teacher-forced branch NLL and correct-versus-mismatched-image loss separately.

If teacher-forced branch NLL falls but complete greedy structures still fail this threshold, loss dilution alone is not an adequate explanation under the tested fixed recipe. If only training-pair memorization improves, the transfer hypothesis remains unsupported. This is a diagnostic overfit/decision test on existing ambiguous material, not a claim that the current targets uniquely follow from pixels. It does not fix template diversity or identify visual ambiguity as a cause.

After a mechanistic signal, freeze unseen-donor controlled pairs and a new source-disjoint real confirmation cohort before model selection or quality access. Exclude all old TRAIN/DEV/confirmation and their descendants. Set the real cohort size prospectively with a precision/power analysis; at least 64 new sources is a minimum scope, not a power guarantee. Never reuse the exposed 64 sources as an untouched test.

This proposal tests whether emphasizing the observed empty/span decisions crosses the greedy decision boundary. It does not promise an accuracy gain or fix the current one-template coverage. No training was launched.

### Remaining uncertainty

- No single causal explanation is identified for why the structural branch remains wrong; empty-cell visual ambiguity, objective allocation, limited template coverage and optimization/capacity can coexist.
- The exact logits at each first erroneous greedy structural branch were not retained; aggregate teacher-forced margins cannot locate every generation error.
- No higher-precision counterfactual or new optimizer schedule was tested; 33 steps and short runtime are not evidence of a defect or proof that more steps would help.
- A one-family prototype does not test broad structure transfer. Eight text donors are not eight structural templates.
- Teacher-forced gains do not establish better free generation or independent generalization. The original 64-source confirmation and the eight DEV sources are now exposed for diagnosis.

### Execution and evidence

The diagnostic used 736.936 allocated GPU seconds (0.204704 GPU-hours), with zero optimizer updates and zero confirmation calls. The worker exited and its lease was released; physical GPU0 was excluded. One CPU-only preflight failed because tokens crossed the initial structure/text boundary definition. That attempt and protocol were preserved; the revised mixed-token treatment passed all 74 inputs before the sole GPU run.

All 240 frozen diagnostic input files, including delivered pilot artifacts and checkpoints, retain their original hashes.

- Ranked diagnosis (retained locally)
- Full forward, image-swap, greedy and precision aggregates (retained locally)
- Original saved-evidence audit (retained locally)
- Actual token supervision coefficients (retained locally)
- Empty-cell error counts (retained locally)
- One proposed next training experiment (retained locally)
- Diagnostic costs (retained locally)
- Unchanged original pilot report (retained locally)

<a id="structural-loss"></a>

Source report SHA-256: `e9a0367d60094aca72fcb46dc458b78a7ed0212aa232567613ac6381535acd58`.

## V7.2.1 structural-loss ablation

**The predeclared seen-pair gate failed. Stop this fixed loss-only recipe; do not add epochs, sweep multipliers, or run the full benchmark from this result.**

Branch NLL decreased, but the required complete greedy structures were not reproduced. Loss weighting alone was insufficient under this exact fixed recipe.

One final-update adapter was trained from the original base and original LoRA initialization. The sole scientific change was synthetic empty/span-token weighting: `1 + 4*f` for the fraction of a target token inside `<ecel>`, `<lcel>`, `<ucel>`, or `<xcel>`. Each example uses weighted cross-entropy sum divided by its weight sum; real examples retain the original model loss. The existing 34 real sources, 32 prototypes, 132 exposures, 33 updates, seed, optimizer, precision and default native inputs were retained.

### Predeclared endpoint

| Checkpoint | Both states correct | Prototype structure exact | Malformed prototypes | Prototype nonempty text sequence exact | DEV structure exact | DEV canonical exact |
|---|---:|---:|---:|---:|---:|---:|
| targeted_sft | 0/16 | 0/32 | 0/32 | 28/32 | 5/8 | 3/8 |
| structural_loss_weighted | 0/16 | 0/32 | 0/32 | 29/32 | 5/8 | 3/8 |

The frozen gate requires at least 12/16 pairs with both states structurally exact and zero malformed prototype outputs. Result: **FAIL**. All denominators are fixed. Text exactness here compares the ordered nonempty cell strings; empty-cell placement is scored separately. These are seen training prototypes; DEV is descriptive and exposed. Eight donors and one template family do not provide sixteen independent document trials.

### Matched conditional probabilities

| Quantity | Original targeted V7.2 | Weighted V7.2.1 |
|---|---:|---:|
| Real TRAIN mean NLL | 0.017342 | 0.018022 |
| DEV mean NLL | 0.024411 | 0.024762 |
| Prototype mean NLL | 0.064248 | 0.065204 |
| Prototype empty/span branch NLL | 0.439890 | 0.396256 |
| Prototype relation-edit NLL | 0.288303 | 0.274803 |
| Wrong-image minus correct-image relation NLL | +0.664818 | +0.655218 |

NLL is averaged equally over examples and uses the correct preceding tokens. Branch NLL includes any token overlapping a selected tag, including mixed tokens; the training objective weights partial overlap fractionally. Image swaps retain target and text donor. These losses are distinct from greedy complete-table accuracy.

### Empty/span errors

- targeted_sft: empty-cell positions exact on 3/32; outputs containing `<ecel>` 6/32. First structural-tag mismatches: `<ecel> -> <fcel>`: 14; `<ecel> -> <lcel>`: 12; `<fcel> -> <ecel>`: 2; `<lcel> -> <ecel>`: 1; `<lcel> -> <nl>`: 3.
- structural_loss_weighted: empty-cell positions exact on 1/32; outputs containing `<ecel>` 7/32. First structural-tag mismatches: `<ecel> -> <fcel>`: 12; `<ecel> -> <lcel>`: 14; `<fcel> -> <ecel>`: 4; `<lcel> -> <ecel>`: 1; `<lcel> -> <nl>`: 1.

These are sequence and parsed-grid audits. Logits at the first erroneous free-generation branch were not separately retained.

### Training checks and numerical recovery

The first allocation stopped before any update because a strict parameter-gradient equivalence check failed. A zero-update diagnostic found identical native/unit logit gradients on all three fixed inputs, while repeated native backward passes themselves showed relative adapter-gradient differences of about 0.51–0.71%. This isolates a repeatability problem in the original computation path; it is not evidence of a different objective.

The repaired preflight used deterministic MATH attention on a shared forward graph. Original numerical thresholds were retained: loss absolute error <=1e-6, parameter-gradient absolute error <=1e-6 and relative L2 error <=1e-4. All three parameter-gradient comparisons were exact; maximum loss error was 1.49e-8. The original SDPA backend and post-initialization RNG were restored before training. All 11 CPU tests passed, including causal shift, ignored labels, EOS, mixed offsets and equal-example normalization.

The completed fit made exactly 33 updates and 132 exposures. Base parameters remained byte-identical; the adapter changed and its fresh-base reload reproduced logits and both losses exactly. The final update was retained without quality-based selection. Default training backward remains nondeterministic, so a single run cannot separate every small effect from optimization variation.

Synthetic branch coefficient mass averaged over all examples increased from 0.063199 to 0.192054. This is objective allocation, not measured gradient attribution. Raw training losses compare different batches and should not be treated as a fixed validation curve.

### Reference identity, cost and preservation

The evaluation ran 212 teacher-forced and 80 greedy calls: 106 and 40 per checkpoint. Recomputing only the frozen original targeted reference was declared before new outcomes because the historical probe lacked exact branch-category NLL. Model assets, TeleOCR/decoder source, runtime versions, all processed image tensors and target token identities were verified. An original binary environment snapshot is unavailable.

The current reference reproduces 40/40 historical greedy strings; maximum native teacher-forced loss difference is 0. Earlier base/ordinary diagnostic results remain historical context, not newly recomputed controls.

Total new allocated GPU cost: **0.171668 hours** (618.004 seconds), including the failed preflight, zero-update diagnostic, successful fit and both-checkpoint evaluation. All four workers exited and released their leases; physical GPU0 was excluded. 261 protected earlier files are unchanged. No new sources, exposed-confirmation inference, additional fits or full benchmark were performed.

An SSH interruption during evaluation was recovered by reading the original worker and its durable results. The original evaluation completed without resubmission or duplicate calls.

Artifacts: results (retained locally), training (retained locally), preflight and failed-attempt diagnosis (retained locally), protocol (retained locally), costs (retained locally), decision (retained locally), V7.2.1 configuration (retained locally).

<a id="identifiability"></a>

Source report SHA-256: `b1627debb45b004b052829ad2eb81923485cd2c0cfdd0f0d0c8a03146dafa72c`.

## V7 prototype identifiability audit

**The targets depend on a layout convention, but the audit does not establish that ambiguity caused the failed training.** All 32 prototypes have a pixel-identical, structurally different explanation when asymmetric cell padding is allowed. Within the much narrower original equal-width, centered-text renderer, all 32 gold structures are distinguishable from the 17 alternatives tested per image. There is no contradictory same-image labeling in the finite training set.

The canonical target/decoder/grid interfaces pass on all 32 prototypes, 34 TRAIN sources and eight exposed DEV sources. Removing representation-only differences leaves both V7.2 and V7.2.1 at **0/32 exact prototype structures and 0/16 both-correct pairs**. No model was run or trained in this audit.

### What was actually checked

- All 74 targets: source cell grid, HTML/OTSL round trip, text/empty/span indexing, strict parser, self-TEDS, and consistency of the cached tokenizer/EOS/masking receipts.
- All 32 cached 200-DPI prototype inputs: full-resolution visual inspection on eight four-image sheets, original renderer replay, pixel comparison and ink-location measurements. Two annotated counterexample diagrams were also viewed privately.
- All 16 relation pairs: both upper text groups move 140 pixels left from state 0 to state 1; the lower half is pixel-identical. Four lower anchors are visible in every image.
- All 32 images: 18 first-row candidates each, for 576 rendered candidates; 32 padding-aware minimal counterexamples; valid target/decoder round trips for alternatives.
- Saved V7.2 and V7.2.1 prototype predictions only: raw encoding, canonical cell grid, reserialized structure and structure TEDS. No exposed confirmation records were analyzed.

The exact cached RGB inputs were inspected. Normalized processor tensors are not stored locally; their historical hashes and patch grids were verified against the repeated evaluation receipts. This does not claim fresh tokenizer execution or direct visualization of those normalized arrays.

### Identifiability depends on the renderer domain

| Scope | Result | What it supports |
|---|---:|---|
| Original renderer replay | 32/32 exact | Current font/render code reconstructs the frozen inputs |
| Restricted candidate set | Gold only in 32/32 | Bounded distinction under equal widths and centered, unpadded text |
| Non-gold candidates in that set | 0/544 pixel collisions | No ambiguity demonstrated within this finite set |
| Padding-aware extension | 32/32 exact alternative images | Non-identifiability in this explicitly broader layout family |
| Same input with conflicting training labels | 0/32 | The finite mapping is not logically impossible to memorize |

The restricted search fixes the two-row, four-column grid, appearance and populated lower row. It enumerates every positive composition of the four upper slots, places the two visible strings in order, and leaves all other cells empty. This yields 18 structures. It does not exhaust arbitrary grids, variable widths or alignment conventions. A failed counterexample search cannot prove global uniqueness.

The minimal counterexample merges the upper empty cell into the neighboring colspan-2 cell, yielding a colspan-3 cell. Adding a 280-pixel content inset on the merged side preserves the original font fitting and every nonempty glyph bounding box. Canvas, background, text, lower cells and overall 2x4 extent remain unchanged; no clipping, overflow or hidden text is used. The two valid structures produce **identical RGB pixels**, not merely similar pictures.

That counterexample uses an isolated padding-aware extension. It is **not** a collision produced by the unchanged original renderer. The frozen rubric therefore classifies all 32 prototypes as **convention-dependent**, with zero proven original-renderer collisions. Their two relation states remain visibly distinguishable. The 42 real tables have passed interface checks but remain **unresolved for semantic identifiability**, which was not visually audited here.

No rules or empty-cell markers occur in the original images. Lower alignment anchors support the intended convention, but a blank region does not directly reveal whether it is a separate cell or padding inside a larger span. Reassembled donor text also does not provide a reliable independent semantic specification of the hidden grid.

### Interface and representation results

All 74 canonical targets preserve their source plain text and cell grid, round-trip through the pinned decoder, and self-score at TEDS 1. All cached target-token counts, EOS positions and processor identities match the repeated receipts. Replacing empty-cell `<ecel>` with empty `<fcel>` where applicable produces a different encoding of the same grid; the grid comparison treats it as equivalent.

| Saved prototype comparison | V7.2 | V7.2.1 |
|---|---:|---:|
| Raw target string exact | 0/32 | 0/32 |
| Canonical text-plus-grid exact | 0/32 | 0/32 |
| Structure exact after parsing/reserialization | 0/32 | 0/32 |
| Structure TEDS equal to 1 | 0/32 | 0/32 |

Thus the failed prototype gate is not an exact-string or empty-token spelling artifact. The padding alternatives change the cell grid itself; accepting them would change the scientific target, not merely normalize a representation. The original gate and all old scores remain unchanged.

#### Real-source HTML carries additional representation

All 42 real-source HTML tables contain section wrappers and typography that canonical OTSL does not encode. The existing TEDS implementation counts `thead`/`tbody`/`tfoot` tree nodes even in structure-only mode. The following are local interface comparisons of a canonical decoded target to its own original source HTML, **not model accuracy or new benchmark scores**.

| Canonical target vs original HTML | 34 TRAIN mean | 8 DEV mean |
|---|---:|---:|
| Full TEDS | 0.938627 | 0.935427 |
| Structure TEDS | 0.961897 | 0.950518 |
| Typography-stripped TEDS | 0.955214 | 0.943127 |
| Structure TEDS after section-wrapper removal | 1.000000 | 1.000000 |
| TEDS after section and typography removal | 1.000000 | 1.000000 |

This identifies a representational difference that future evaluation contracts must declare. It does not explain the prototype failure: those targets have no section wrappers and were already assessed by cell-grid structure. No historical confirmation data or scores were reopened or changed.

### Learnability decision

**No-go for treating the current pool as an ambiguity-free visual-learning diagnostic.** None of its 32 prototypes has been certified uniquely determined by pixels across the documented layout alternatives, and the 42 real interfaces are not semantically certified. No new fit follows from this audit.

This is not a claim that the existing labels cannot be memorized. All 32 input images are distinct and no duplicate image has conflicting labels. A model can learn a stipulated convention even when other pixel-compatible tables exist. Failure at 33 updates does not establish an inability to learn, and ambiguity alone does not identify an optimization cause.

The recommended next construction, for a separate decision, is a **visible-boundary positive control**: eight small ruled fixtures spanning two neutral text contents, both span/empty states and both existing appearances. Explicit cell boundaries would mark the empty slot and omitted boundaries would mark spans. Keep matched borderless versions as an observability control, and retain the same PDF 200-DPI/default input path. This is a scaffolded diagnostic, not evidence of borderless transfer.

No such fixtures were created for training here. Their exact image/PDF/target/token hashes and the training parameter set, update duration, fixed checkpoints and stop rules must be frozen in a later assignment before any model outcome. The proposal (retained locally) records the concrete construction and acceptance requirements. It deliberately does not present an unbuilt data set as an executable, hash-bound training protocol.

### Limits and closure

Established: canonical interfaces work; visible states differ; encoding normalization does not rescue the result; exact alternatives exist under a documented broader convention. Unresolved: whether the trained model prefers that convention, how much ambiguity contributes to failure, and whether another fixed optimization design would memorize the targets. Small-sample success would still not establish visual reasoning or generalization.

All prior result/model/code files remain unchanged. The audit used local CPU and image viewing only: zero GPU seconds, remote jobs, new model calls, training updates or external sources. Compute and failures are recorded in costs (retained locally). Private source images, labels, counterexamples and prediction bodies are not included in these public artifacts.

Artifacts: interface audit (retained locally), identifiability (retained locally), visual coverage (retained locally), evidence and unknowns (retained locally), conditional no-go/redesign (retained locally), source hashes (retained locally).

<a id="native-input-gate"></a>

Source report SHA-256: `f4bade2554651809aacb75ad995aacc01e41b9d25106f3db22c11007da9605fe`.

## V7.2.2 native-input blocker

**The new diagnostic stopped at the required native input gate, before training.** All eight ruled fixtures produce complete default table regions. None of the eight matched borderless controls produces a table region; the detector instead emits title/text blocks. This prevents the required matched 16-input study under the frozen contract.

This is a preprocessing result, **not a failed 200-update learning result**. Training updates, exposures and scored recognizer calls are all zero. The planned base and checkpoints25/50/100/200 comparison has not run, and no V7.2.2 adapter exists.

### Completed work

Exactly eight ruled fixtures and eight paired borderless fixtures were constructed from the two prescribed neutral contents, two relation states and two appearances. The source geometry, pinned fonts, four-pixel dark rules, PDF/bitmap/target hashes, all800 exposure slots and prospective200-update protocol were frozen before any scored recognizer output.

All16 targets round-trip through the existing HTML/OTSL decoder and self-score at1. All16 exactly sized PDFs render at200DPI with pixel identity to their source bitmaps. Within each matched source pair, pixels differ only on the frozen cell-rule mask. The previously documented padding counterexample contradicts an explicit cell boundary under this line grammar. All16 source inputs were visually reviewed.

The unchanged default native layout worker then processed all16 PDFs. These16 detector calls are counted separately from content recognition. No targets were supplied to that worker, and it performed no table-text recognition.

| Native input outcome | Ruled fixtures | Borderless controls |
|---|---:|---:|
| Required | 8 | 8 |
| Complete unique table regions | 8 | 0 |
| Missing table regions | 0 | 8 |
| Actual crops visually reviewed | 8 | 0, unavailable |

All eight returned ruled crops are1129x327 pixels, contain every glyph and the complete outer frame, and exactly reproduce the pinned native crop operation. All were inspected visually. The complete matched-pair count is0/8. Missing borderless inputs remain in the full16-fixture denominator; they were not discarded or replaced.

### Why training did not begin

The assignment explicitly says to stop if the default pipeline cannot provide the required matched full-table regions, and forbids silent crop substitution. That condition occurred. No full-image fallback, ground-truth crop, modified detector, wrapper retry or ruled-only reduced experiment was substituted.

The required actual16-input processor tensors, patch grids and target-token freeze could therefore not be completed. Gradient preflight, training-loop entry, first exposure and first update have not occurred; their durable event fields remain null. Checkpoints25/50/100/200 were not created. The planned80 greedy and160 teacher-forced recognizer calls are **not run**, not zero-accuracy observations. The8/8 ruled learning endpoint remains unevaluated.

### Interpretation and next boundary

The visible rules make these fixtures recognizable as table regions to the default detector. That is useful input-pipeline evidence, but it says nothing yet about whether language-attention LoRA can memorize their output structures. The route's recognizer-learning question remains open. This result neither establishes nor disproves capacity, parameter adaptation or real-table usefulness.

The exact fixtures, schedule, protocol and eight valid ruled crops are retained. A subsequent assignment must explicitly resolve the missing borderless-input contract before the planned fit can proceed. No extra examples, task branch, detector tuning or further training was launched here.

### Execution and preservation

The sole GPU allocation ran from **2026-10-08T14:47:20.839930+01:00** to **2026-10-08T14:48:14.207561+01:00**, Europe/London. It used a verified idle non-GPU0 device and charged **53.159 GPU seconds** (0.014766 hours). The detector worker exited successfully and released its lease. There were no retries or scored recognizer calls.

All637 protected historical files remain unchanged. Raw images, labels, detected-block bodies and deployment bindings remain private. Public artifacts report the full denominator and distinguish the input-contract blocker from model accuracy.

Artifacts: results and missing-input accounting (retained locally), preflight (retained locally), prospective unexecuted protocol (retained locally), costs (retained locally), decision (retained locally), source hashes (retained locally).

<a id="ruled-only"></a>

Source report SHA-256: `5d56a80004acfc8eeaa4d213b5a4e7f54cefb519b1019bbcc44515f9e9bfeacd`.

## V7.2.2 ruled-only learnability control

**The endpoint passed, but the original base was already perfect; learned improvement is not established.**

One fixed 200-update LoRA fit used the eight previously detected ruled native crops, with 800 exposures (100 complete epochs). The final checkpoint at update 200 is primary; checkpoints at updates 25/50/100 are diagnostics and were not selected by quality. All evaluation followed training.

| Model | Exact structure | Both-state pairs | Malformed | Full canonical exact | Nonempty text exact | Full TEDS | Structure TEDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8/8 | 4/4 | 0/8 | 8/8 | 8/8 | 1.000000 | 1.000000 |
| checkpoint25 | 8/8 | 4/4 | 0/8 | 8/8 | 8/8 | 1.000000 | 1.000000 |
| checkpoint50 | 8/8 | 4/4 | 0/8 | 8/8 | 8/8 | 1.000000 | 1.000000 |
| checkpoint100 | 8/8 | 4/4 | 0/8 | 8/8 | 8/8 | 1.000000 | 1.000000 |
| checkpoint200 | 8/8 | 4/4 | 0/8 | 8/8 | 8/8 | 1.000000 | 1.000000 |

The fixed endpoint requires 8/8 exact structures, 4/4 pairs and 0 malformed outputs. Result: **PASS**. The final model changes exact-structure count by **+0/8** relative to base. Exact structure includes cell positions, spans and whether each cell is empty. Text errors remain visible in canonical and text metrics. Structure-only TEDS omits text and therefore does not itself encode empty-versus-nonempty identity.

### Conditional losses and image swaps

| Model | All-token NLL | Empty/span branch NLL | Relation-edit NLL | Wrong-image minus correct-image relation NLL | Empty positions exact | Span positions exact |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.000171664 | 0.000175173 | 0.000271688 | +3.727243 | 8/8 | 8/8 |
| checkpoint25 | 2.42035e-05 | 5.52575e-06 | 9.64461e-06 | +4.885025 | 8/8 | 8/8 |
| checkpoint50 | 7.5233e-06 | 5.77951e-07 | 1.13249e-06 | +5.586115 | 8/8 | 8/8 |
| checkpoint100 | 3.18247e-06 | 2.07907e-07 | 4.13507e-07 | +5.910286 | 8/8 | 8/8 |
| checkpoint200 | 2.29603e-06 | 1.3411e-07 | 2.68221e-07 | +6.084100 | 8/8 | 8/8 |

Each model has 8 correct and 8 opposite-relation teacher-forced calls plus 8 default greedy generations: 80 conditional and 40 greedy calls total. Swaps keep text content, appearance, target and prompt fixed. NLL uses the correct preceding tokens; it does not measure complete free-generation correctness. Category means give examples equal weight. Branch-token categories include any overlap with empty/span tags; training itself remains unweighted.

### Free-generation error evidence

- base: no structural-tag mismatch.
- checkpoint25: no structural-tag mismatch.
- checkpoint50: no structural-tag mismatch.
- checkpoint100: no structural-tag mismatch.
- checkpoint200: no structural-tag mismatch.

Private traces retain generated token steps, including top-five and structural-vocabulary log probabilities. The first-error mapping is empty for outputs with no structural mismatch; all 40 outputs in this run have no such error. The recorder returns scores unchanged after default generation processors; generated token IDs are checked against recorded argmax decisions. These are model-generated-prefix observations. Equal preceding structural tags do not imply a correct text prefix, and no counterfactual extra probe was run.

### Inputs, training checks and scope

The original 16-case input gate detected 8/8 ruled tables and 0/8 borderless tables. The amended run reuses the exact eight native crops; all eight borderless recognizer results remain unavailable. They are not recognition failures or zero accuracy. No new rendering, detector pass, ground-truth crop or whole-image fallback was used. All actual processor tensors, grids, target tokens and masks were frozen before training and reproduced exactly for evaluation. Native crop images were visually inspected; normalized tensor pixels were not directly visualized.

The model starts from the original base and original seed 0 zero-B initialization. Only 112 language-attention LoRA projections are trainable (rank 8, alpha 16, dropout 0.05, float32 adapters, bf16 frozen base and vision). AdamW, the fixed warmup/cosine schedule, assistant/EOS-only mean loss and equal four-example accumulation follow the frozen recipe. Eleven CPU tests and three actual-model unit-weight gradient checks passed. Deterministic MATH attention was confined to preflight; original SDPA and RNG state were restored. Base parameters remained unchanged; all four finite checkpoints reproduced saved logits/loss exactly after fresh-base reload.

These are seen fixtures with two contents, two relation states and two appearances. A pass establishes only this fitting task. It establishes neither real-table gain, borderless transfer nor independent generalization. Several factors changed relative to prior prototypes, so this run cannot attribute previous failure to boundary ambiguity. One run does not estimate optimization variance.

### Cost and closure

Incremental allocated GPU cost: **275.814 seconds** (0.076615 hours). The reused parent input-gate cost is 53.159 seconds; combined study cost is 328.973 seconds, counting it once. All owned workers exited and leases were released; physical GPU0 was excluded. 746 protected earlier files are byte-identical.

Stop this completed fit. A future task must contain predeclared difficulty beyond base performance; no such experiment is started here.

Artifacts: results (retained locally), training (retained locally), preflight (retained locally), protocol (retained locally), costs (retained locally), decision (retained locally), original input blocker (retained locally), final adapter configuration (retained locally).

<a id="real-TRAIN-inventory"></a>

Source report SHA-256: `f9ec7b8e96c89c108d84320e8b82578a44bfc9646a89f600121fabab8a79420b`.

## Real-TRAIN error inventory

All 34 original-base generations and visual reviews are complete. **The next whole-table repair fit is not recommended from this inventory.** One unambiguous error table and eight correct-retention tables are eligible. Twenty-five tables remain excluded because of source-label mismatch, text encoding, font ambiguity, or unresolved grid conventions. No new fit was launched.

Four tables contain directly verified content errors. Only one has an entirely eligible target: a footnote-symbol substitution. Two others contain clear character deletions together with unresolved 1/I glyph identities; another combines a visible punctuation error with uncertain header spans. The frozen priority excludes these entire tables. A partial-cell loss was considered during review but is not adopted as an exception to that rule.

### Saved generation results

| Measure | Result |
|---|---:|
| Complete generations / denominator | 34 / 34 |
| Canonical grid and text exact | 8 / 34 |
| Structure including empty-cell identity exact | 25 / 34 |
| Original-source-HTML mean TEDS | 0.863120241 |
| Canonical mean TEDS | 0.915531142 |
| Canonical structure-only mean TEDS | 0.939753090 |
| Section/typography-normalized source mean TEDS, diagnostic only | 0.915531142 |
| Malformed / failed / length-limited | 0 / 0 / 0 |

Scores use a 0–1 scale. The 8/34 strict exact count is not a measured visual-error rate. All eight canonical-exact cases have original-HTML score gaps explained by section wrappers and typography unsupported by the canonical OTSL representation; normalized source TEDS is 1 for each. Original scores and labels are retained unchanged. Across all34, comparing canonical targets with original HTML itself gives mean TEDS 0.938626504.

The nine strict structure mismatches involve segmentation, heading spans or empty-cell ownership. This review did not establish a uniquely supported structural-repair family. Exclusion records uncertainty; it does not declare the predictions correct. Two separate source-label/image mismatches are retained explicitly. Thirteen further tables differ in text serialization or spacing, and two have verified deletions alongside ambiguous glyph identities.

### Disposition and next-fit decision

| Frozen primary category | Tables |
|---|---:|
| Reliable text/content error | 1 |
| Unsupported representation-only difference, retention eligible | 8 |
| Source-label mismatch or visual ambiguity, excluded | 25 |
| Reliable structural, combined, crop fault, invalid/missing, or fully source-exact correct | 0 |

The proposed whole-table training pool is insufficient for a multi-case structural-repair study. A one-case glyph memorization probe is technically possible, but its outcome would say little about the motivating problem. The 8–16 planning target is not being imposed as a minimum gate; the recommendation follows from the narrow surviving error and uncertain supervision. No rank change, vision unfreezing, new source acquisition, relabeling or partial-cell exception is presumed to solve this.

The no-go decision (retained locally) and next-fit record (retained locally) state why no optimizer, training schedule or checkpoints were selected. Exact candidate IDs, native image/target/tensor hashes, all34 dispositions and evidence references are frozen privately. The public inventory (retained locally) retains all34 ordinal cases. A future revised design must be prospective, retain the base reference and all34 denominators, and measure repaired, harmed and unchanged cases. TRAIN repairs would still require separate fresh-source validation before a generalization claim.

### Execution and limits

Inputs were the unchanged cached PDF200DPI/default predicted native crops; image tensor, grid and prompt identities matched prior preprocessing. Original-base Transformers inference was label-free with one greedy generation per image. Offline scoring used the unchanged source and canonical labels. No rendering, detector rerun, teacher-forced sweep, training, new sources, DEV, confirmation or benchmark calls occurred. This is not equivalent to historical vLLM scoring.

One non-GPU0 allocation consumed 557.128062 GPU seconds (0.154758 GPU hours). There were no retries. Worker and lease release are verified. Prior native-input preparation is reused and not charged again. All 826 protected prior files remain byte-identical.

Visual adjudication used one assistant reviewer and the exact full-resolution cached crops. Processed tensors were hash-verified rather than directly viewed; there was no independent second review. The labels and source correspondence were checked, but this selected TRAIN inventory supports neither an independent accuracy estimate nor an inferential confidence interval.

See results (retained locally), input audit (retained locally), frozen taxonomy (retained locally) and costs (retained locally).
