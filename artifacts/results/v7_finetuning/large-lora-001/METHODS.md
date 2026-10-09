# V8 large LoRA campaign: frozen experimental design

The study compares ordinary attention LoRA (V8.1), gradient-initialized LoRA-GA
(V8.2), and LoRA-GA with training-only cell geometry supervision (V8.3). Each
recipe has seeds 0 and 1. All six fits use the same original TeleOCR checkpoint,
paired accepted data, per-seed example order and generation loss. These are planned
comparisons, not demonstrated improvements or a claim of a new LoRA algorithm.

Effective October 9, 2026, V8.1/V8.2/V8.3 are the canonical names for the former
V7.3.0/V7.3.1/V7.3.2 recipes. Public fit identifiers use `V8.1/seed0`, for example.
The [version mapping](VERSION_MAPPING.json) binds all six public identifiers to
the unchanged internal run IDs. Naming changes do not add fits or change data,
methods, seeds, budgets, selection or evaluation. Existing runtime paths and
immutable receipts retain their historical names. Future model packages contain
both canonical identities and historical aliases; reconstruction resolves and
checks both before choosing ordinary, GA or grounded-GA behavior.

The [dataset card](DATASET_CARD.md) specifies source isolation and default native
inputs. The first formal fit waits for the common full dataset and input freeze.
Independent fits start on available devices and additional fits start when capacity
becomes free. Running fits do not change batch size, learning rate or worker count.

## Common model and losses

The 112 language attention projections receive rank 8, alpha 16 LoRA with dropout 0.05:
2,293,760 trainable adapter parameters. The existing visual-to-language merger,
`model.visual.merger`, contains 31,464,704 parameters and is trained in every arm.
The remaining visual encoder and original language parameters remain frozen.
Adapters and merger use FP32 trainable parameters with common BF16 model autocast.
Actual-model checks verified the merger identity and a nonzero gradient path.

Generation cross entropy is averaged over supervised assistant tokens plus EOS
within each example, then averaged equally over 16 examples per update. Token NLL
numerator/count are also logged; their ratio is not the same as the equally weighted
example mean. All arms use identical masks and reduction.

The grounded arm adds a 4,100-parameter linear 1024-to-4 head with sigmoid outputs
at final normalized decoder cell-anchor states. Its auxiliary loss averages L1 over
four normalized box coordinates and valid origin cells within each example, then
over examples. Total objective is generation CE + 0.1 times geometry L1. The head is
used only during training. A zero coefficient directly reuses the generation graph.
The head does not receive reference boxes during inference.

## LoRA-GA initialization and packaging

Initialization follows the [author implementation](https://github.com/Outsider565/LoRA-GA)
at commit `c4cd5372c75b290924214b348008891f744512ef` and the
[LoRA-GA paper](https://arxiv.org/abs/2407.05000). The configuration is ArB2r, stable
scaling with gamma 16, rank 8, and randomized SVD with q=min(4r,min(matrix dimensions))
and four iterations. There is no method, rank or loss-weight sweep.

The formal mean generation gradients use a fixed stratified 256-example
accepted-TRAIN subset, the same IDs/order/masks for both GA recipes and both seeds,
with separately recorded gradient-estimation seed 0. This subset was frozen with
the full corpus on October 9, 2026. The 256-example count is our design choice,
not a paper requirement.

The implementation keeps the original checkpoint unchanged and evaluates
`W_original x + 2 [B A dropout(x) - B_initial A_initial x]`. This is the algebraic
residual-base compensation of LoRA-GA, evaluated through explicit FP32 low-rank
branches rather than rounding the compensated base matrix to BF16. Consequently,
the saved package includes both initial and final factors, plus the merger and
training-only head where applicable. It is not an ordinary rank 8 adapter that can
be loaded on an untouched base while ignoring the initial-factor correction.

Technical checks on three excluded source-feasibility examples verified exact
initial evaluation logits and exact fresh-base save/reload logits. They also
verified FP64 algebraic weight merging. No equivalence claim is made for a rounded
merged-BF16 export. Those three examples and their technical factors cannot replace
the formal 256-example calibration. No optimizer update occurred in these checks.

Repeated BF16 fused-SDPA backward passes showed numerical variation even for the
same computation graph. The zero-coefficient diagnostic therefore selected SDPA's
math backend for its equality check, retaining the production SDPA configuration.
That diagnostic produced zero maximum absolute and relative-L2 gradient difference.
Failures and recovery costs remain in the experiment ledger.

## Schedule, telemetry and selection

AdamW uses adapter/head LR1e-4 and merger LR1e-5, betas(0.9,0.999), eps 1e-8,
weight decay 0.01 and global gradient clipping 1. Effective batch size is 16, with
three complete passes, 5% warmup and cosine decay. At 20,000 examples, each fit has
3,750 updates and 60,000 exposures. Six fits total 22,500 updates and 360,000 exposures,
excluding initialization and diagnostics. Padding adds no loss-bearing duplicates.

Every update, including the initial state, records recipe, seed, update, microsteps,
epoch, distinct examples visited, exposures, valid target tokens, generation loss
components, geometry loss/cell count, total objective, learning rates, gradient
norm/clipping, nonfinite flags and elapsed time. Every 50 updates adds adapter update
norms and projection/head gradient summaries. Resource samples use a fixed low cadence.

Fixed stratified 128-TRAIN and 128-DEV panels receive no-dropout teacher-forced CE
checks at 0, every 250 updates and the final update. RNG and training mode are restored.
These are diagnostics, not extra checkpoint-selection opportunities. Whole-DEV greedy
evaluation occurs at the three epoch checkpoints. Mean full TEDS selects each fit's
checkpoint, ties choosing the earlier epoch. Average DEV TEDS across the two seeds
selects the recipe, ties preferring ordinary LoRA, then GA, then grounded GA. Seed 0
of that recipe is the deployment nominee; seed 1 is replication, not a best-seed search.

## Confirmation and full benchmark

The nominee is frozen before confirmation. All six DEV-selected fits, matched base
and the frozen legacy simple rule are evaluated on all 512 confirmation sources,
including failures. Two primary nominee-control full-TEDS differences receive 10,000
paired source bootstrap draws, seed 0, and 97.5% two-sided marginal intervals. The
prespecified gain 0.005 and positive lower bounds are retained as diagnostics.
They no longer gate full benchmark execution.

The October 9, 2026 scope amendment extends the full benchmark to **all six
DEV-selected models**, each on the same 1,651 pages (9,906 model-page outputs over
1,651 distinct pages). All six checkpoint hashes and DEV selection evidence are
frozen before any full-benchmark quality is read. The original seed0 deployment
nominee remains a prespecified analysis; it does not exclude the other five fits.
Confirmation gains do not gate any of the six full evaluations.

Wave A evaluates V8.1, V8.2 and V8.3 seed0. Wave B evaluates their seed1 counterparts
only after all three Wave A inference and scoring tasks have recorded terminal
states. Evaluation uses at most three GPU workers across DEV, confirmation and
benchmark work. Existing training runs retain their original 3,750-update protocol.

The matched base and fixed legacy rule are evaluated once. Their default native
preprocessing is cached with hashes and reused by all six models. Candidate table
pixels, processor tensors, prompts and generation parameters must match the base
inputs exactly; layout, geometry and non-table content remain shared. Full-page
composition also checks non-table identity. This is a table-specialist comparison
inside the same native document pipeline.

No model is reselected using confirmation or benchmark quality. All eight official
metric fields, category breakdowns, failures, empty predictions, per-seed results,
paired control comparisons and shared/incremental costs are retained. A technical
failure receives an explicit disposition; it does not silently trigger a retry or
remove pages. Inference uses the common default Transformers path; historical vLLM
scores are not asserted to be equivalent. There is no external leaderboard submission.

Secondary outcomes include structural TEDS, text errors, exact grids, malformed,
missing and truncated outputs, repair/harm/tie counts, strata and individual seed
results. Two seeds do not establish a reliable seed-level confidence interval;
source bootstrap intervals and seed variation are reported separately. Negative
results constrain this recipe and setting, not all LoRA or all data scaling.
