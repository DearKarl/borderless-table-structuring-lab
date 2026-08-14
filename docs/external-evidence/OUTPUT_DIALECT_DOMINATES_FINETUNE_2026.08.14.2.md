# External Evidence: Output Dialect Dominates Fine-Tuning Outcome 2026.08.14.2

## Status of this document

External evidence contributed by a collaborating project, measured on a
different parser and corpus. It is not an experiment record of this repository
and makes no claim about either route here. It is filed because it concerns a
failure mode that any generative adaptation route can hit, and that is very hard
to see by reading the training data.

## 1 Problem this addresses

A generative route fine-tunes a model to emit a table representation. The
obvious knobs are learning rate, steps, rank, and module placement. This
document reports that on the parser we tested, none of those knobs mattered as
much as a question that is easy to never ask:

**Is the target string written in the exact surface form the base model already
emits?**

Not the same information. Not an equivalent serialization. The same characters.

## 2 The observation that forced the diagnosis

Six fine-tuning checkpoints all lost accuracy against the base model. Common key
intersection, fixed denominator:

| Arm | n | denominator | base | arm | delta (pp) |
|---|---|---|---|---|---|
| lr 1e-4, 3000 steps | 100 | 4323 | 0.7451 | 0.5316 | -21.35 |
| lr 2e-5, 2000 steps | 100 | 4323 | 0.7451 | 0.5279 | -21.72 |
| lr 2e-5, 6000 steps | 50 | 1597 | 0.7345 | 0.5429 | -19.16 |
| **lr 5e-6, 2000 steps** | 75 | 3124 | 0.6863 | 0.3998 | **-28.65** |

The last row is the one that mattered. A learning rate two orders of magnitude
smaller, trained for fewer steps, did **more** damage than the largest one. If
the cause were "training too hard," damage would be monotone in learning rate.
It was anti-monotone, so that hypothesis was refuted by its own data.

## 3 The actual cause was one character

No new inference was needed. The base model's own raw output on the same tables
was already on disk. Comparing it to the training target on **structural habits
only** (token counts, trailing token, literal newline count, leading and
trailing whitespace):

```
base output: '<fcel>A<fcel>B<fcel>C<nl>\n<fcel>D...'
our target:  '<fcel>A<fcel>B<fcel>C<nl><fcel>D...'

contains a literal newline:  base 40/40,  target 0/40
ends with <nl>:              base 40/40,  target 40/40   (agreed)
```

The model's native dialect places a real newline after every `<nl>` except the
last. Our targets had none. Every row boundary in every sample therefore carried
a gradient penalising the model's own native habit. The signal was dense and
perfectly consistent, which is exactly why the damage barely depended on
learning rate.

Correction used afterwards:

```python
def to_native_dialect(s):
    return s.replace("<nl>\n", "<nl>").replace("<nl>", "<nl>\n").rstrip("\n")
```

## 4 The 2x2: dialect is the main effect, training budget is secondary

Same base model, same data, same evaluation, fixed denominator 3170 over a
39-table holdout. Base model scores 0.8719.

| fixed denominator 3170 | 200 steps (5 epochs) | 6000 steps (150 epochs) |
|---|---|---|
| **HTML target** | 0.5722 (-30.0 pp) | 0.5202 (-35.2 pp) |
| **native OTSL target** | **0.8760 (+0.41 pp)** | 0.7703 (-10.2 pp) |

Decomposition:

- **Dialect**, at equal budget: 30.4 pp at 200 steps, 25.0 pp at 6000 steps.
  Dominant at every budget tested.
- **Training budget**, at equal dialect: 10.6 pp for OTSL, 5.2 pp for HTML.
  Real, but secondary.

The short-budget HTML cell is the one that closes the argument. If over-training
were the whole story, HTML at 200 steps would return to 0.87. It reaches 0.5722.

**The `+0.41 pp` cell must not be read as an improvement.** Of 39 tables, 30 are
byte-identical to the base output, 3 better, 2 worse, and the two worse ones are
off by a single cell each. Paired bootstrap over tables, 10000 resamples:
delta = +0.0041, 95% CI [-0.0009, +0.0120], which contains zero. The honest
statement is **"not worse than the base model,"** not "better."

## 5 Failure mode, for recognition

The damage is not a parse failure. Of 39 tables, **0** failed to parse. The
failure mode is grid explosion:

- The scorer's denominator is `max(pred_rows, gt_rows) x max(pred_cols, gt_cols)`,
  so it inflates with the prediction: 3170 for ground truth against itself,
  3200 for the base model, **5363** for the HTML-target arm.
- Worst single table: 87 ground-truth cells, 2019 predicted grid cells, 2 correct.

This matters for metric reporting: a floating denominator makes cross-arm ratios
sit on different bases. Every number above is on the fixed
`compare(GT, GT)["total_cells"]` denominator.

## 6 The cheap defence

Print a **dialect receipt** at training start and refuse to launch on mismatch.
Ours compares, over a sample of targets, the structural features of the training
target against the base model's own raw output, and raises if they disagree:

```
[dialect receipt] targets containing a literal newline = 200/200
[dialect receipt] first 80 chars of target = '<fcel>In millions<fcel>2009...'
```

The cost is a few seconds. It would have saved six training runs.

A second, related guard: force an explicit output tag on every run and refuse to
start if the output directory already holds a checkpoint. Mixing old-dialect and
new-dialect weights in one directory makes attribution impossible after the fact.

## 7 Limits of validity

1. One base model, one structural vocabulary, one corpus. The **direction** is
   about matching whatever a given base model natively emits; the specific
   newline convention is ours and will not transfer.
2. The 39-table holdout is small. It is adequate for effects of 25 to 35 pp; it
   is not adequate for the +0.41 pp cell, which is why that cell is reported as
   indistinguishable from zero rather than as a gain.
3. All arms are LoRA on the same module set. No claim is made about full
   fine-tuning.
4. This says nothing about whether generative adaptation is the right route. It
   says that a route evaluated with a mismatched target dialect has not actually
   been evaluated.
