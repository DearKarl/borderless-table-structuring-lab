# External Evidence: A Zero-Cost Self-Confidence Signal for KEEP Routing 2026.08.14.1

## Status of this document

External evidence contributed by a collaborating project, measured on a
different parser and corpus. It is not an experiment record of this repository
and makes no claim about either route here. It is filed because it offers a
candidate input feature for the safety layer's KEEP versus EDIT decision.

## 1 Problem this addresses

The safety layer has to answer one question before any candidate is considered:
**is the Raw parse already good enough to keep?** A router that cannot answer it
will take over cases it should have left alone.

The usual instinct is to add evidence from outside the parser: a second model, a
re-render comparison, or repeated sampling. All of them cost inference. This
document reports that, on the parser we tested, a **free** signal already
carries most of that information.

## 2 The signal

Run the parser once, normally. Then feed the parser its **own output** in a
single additional forward pass and record, at every structure position, the
probability distribution the model assigns:

- `margin` = p(top-1) - p(top-2)
- `p_top1`
- `entropy`

Ground truth is used only for evaluation. The signal itself is computed
entirely from the model's own logits on its own output, so it is available at
inference time.

Table-level score: the **minimum margin** over all structure positions.
Low minimum margin means the model hesitated somewhere.

## 3 Results

Parser MinerU2.5-Pro-1.2B, 653 OmniDocBench table crops, 54331 structure
positions, NVIDIA RTX A6000. 118 tables diverge structurally from ground truth,
535 do not.

### 3.1 Table level: does this table need attention at all?

Discriminating "structure will diverge" from "structure will be correct":

| Feature | AUROC |
|---|---|
| min margin | **0.8621** |
| min p_top1 | 0.8595 |
| max entropy | 0.8522 |

Three independent features agree, so this is not a single-metric coincidence.

### 3.2 Position level: where is the first error?

Pooled over all 54331 structure positions (116 are true first-error positions):

| Budget | recall | lift over random |
|---|---|---|
| 2% | 0.724 | 36.2x |
| 5% | 0.828 | 16.6x |
| 10% | 0.922 | 9.2x |
| 20% | 0.966 | 4.8x |

Pooled numbers can be inflated by between-table separation, so the same test was
repeated **within each table**, ranking only that table's own positions
(n = 116 divergent tables):

- The true first error is the table's **single lowest-margin position**: **47.4%**
- Within the lowest 3 positions: 68.1%; lowest 5: 75.9%
- Within-table recall at a 10% budget: **0.793**
- Median rank 1; mean relative rank 0.058 against 0.5 for random

### 3.3 Cost

One extra forward pass over an already-generated sequence. No second model, no
resampling, no rendering.

## 4 How this maps onto a KEEP router

The measured quantity is "will this parser's structure diverge from ground
truth", which is a proxy for "is Raw good", not the same thing. Two honest
caveats before using it:

1. It predicts **structural** divergence. A table can be structurally correct
   and still wrong in cell content, so a high margin is not proof that Raw is
   good on a content-inclusive metric such as TEDS.
2. It is measured on one parser. The signal comes from that parser's own logits,
   so it must be recalibrated for any other parser before its numbers transfer.

With those caveats, the natural use is as a **prior on the KEEP side**: a high
minimum margin is evidence that the router should not spend a takeover on this
table, and a low minimum margin marks a table worth examining. As one input
feature among others it costs nothing to add and can be ablated cleanly.

## 5 A scoping result that matters

We pre-registered this experiment with the expectation of a **negative** result,
based on a prior finding that self-consistency signals had **zero** recall on
deterministic catastrophic failures, worse than random.

The measurement contradicted that expectation, and the boundary is worth stating
because it affects which "confidence" signals are worth trying:

- **Self-consistency across samples fails** when the failure is deterministic.
  Repeated decoding of the same input produces the same wrong answer, so the
  variance across samples is zero and carries no information.
- **Single-forward distribution sharpness still works.** Even when the output is
  fully determined by greedy decoding, how sharply the distribution is peaked at
  each position still carries information about whether that choice is right.

"Self-consistency signals do not work here" therefore does not imply "confidence
signals do not work here". We had over-generalised from one to the other, and
that over-generalisation nearly cost us this result.

## 6 Limits of validity

1. One parser, one corpus, one structural vocabulary.
2. The signal is evaluated against structural divergence, not against a KEEP or
   EDIT label produced by an offline Gold comparison. It is a proxy.
3. All numbers are diagnostic, measured without any router or safety layer in
   the loop. No end-to-end effect on any final metric is claimed.
4. Thresholds are not transferable. Any deployment must calibrate its own
   operating point on its own development split.
