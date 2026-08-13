# External Evidence: Ceiling of Topology-Only Repair 2026.08.14.1

## Status of this document

External evidence contributed by a collaborating project. It is **not** an
experiment record of this repository, was produced on a different parser and a
different corpus, and does not evaluate any route in this repository. It is
filed here because it bears directly on how the Explicit route should report
its results.

Nothing in this document should be read as a claim about
`borderless-table-structuring-lab` itself.

## 1 Research question

The Explicit route is topology-only: it defaults to KEEP, keeps text frozen,
and proposes minimal split/merge changes. A natural question is how much a
metric can move when **only** topology is corrected and cell content is left
exactly as the parser produced it.

We measured the **oracle** answer to that question on a different parser, so the
number is an upper bound rather than an achievable result.

## 2 Setup

| Item | Value |
|---|---|
| Parser | MinerU2.5-Pro-1.2B |
| Corpus | OmniDocBench table crops, 653 tables |
| Metric | Official exact-cell match (content and position must both be correct) |
| Denominator | Fixed, `compare(GT, GT)["total_cells"]` = 43231 cells |
| Scoring exception | Whole table scored 0, per the official metric's semantics |
| Decoding | Greedy, `max_new_tokens=6144`, `no_repeat_ngram_size=100` |
| Hardware | NVIDIA RTX A6000 only, single GPU model across all arms |

**Procedure.** Find the first structure token in the free-run output that
differs from ground truth. Replace **only that one token** with the ground-truth
token, then let the model continue generating on its own. Repeat up to `k`
times. Both the position and the correct token come from ground truth, so this
is an oracle measurement and not a deliverable method.

**Pre-registered control.** For the 535 tables whose structure was already
exact, no repair is applied, so their per-table score must be **identical across
every `k`**. Observed: identical for all 535 tables, 0 violations.

## 3 Results

### 3.1 Overall curve

| k | cell score | vs k=0 |
|---|---|---|
| 0 (free run) | 0.8593 | — |
| 1 | 0.8723 | +1.30 pp |
| 2 | 0.8904 | +3.11 pp |
| 4 | 0.8953 | +3.60 pp |
| 8 | 0.8973 | +3.80 pp |
| 12 | 0.8960 | +3.67 pp |
| 16 | **0.8990** | **+3.97 pp** |

- The k=1 gain has a **document-cluster bootstrap 95% CI of [-1.64, +3.85] pp**
  (454 clusters, 5000 resamples), which **contains zero**.
- k=12 scores below k=8, so the curve has entered its noise band.
- Quadrupling the budget from k=4 to k=16 buys +0.37 pp. The curve is saturated.

### 3.2 Three-way split of the divergent tables (118 tables, 12893 cells)

| Category | n | cells | score at k=0 | score at k=16 | cells still lost |
|---|---|---|---|---|---|
| Structure repaired, **full recovery** | **34** | 2480 | 0.8544 | **1.0000** | **0** |
| Structure repaired, content still wrong | 64 | 6537 | 0.6778 | 0.8687 | 858 |
| Resistant tables | 20 | 3876 | 0.5488 | 0.5766 | **1641** |

Both directions matter:

- **Topology repair genuinely works.** 59 of 118 divergent tables (50%) become
  structurally exact after a **single** oracle correction, which matches the
  `single minimal edit` category in this repository's corpus design. 34 of them
  reach a cell score of exactly **1.0000** once structure is correct.
- **But cell-weighted totals are dominated elsewhere.** 66% of the remaining
  loss (1641 of 2499 cells) sits in 20 tables that absorb up to 16 oracle
  corrections, 14 of which end up structurally identical to ground truth, and
  still score 0.5766.

At k=16, 112 of 118 tables are structurally identical to ground truth token by
token, yet the divergent subset scores 0.8062 against 0.9385 for tables that
were structurally correct from the start.

### 3.3 Table size does not explain the residual

Divergent tables are larger (median 49 cells versus 26). Matching each divergent
table to its nearest-size structurally-correct table (118 pairs, 12168 cells)
gives the size-matched control a score of **0.9537**, which is *higher* than the
0.9385 of the full control group. Larger tables are not harder for this parser.
After size matching, the gap to the divergent subset at k=16 is still 14.75 pp.

## 4 Implication for reporting

| Metric family | Does topology-only repair score? | Basis |
|---|---|---|
| TEDS-Struct, GriTS-Topology | **Yes** | 112 of 118 become structurally exact |
| Exact-cell (content and position) | **Ceiling +3.97 pp** | Section 3.1 |
| TEDS-all, GriTS-Content | Between the two | Not measured here; no claim made |

A topology-only route that freezes text should score clearly on structural
metrics. On content-inclusive metrics, the oracle ceiling we measured is
+3.97 pp.

**Suggestion:** report structural and content-inclusive metrics as separate
numbers rather than a single combined score. Separating them shows what a
topology-only method actually achieves, and prevents a structural gain from
being read as a content gain or vice versa.

## 5 Limits of validity

1. **One parser.** MinerU2.5-Pro-1.2B. A different base model fails differently.
2. **One corpus.** OmniDocBench table crops, not a borderless-specific set. In a
   separate experiment we found hard-table phenotypes to be unstable across
   corpora, so this limit is substantive rather than formal.
3. **One metric.** Official exact-cell, not directly comparable to TEDS or GriTS.
4. **Oracle.** Both the error position and the correct token come from ground
   truth. Any real detector performs worse.
5. **Single-table dominance.** One table
   (`magazine_TheEconomist.2024.*`) loses 525 cells, which is **21% of the
   entire residual**. Any conclusion drawn on this data must be reported with a
   leave-one-out check.

## 6 A caliber error we made, offered as a caution

For a long time we cited an upper bound of "+9.63 pp for repairing structure"
and planned a batch of experiments around it. Tracing its provenance showed it
was **not a measurement**. It was an arithmetic construction that raised the
divergent tables' cell score to **1.0**, that is, it assumed those tables became
*perfect*, not that their *structure* became correct. The measured structural
ceiling is +3.97 pp.

The rule we adopted afterwards, offered here as a suggestion:

> Any number cited as an upper bound must be traceable to a specific
> measurement. Constructed values must carry the word "constructed" in their
> name.

## 7 Accompanying manifest

`TOPOLOGY_REPAIR_CEILING_2026.08.14.1.manifest.json` lists all 118 divergent
tables with per-table language, ground-truth cell count, score before and after
repair, the value of `k` at which structure first became exact, and the
self-confidence minimum margin described in the companion document.

The manifest contains derived measurements only: benchmark sample identifiers
and our own scores. It contains **no benchmark images, no ground-truth text, and
no annotations**.
