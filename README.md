# Borderless Table Structuring Lab

Research on reliable document parsing: input resolution, specialist models, and
selective correction of text, formulas, and table structure.

Our goal is to improve end-to-end parsing on OmniDocBench and develop methods
whose gains hold beyond a single benchmark. We study when an existing parser
benefits from a different input view or specialist, and when an intervention
damages a correct prediction. The project includes successful and unsuccessful
experiments from V0 through V4. V5 is currently at the research-design stage.

[Results](research/RESULTS.md) · [Methods](#methods) ·
[Reproduction](research/REPRODUCIBILITY.md) · [V5 research plan](research/V5_PROPOSAL.md) ·
[Research status](research/CURRENT_STATE.md) · [Contributing](CONTRIBUTING.md)

## Research questions

- **Input resolution:** when does changing the input scale improve recognition,
  and can an adaptive policy outperform a strong fixed policy at comparable cost?
- **Specialist integration:** when does replacing a region with an expert
  prediction improve the complete document rather than only an isolated crop?
- **Selective correction:** how can a system recover errors while preserving
  correct content and controlling additional computation?

OmniDocBench is the primary benchmark. The performance target is an officially
recognized Overall score above the leaderboard leader. The accompanying research
goal is a generalizable contribution suitable for a CCF A venue; no paper or
official leaderboard result is claimed at this stage.

## Results

Selected local results are shown below. Scores use a 0–100 scale. Rows belong to
different experimental groups and are not a single controlled ranking. The
[results record](research/RESULTS.md) includes controls, available component
metrics, sources, and evaluation conditions. A dash means the metric was not supplied.

| Experimental group | Method | Overall ↑ | Table TEDS ↑ | Formula CDM ↑ |
|---|---|---:|---:|---:|
| V0 / V1 | MinerU control | 93.1680 | 90.4345 | 95.5532 |
| V0 / V1 | V0: MinerU + NaviDC tables | 94.5993 | 94.7279 | 95.5532 |
| V0 / V1 | V1: MinerU + Paddle formulas | 93.2175 | 90.4345 | 95.7276 |
| V2 | TeleOCR control | 97.4351 | 96.6919 | 98.5996 |
| V2 | V2: TeleOCR + Paddle formulas | 97.0600 | 96.6919 | 97.4793 |
| V3.1 | Original formula-replacement pipeline + original image | 97.6973 | — | — |
| V3.2 OFF / ON | V3.2 OFF | 93.4951 | 86.9556 | 98.2000 |
| V3.2 OFF / ON | V3.2 ON | 93.7208 | 87.3918 | 98.4454 |
| V4 | GBDT input selection + TeleOCR | 97.1719 | 96.0459 | 98.4993 |

V3.1 and V3.2 values were supplied by the researcher on 2026-10-07; raw scoring
files and run revisions have not yet been added. V3.2 includes all 1,651 pages,
with failures retained, and improves Overall by **0.2257 points** over OFF.
V4 also includes all 1,651 pages, including seven timeout predictions kept empty.
V3.1's evaluation denominator was not supplied.

The paired V2 experiment decreased Overall by **0.3751 points**, showing that
specialist replacement can harm a strong parser. V3.2's scores on each arm's own
successful pages are reported separately, since those subsets may differ.
These findings motivate measuring both repairs and regressions in V5.

An earlier native-table experiment reached **98.7507 Table TEDS** under a
different protocol. This is a component result, not a whole-page Overall score;
see the [table experiments](artifacts/results/README.md).

## Methods

| Version | Method studied | Source |
|---|---|---|
| V0 | Native table-region replacement with a specialist parser | [V0](versions/v0/README.md) |
| V1 | Geometry- and syntax-checked formula replacement on MinerU | [V1](versions/v1/README.md) |
| V2 | Formula expert integration on a TeleOCR backbone | [V2](versions/v2/README.md) |
| V3.1 | Layout/formula integration and original-image input comparisons | [V3.1](versions/v3_1/README.md) |
| V3.2 | OvisOCR2 text-region integration with adoption and fallback rules | [V3.2](versions/v3_2/README.md) |
| V4 | Learned selection between input actions before TeleOCR recognition | [V4 study](research/V4.md) |
| V5 | Proposed study of intervention benefit, regression risk, and cost | [Research plan](research/V5_PROPOSAL.md) |

The [earlier table method](LEGACY_NATIVE_TABLES.md) and
[table-structuring training experiments](training/README.md) remain available.
Method descriptions and measured results are linked separately so new evidence
can be added without rewriting earlier experiments.

## Reproducing the research

Start with the [reproduction guide](research/REPRODUCIBILITY.md) to select a
method and its corresponding model, input, and evaluation configuration.
The root package contains the earlier table implementation. Its local checks are:

```bash
git clone https://github.com/DearKarl/borderless-table-structuring-lab.git
cd borderless-table-structuring-lab
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[test]"
python -m pytest -q tests
```

These checks exercise implementation behavior; model inference requires the
method-specific dependencies and separately obtained weights and data.

## Repository structure

| Location | Research content |
|---|---|
| `research/` | Research questions, experiment summaries, protocols, and future work |
| `versions/` | V0–V3.2 implementations and method-specific instructions |
| `releases/` | Frozen source, models, and results for reproducible experiments |
| `artifacts/` | Machine-readable results, including negative outcomes |
| `btsl/`, `hybrid/`, `training/` | Table structuring, expert integration, and training code |
| `evaluation/` | Evaluation code and metric adapters |
| `tests/`, `examples/`, `scripts/` | Implementation checks and usage examples |

## Attribution

This project builds on OmniDocBench and the upstream parsers named in each
experiment. Cite their original work alongside any result from this repository.
Until a project paper is available, reference the repository commit, method, and
experiment record. See [third-party notices](notices/THIRD_PARTY.md) and
[contribution guidance](CONTRIBUTING.md#attribution-and-publication) for attribution
and licensing information.
