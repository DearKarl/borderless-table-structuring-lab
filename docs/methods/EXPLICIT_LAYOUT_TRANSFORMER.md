# Explicit Layout Transformer

## Research question

The Explicit Layout Transformer studies whether table structure can be
recovered as an interpretable prediction over a latent grid rather than as an
unconstrained serialization. The central hypothesis is that topology, physical
geometry, and text ownership should be modeled as related but distinct
variables.

## Representation

Let a table state be

\[
T = (R, C, \mathcal{V}, \mathcal{G}, \mathcal{O}),
\]

where `R` and `C` are grid dimensions, `V` is the set of logical cells, `G`
contains physical geometry, and `O` maps OCR tokens to cells. Each logical cell
is represented by a half-open span

\[
v_i = (r_i^0, r_i^1, c_i^0, c_i^1).
\]

The learning target is an order-invariant difference between an observed table
hypothesis and the final Canonical Table state. Equivalent sequences of merge
and split operations therefore share the same target.

## Model interface

The model consumes visual features together with a structured table prior and
produces:

1. a table-level identity or edit decision;
2. row and column boundary evidence;
3. cell-span or adjacency hypotheses;
4. a partition of source cells into proposed logical cells;
5. confidence values suitable for calibrated evaluation.

The current codebase implements the representation boundary around this model:

- `build_explicit_topology_candidate` constructs a complete table hypothesis
  from a partition of source cells;
- OCR token ownership is copied from the contributing source cells;
- physical geometry is derived from their geometric union;
- every source cell must be covered exactly once;
- `replay_explicit_to_raw` verifies the source-state binding and reconstructs
  the unmodified input state;
- `select_explicit_candidate` evaluates the hypothesis using the shared
  candidate-validation interface.

The trainable transformer is intentionally kept separate from these invariants
so architecture ablations cannot silently redefine the table representation.

## Supervision

Primary supervision uses the final Canonical Table state and a set-valued,
order-invariant topology difference. Historical action sequences may be used
for debugging or replay analysis, but they are not assumed to be unique.

Useful auxiliary objectives include:

- row and column boundary evidence;
- cell adjacency and span consistency;
- edit sparsity or identity prediction;
- token-ownership preservation;
- geometry consistency;
- table-level structural validity.

Each auxiliary objective must be reported separately from the aggregate loss
so a lower training loss cannot conceal a degraded topology or content metric.

## Evaluation

The Explicit track is evaluated with the same data roles and canonical output
format as the LoRA track. Core measurements include:

- GriTS Topology, Location, and Content;
- TEDS-compatible structure and content scores;
- precision and recall for structural corrections;
- cell-count inflation and deflation;
- OCR token-preservation rate;
- canonical validity and geometry coverage;
- results stratified by header, span, border, text, and imaging phenomena.

## Current repository status

The Canonical Table schema, topology-candidate interface, deterministic replay,
shared validation primitives, and synthetic regression tests are available.
Model architecture and training code will be added as a separate, reviewable
research contribution together with its configuration and ablation record.
