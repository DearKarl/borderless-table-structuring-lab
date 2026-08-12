# Synthetic Data Specification 2026.08.12.1

Status: `FROZEN_FOR_GENERATOR_SMOKE`

Dataset identifier: `data-smoke-2026.08.12.1`

Schema identifier: `synthetic-table-record-2026.08.12.1`

## Research question

Can a shared synthetic corpus teach a table model to preserve an already
correct prior and to correct only deterministic structural defects, without
encoding one arbitrary sequence of split and merge actions as the answer?

The first bounded test is a data-generation and validation smoke. It does not
train a model and does not support a performance claim.

## Evidence basis

The design follows four observations from primary research and official
benchmark materials:

1. PubTables-1M identifies inconsistent over-segmentation as a major source of
   table ground-truth noise and uses canonicalization to remove representational
   ambiguity.
2. GriTS evaluates topology, location, and content as separate matrix-level
   properties, so a legal serialization alone is not sufficient evidence of a
   good table.
3. OmniDocBench exposes table attributes such as border condition, merged
   cells, formulas, color, and rotation. These public categories define
   phenomenon coverage, not page-level source material.
4. Synthetic-table research reports benefits from controlling structural
   complexity, especially spanning-cell distributions, while preserving an
   independently sourced evaluation boundary.

Primary references:

- [PubTables-1M](https://arxiv.org/abs/2110.00061)
- [GriTS](https://arxiv.org/abs/2203.12555)
- [OmniDocBench official repository](https://github.com/opendatalab/OmniDocBench)
- [Synthesizing Realistic Data for Table Recognition](https://arxiv.org/abs/2404.11100)

## Supervision contract

Every attempted record receives exactly one gate label:

- `KEEP`: the Raw-like prior and Gold are semantically equivalent in topology,
  token ownership, and valid geometry. Serialization differences do not turn a
  record into an edit.
- `EDIT`: the prior contains a deterministic, replayable structural difference
  from Gold.
- `QUARANTINE`: source rights, semantics, geometry, ownership, determinism, or
  isolation are ambiguous or invalid.

An OCR transcription error does not by itself authorize an Explicit topology
edit. The Explicit target is an order-invariant structural difference and is
empty for `KEEP`. The LoRA target is one complete Canonical Table state and is
semantically equivalent to the prior for `KEEP`.

## Gate and editor boundary

The gate decides `KEEP` or `EDIT` before a route-specific candidate is used.
The editor is invoked only for `EDIT`. Both routes share canonical legality,
OCR-token preservation, geometry completeness, table-only assembly, and exact
prior rollback checks. A rejected or uncertain candidate leaves the prior
unchanged.

The initial decision cost assigns a false edit four times the cost of a missed
edit. This value is preregistered for the smoke and may be changed only by a
new nonterminal single-variable experiment. Current official benchmark scores
may not calibrate it.

## Bounded smoke composition

The smoke contains exactly 256 accepted records if and only if every frozen
seed passes. There is no silent replacement of a failed seed.

| Category | Label | Count | Purpose |
|---|---|---:|---|
| Exact KEEP | KEEP | 64 | Byte- or state-equivalent correct priors |
| Hard KEEP | KEEP | 64 | Visually difficult but semantically correct priors |
| Single minimal edit | EDIT | 64 | One controlled reversible topology defect |
| Complex correction | EDIT | 64 | Multiple interacting but deterministic defects |

At least 64 counterfactual pairs are required. Each pair shares authored
content, template family, font family, renderer family, and base degradation.
One member remains correct (`KEEP`); the other receives a frozen controlled
structural corruption (`EDIT`). The pair must not cross data roles.

The four smoke categories allocate `44/12/8` records to
`train/development/holdout`, respectively. A terminal role is not created in
the smoke. A later corpus may use terminal records only after the generator,
schema, and distribution are frozen and under a separate one-shot protocol.

## Prior construction

Two prior families are allowed:

1. `controlled_identity_or_error`: start from newly authored Canonical Gold;
   use identity for KEEP or a deterministic corruption operator for EDIT.
2. `approved_real_parser`: run a frozen parser on a newly authored or approved
   nonterminal image and retain complete parser provenance. This family is
   initially limited to KEEP-oriented evidence until its licensing and output
   determinism are separately sealed.

No current OmniDocBench or Customer50 page, crop, text, coordinate, annotation,
embedding, identifier, or transformed derivative may be used as a source or
retrieval query.

## Record views

Each accepted record exposes three synchronized views:

- Gate view: prior evidence and the `KEEP` or `EDIT` target.
- Explicit view: the minimal order-invariant topology difference; empty for
  KEEP, with prior text frozen.
- LoRA view: a complete Canonical Table target with topology, geometry, content,
  and ownership.

All views bind to the same image, prior-state hash, Gold-state hash, split role,
and provenance record.

## Deterministic generation

The generator samples structure before rendering. Template, content, font,
renderer, corruption, and degradation parameters are derived from the frozen
seed namespace in
[`generation_parameters_2026.08.12.1.json`](../../configs/generation_parameters_2026.08.12.1.json).

The same inputs must reproduce identical normalized semantic records and
normalized pixels. Encoders that cannot guarantee byte-identical image files
must record both file SHA256 and normalized-pixel SHA256.

## Split and isolation

Document, template, content, renderer, and counterfactual-pair families are
assigned to one role before rendering. The policy is defined in
[`SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md`](SPLIT_AND_ISOLATION_POLICY_2026.08.12.1.md).

Exact file, normalized pixel, perceptual image, normalized structure,
normalized text, source, and geometry-signature audits are mandatory. Any
unresolved cross-role overlap is a hard stop.

## Rights and provenance

The smoke uses newly authored synthetic content and fonts/assets with explicit
compatible rights. Every source and font decision follows
[`LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md`](LICENSE_AND_SOURCE_POLICY_2026.08.12.1.md).
Unknown, non-redistributable, or incompatible material is quarantined before
generation.

## Acceptance and escalation

The smoke passes only when every criterion in
[`ACCEPTANCE_CRITERIA_2026.08.12.1.md`](ACCEPTANCE_CRITERIA_2026.08.12.1.md)
passes and the Evidence Card is sealed. A pass authorizes a new,
non-overwriting shared-corpus build. A failure stops bulk generation and
requires a new dated specification revision.

Neither a smoke pass nor a corpus build authorizes model training.
