# Collaborator Handoff

## Goal

Build a source-traceable, category-balanced synthetic corpus for borderless
table structure recognition without exposing restricted evaluation artifacts.

## Recommended first assignment

The best independent contribution is the rendering-and-degradation generator
because it is bounded, parallelizable, and does not require model code or
terminal data.

Deliverables:

- deterministic weak, partial, missing, and low-contrast border rendering;
- resolution, blur, compression, rotation, scanning-noise, and background
  parameterization;
- font-source and license inventory;
- exact source-to-render provenance;
- replay tests for every parameter family;
- a small smoke manifest with pass, quarantine, and failure reason codes;
- exact and perceptual duplicate reports;
- an English experiment record.

## Other work packages

### Coverage specification

Maintain the phenomenon matrix, parameter ranges, and minimum requested counts
using public task attributes and independently confirmed generic risks.

### Structural generator

Generate multi-level headers, irregular spans, split-only, merge-only, joint
split-plus-merge, empty-cell, minimal-correction, and KEEP-only tables. Emit
direct Canonical Gold, geometry, ownership, and deterministic seeds.

### Validator and replay QA

Independently validate legality, complete grid coverage, token ownership,
geometry, deterministic rendering, and Gold recompilation. Produce
machine-readable failures.

### Isolation audit

Audit document, template, content, renderer, image, structure, text, geometry,
and near-duplicate overlap across all roles.

## Prohibited work

The collaborator must not inspect terminal artifacts, select benchmark cases,
tune official metrics, change the official evaluator, receive holdout or
terminal labels, or start a full model training run.

## Definition of done

A contribution is accepted only after its source/license manifest,
deterministic replay test, isolation audit, complete count accounting, English
experiment record and SHA256 manifest pass review.
