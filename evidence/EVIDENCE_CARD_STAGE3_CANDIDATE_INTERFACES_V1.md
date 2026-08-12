# Evidence Card: Stage 3 Candidate Interfaces V1

Status: `PASS_SEALED_FOR_STAGE4`

Authority: `POST_OMNIDOCBENCH_EXECUTION_CONTRACT_V4.md`

## Objective

Verify that the Explicit Layout Transformer and LoRA Table Model expose two
independent table-only candidate interfaces behind the sealed Stage 2 shared
safety layer. This stage verifies deterministic interface correctness only. It
does not train a model, select a configuration, inspect a terminal benchmark,
or make a performance claim.

## Frozen hypothesis and single variable

Hypothesis: independent table-only interfaces can enforce default `KEEP`, OCR
token preservation, reversible Explicit topology replay, complete Canonical
LoRA output, shared validation, and deterministic Raw rollback without terminal
data visibility.

Single implementation variable: `candidate_interface_contract`.

Preregistration SHA256:
`8f41e6e9577529ef75435d21af04d50b208f477127c999134f92c408b9503d87`

## Inputs and isolation

- The interface-specific checks use one deterministic synthetic table fixture.
- The shared-safety regression uses the first 32 records in the frozen order of
  the compliant 169-sample nonterminal development manifest.
- Frozen development manifest SHA256:
  `c8ce22b7cbc3964ddc1de895842110db7d59123274772b8458678f43539ff103`.
- Customer50 and OmniDocBench inputs, predictions, cases, and metrics were not
  opened or used.
- V6 and Formal20k v1 were not executed or modified.
- No GPU and no model runtime were required.

## Implementation boundary

Explicit interface:

- `None` is the default `KEEP` proposal and yields exact Raw pass-through.
- A changed proposal must partition every Raw cell exactly once.
- Text is reconstructed only from owned OCR tokens.
- Geometry is the deterministic union of source Raw boxes.
- Replay metadata binds the candidate to the exact Raw state hash.

LoRA interface:

- Output is one complete Canonical Table candidate.
- Full-page `blocks` or `page` output is rejected.
- OCR-grounding, geometry, coverage, provenance, and Canonical validity remain
  enforced by the shared safety layer.
- The interface does not define or select any LoRA training configuration.

## Preregistered results

All eight assertions passed:

1. Default `KEEP` is exact Raw pass-through.
2. Changed Explicit topology preserves complete OCR token coverage.
3. Explicit replay restores the exact Raw hash.
4. Incomplete Explicit source coverage is rejected.
5. LoRA output remains table-only.
6. Hallucinated LoRA text rolls back exactly to Raw.
7. Full-page output is rejected at the interface boundary.
8. The Stage 2 shared-safety regression remains fully passing.

The Stage 2 regression retained:

- `32/32` valid baselines.
- `32/32` exact identity pass-throughs.
- `160/160` deterministic corruption rollbacks.
- `160/160` exact rollback hashes.

Stage 3 report SHA256:
`54bb525b0d7a383515c251b9752a4821cefde7b3f82182eaed662a41f5368e97`

Stage 3 SHA manifest SHA256:
`e50f441d7a98fc741e002a72a9ed88e94cce3426b336ef3bbaea170925e7616b`

## Code and contract hashes

- Candidate interfaces:
  `e58b17893b29583eb6a8293413fdf034bf1f591a2c73a58245465179d17a825f`
- Shared safety layer:
  `6cd41d5a01d5942f6ce7b82f06c17d0b4302b02836242117ab295698d71d2e90`
- Stage 3 deterministic check runner:
  `b72b68482f11624d8231d97c7cd04cda22bd8539d50d00808f52a411076a8d77`
- Stage 3 contract:
  `79d677f8ca82bcf9b33b0de8da274bfdc977e13e9bc732b11a40017321e46a9f`
- Stage 2 regression report:
  `5948ba25645cb669356ec3349828178dc27e900e47ffc8f5ca32b273f9481683`

## Interpretation

Stage 3 establishes that both route boundaries now enforce the intended safety
contracts before any model candidate can replace Raw MinerU. The Explicit route
can express reversible topology changes while freezing OCR-grounded text. The
LoRA route can submit a complete Canonical Table candidate but cannot rewrite a
page or pass hallucinated text through the shared validator.

These checks do not establish model quality or expected metric gain. They only
make the next correctness-smoke stage auditable and safe.

## Hard-stop audit

- Terminal input or metric visibility: `FALSE`.
- Full model training command: `FALSE`.
- V6 modification or execution: `FALSE`.
- Formal20k v1 overwrite: `FALSE`.
- Route hybridization: `FALSE`.
- Assertion failure: `FALSE`.
- Performance claim authorized: `FALSE`.

## Decision

Stage 3 is complete and sealed. The next authorized activity is Stage 4:
preregistered correctness smokes. Stage 4 remains correctness-only, and all
work must still stop before any full model training command.
