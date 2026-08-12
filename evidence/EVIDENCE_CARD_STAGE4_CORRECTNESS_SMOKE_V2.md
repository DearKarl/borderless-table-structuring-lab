# Evidence Card: Stage 4 Correctness Smoke V2

Status: `PASS_SEALED_FOR_STAGE5`

Authority: `POST_OMNIDOCBENCH_EXECUTION_CONTRACT_V5.md`

## Objective

Verify bounded end-to-end correctness of the independent Explicit and LoRA
table-only interfaces, the shared validator, exact Raw rollback, and table-only
page assembly on eight frozen nonterminal records.

## Isolation and hard caps

- Records: `8/8`, first records in frozen manifest order.
- Customer50 or OmniDocBench content visible: `false`.
- Development Gold or performance metrics visible: `false`.
- Training steps: `0`.
- GPU seconds: `0`.
- Full-page rewrites: `0`.

## Results

All ten preregistered assertions passed. Explicit default KEEP, valid OCR-copy
candidate acceptance, and exact replay passed for `8/8`. Valid complete LoRA
table candidates passed for `8/8`. Hallucinated text and missing geometry
rolled back exactly to Raw for `8/8`. Non-table page state remained byte-stable
for `8/8`. Full-page output was rejected at the interface boundary.

## Interpretation

Stage 4 establishes correctness only. It does not establish model quality,
expected metric gain, or a surviving training configuration. A pass authorizes
only Stage 5 bounded nonterminal single-variable development experiments.

## Hashes

- Frozen manifest SHA256: `c8ce22b7cbc3964ddc1de895842110db7d59123274772b8458678f43539ff103`.
- Report SHA256: `2fd1a1c3d182f9f97df0a249dc1a0523045a09b92274aedf39cdca9c35fbf576`.
- SHA manifest SHA256: `50d1766c95036f5b6f9e519a31d97c21da3287687404a4a99ab8e03bb94066df`.
- Runner SHA256: `8776b0609a06dea7a6f4e03e03c42dd40cc74500c03c7afd1cad63f048d6dc1f`.
- Candidate interface SHA256: `e58b17893b29583eb6a8293413fdf034bf1f591a2c73a58245465179d17a825f`.
- Shared safety layer SHA256: `6cd41d5a01d5942f6ce7b82f06c17d0b4302b02836242117ab295698d71d2e90`.
- Preregistration SHA256: `a9c12c8d736815f2d2efecd6a9e2dbd57e18b51899e0f71ee280c3dd2f784512`.

## Decision

Stage 4 is complete and sealed. Stage 5 may begin under V5. Full model training
remains prohibited until the mandatory pre-full-training discussion gate and
new explicit user approval.
