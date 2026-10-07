# Research roadmap

Current update2026-10-07: all engineering is paused for research discussion.
See [current state](CURRENT_STATE.md), [V6 full results](../versions/v5_native/README.md),
[V7 training and development](../versions/v7/README.md), and the
[progress archive](../releases/v6-v7-progress-20261007/README.md).
The sections below retain historical evidence or earlier milestones; they do not
authorize resuming experiments. No official ranking or new generator model is claimed.


## Objective and success criterion

Achieve an officially recognized OmniDocBench Overall score above the leaderboard
leader. TeleOCR is the current target. V0–V4 are attempts toward this objective;
no particular model, training method, fusion design or input policy is mandatory.

## Completed evidence

- Earlier Training and native-table routes have preserved source and table metrics.
- V0–V2 have full historical whole-page results, including strong raw controls
  and the negative paired V2 result.
- V3.1 has a researcher-supplied Overall result; V3.2 has a researcher-supplied
  full-set OFF/ON comparison. Missing metrics and run identities remain explicit.
- V4 has a completed 1,651-page local evaluation and a public source/model/result
  archive. Causal selector benefit and an official leaderboard win remain unproven.
- V5.1-V5.5 each completed all 1,651 pages with zero inference failures. The
  original-image baseline scored 98.173146; the combined pipeline scored
  98.347514. Independent native outputs varied, limiting causal attribution.

The [V5 research proposal](V5_PROPOSAL.md) adds novelty assessment, strong controls,
residual-error analysis and independent validation toward a CCF A publication.

See [current state](CURRENT_STATE.md) for exact evidence and limitations.

## Remaining milestones

| Milestone | Completion evidence |
|---|---|
| Isolate intervention effects from existing evidence | Score saved native/intermediate stages on aligned pages; separate within-run changes from independent native variation before interpreting the five-arm differences |
| Identify the next method from residual errors | Quantified failure types, repair/regression tradeoffs and cost; a specific mechanism distinct from the attributed reference pipeline |
| Validate generalization | Frozen development/evaluation boundary and independent documents or dataset; avoid treating this development benchmark as blind evidence |
| Prepare the V5.5 submission candidate | Resolve upstream model identity, provide permitted assets and portable instructions, then verify a fresh-environment reproduction |
| Complete official submission | Confirm official comparison conditions and obtain an accepted leaderboard result; local scores do not establish rank |

These are evidence milestones, not claims of scheduled or running experiments.
Whether existing results already support a valid candidate should be resolved
before assuming a new architecture is necessary.

## Ongoing repository maintenance

Keep a unique record per experiment using [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md).
Update the current-state index and changelog when evidence changes. Preserve
frozen releases rather than replacing their contents. Document future group
contributions with authorship, source, publication permission and comparison
conditions before incorporating them into this project's results.
