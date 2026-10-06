# Research roadmap

## Objective and success criterion

Achieve an officially recognized OmniDocBench Overall score above the leaderboard
leader. TeleOCR is the current target. V0–V4 are attempts toward this objective;
no particular model, training method, fusion design or input policy is mandatory.

## Completed evidence

- Earlier Training and native-table routes have preserved source and table metrics.
- V0–V2 have full historical whole-page results, including strong raw controls
  and the negative paired V2 result.
- V3.1/V3.2 have source and bounded engineering checks, without full-benchmark scores.
- V4 has a completed 1,651-page local evaluation and a public source/model/result
  archive. Causal selector benefit and an official leaderboard win remain unproven.

See [current state](CURRENT_STATE.md) for exact evidence and limitations.

## Remaining milestones

| Milestone | Completion evidence |
|---|---|
| Reconcile existing scores | Explain local raw TeleOCR 97.435109, V4 97.171890 and public TeleOCR 96.91 using their exact inputs, inference settings and scoring protocols |
| Establish a comparable baseline | A frozen strong control with explicit configuration, denominator, failure policy and resource accounting |
| Select and evaluate an improvement | A bounded design, paired evidence and preserved failures; no route is selected by this roadmap |
| Prepare a reproducible candidate | Versioned code, accessible permitted assets, environment instructions, manifests and a checked reproduction record |
| Complete official submission | Confirm the current submission process and obtain an accepted leaderboard entry above the leader |

These are evidence milestones, not claims of scheduled or running experiments.
Whether existing results already support a valid candidate should be resolved
before assuming a new architecture is necessary.

## Ongoing repository maintenance

Keep a unique record per experiment using [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md).
Update the current-state index and changelog when evidence changes. Preserve
frozen releases rather than replacing their contents. Document future group
contributions with authorship, source, publication permission and comparison
conditions before incorporating them into this project's results.
