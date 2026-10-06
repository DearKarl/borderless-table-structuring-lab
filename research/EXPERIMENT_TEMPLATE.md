# Experiment record template

Copy this template into a new English-named record when work is agreed. Use
`not available` or `not applicable` with a reason instead of inventing evidence.
Do not put private deployment details or benchmark payloads in public records.

## Identity and question

- Experiment ID and record date:
- Authors/contributors and source attribution:
- Status: proposed / running / completed / incomplete / stopped
- Question and connection to the OmniDocBench objective:
- Prior evidence and expected comparison:
- Acceptance criterion, resource budget and stopping condition:

## Frozen inputs and method

- Repository commit and working-tree status:
- Model identifiers, revisions and hashes:
- Training/development data, splits and benchmark exposure:
- Benchmark revision, input roster hash and page count:
- Rendering, preprocessing and candidate-selection policy:
- Prompts, decoding parameters, seeds and recognition-call policy:
- Runtime, dependencies, hardware and external assets:
- Evaluator commit, configuration and ground-truth hash:
- Baseline/control and which factors differ:
- Timeout, truncation, retry, fallback and missing-output rules:
- Output locations and immutable manifests:

## Results and validation

- Full denominator; completed, failed, timed-out and unstarted counts:
- Exact aggregate metrics, units and direction:
- Paired differences and applicable uncertainty/analysis assumptions:
- Measured cost and accounting boundaries:
- Data-free tests versus actual inference/scoring checks:
- Links to aggregate reports, logs that are safe to publish, and hashes:
- Deviations from the frozen plan:
- Negative findings and limitations:
- Interpretation supported by this evidence:

## Reproduction and publication

- Commands and working directory:
- Availability and attribution of required code, data and model assets:
- Independent reproduction status and evidence:
- Official leaderboard submission/acceptance status and link, if any:
- Next decision and remaining unknowns:

A successful run is not automatically a method gain. A local benchmark score is
not automatically an accepted leaderboard result.
