# Contributing

Contributions should advance the [project objective](README.md): an officially
recognized OmniDocBench Overall score above the leaderboard leader. Methods are
open to change; preserve evidence about what succeeded, failed or remains unknown.

## Propose and record work

1. Describe the problem, relevant prior result and proposed change in an issue or
   pull request. Documentation fixes can proceed directly.
2. Before a new experiment, complete the applicable fields in the
   [experiment record template](research/EXPERIMENT_TEMPLATE.md). State controls,
   evaluation settings, resource limits and the planned stopping condition.
3. Give each run a unique identifier and output directory. Freeze code, model,
   input and evaluator identities before execution; do not change an active run's
   bindings or overwrite a released result.
4. Report outcomes and failures, link aggregate evidence, and update the
   [current state](research/CURRENT_STATE.md) and [changelog](CHANGELOG.md) when
   public evidence changes. Mark unrun designs as proposals.

Keep Training, native-table, V0–V4 and later routes identifiable. Run archived
versions from their documented working directories. A different checkpoint,
policy or runtime must not inherit an earlier run's score or identifier.

## Evidence standards

- Distinguish implementation checks, bounded smoke runs, full local benchmarks,
  paired improvements and officially accepted leaderboard entries.
- Include the full denominator, timeout/failure handling, metric units and
  direction, exact protocol, comparison conditions and measured resource cost.
- Report negative results and strong raw baselines. A higher score from a
  different setup is contextual evidence, not a controlled method gain.
- Keep benchmark ground truth evaluator-only. Record development data and any
  benchmark-informed decisions so evaluation exposure is visible.
- Preserve original reports, manifests, hashes, model bytes and source archives.
  Correct interpretation in a new note; do not silently rewrite frozen evidence.

## Checks

Run the root package's data-free checks when changing its implementation:

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
```

For documentation-only changes, check relative links, factual source references,
and `git diff --check`. Route-specific changes also need that route's documented
checks. CI and invented fixtures do not reproduce benchmark performance. V4's
archive verifier checks bytes, not model accuracy; see
[reproduction guidance](research/REPRODUCIBILITY.md).

## Public and private material

Publish attributable source, permitted artifacts and aggregate evidence. Exclude
credentials, machine bindings, internal chat/handoff records, customer documents,
benchmark page/ground-truth bodies and per-page predictions. Large weights belong
in permitted release assets or pinned upstream downloads, not a new Git commit.
The already archived V4 model remains preserved with its provenance.

Local Codex instructions, receipts, maintenance scripts and coordination records
belong under the Git-ignored `codex/` directory. The local root `AGENTS.md` is only
an ignored loader. Do not force-add these files. Public research documentation
must not depend on private operational files or chat identifiers.

Before incorporating another group member's work, record authorship, source
revision, permission to publish, data/model provenance, evaluation conditions and
the relationship to existing baselines. Do not attribute their result to this
project or label it reproduced until the supporting work has been performed.

## Attribution and publication

Retain [third-party notices](notices/THIRD_PARTY.md), component license texts and
release-specific publication boundaries. The repository currently has no
project-wide license declaration; this cleanup does not select one or relicense
upstream code. Project authorship and a repository-wide license need maintainer
decisions before adding a formal citation file or broader reuse grant.

Until a project paper or DOI exists, cite the repository, exact commit or dated
release, route and report. Cite the benchmark and upstream methods separately.
Publication of source is distinct from an official leaderboard submission.
