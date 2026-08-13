# Raw-preserving bounded smoke

The first smoke is intentionally small and deterministic:

- 256 base table groups;
- 10 counterfactual records per group;
- all authorized phenomenon families represented;
- 90% `KEEP_RAW` and 10% `ACCEPT_EDIT` oracle labels after Gold-only labeling;
- no document, template, content, renderer, or seed family crosses roles.

The smoke must report requested, generated, passed, quarantined, and failed
counts. It must also check canonical legality, complete grid coverage, OCR token
ownership, geometry, direct Gold recompilation, deterministic replay, and exact
/perceptual/structure/text/geometry overlap. Family identifiers are not proof of
visual isolation: cross-role images are screened with 9-by-8 dHash and confirmed
with 32-by-32 pHash. A confirmed perceptual hit blocks the corpus rather than
silently removing records or retuning the threshold after holdout inspection.
Large-corpus audits must stream records or retain only compact group summaries;
they must not require all Raw, Gold, and candidate objects in memory.

## Oracle action contract

The offline labeler receives `raw_record`, `candidate_record`, and
`gold_record`. It emits `KEEP_RAW` unless all of the following hold:

1. the candidate passes the shared safety validator;
2. the candidate has strictly more exact Gold cells than Raw;
3. candidate text agreement is not lower than Raw;
4. candidate geometry coverage is not lower than Raw;
5. non-table page state is unchanged.

A Raw state that exactly matches Gold is always `KEEP_RAW`, even when a
candidate is also valid. Ties and partial improvements are `KEEP_RAW`.

Gold and all Gold-derived values are offline-only fields. The runtime selector
view contains only the image, Raw record, candidate record, and observable
candidate-vs-Raw differences.

## Initial distribution

The frozen first distribution is 90% KEEP and 10% ACCEPT:

- 35% Raw-correct with a legal harmful over-edit candidate;
- 25% Raw-near-correct with inflation, over-split, or over-merge candidates;
- 15% Raw-bad with a tied, partial, or trade-off candidate;
- 15% identity candidates;
- 5% Raw-bad with a clearly better local candidate;
- 5% Raw-bad with a clearly better complete candidate.

The first two KEEP-heavy curricula are 95/5 and then 90/10. A K0 all-KEEP
execution baseline must pass before any selector training. Selector-only audits
use a frozen controlled candidate bank and report routing separately from
candidate generation. In particular, an offline Gold-derived positive candidate
may measure takeover logic, but it is never runtime evidence and cannot support
a claim about a deployable candidate generator.
