"""Prospective V7.3 confirmation statistics; no OCR or model fitting."""
import hashlib
import itertools
import math
import random

VERSIONS = ('V7.1', 'V7.2', 'V7.3', 'V7.4')
REQUIRED_CHECKS = ('correspondence_frozen_before_quality', 'lineage_verified',
                   'failure_dispositions_complete', 'resource_limits_satisfied',
                   'package_hashes_unchanged', 'prospective_amendment_verified')


def percentile(sorted_values, p):
    index = (len(sorted_values)-1)*p
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_values[lower]+(index-lower)*(sorted_values[upper]-sorted_values[lower])


def evaluate(rows, frozen_source_groups, checks):
    """Require all frozen 128 dispositions; never substitute a success subset.

    Each row has source_group, disposition_valid (bool), and scores mapping the
    four versions to finite TEDS values. OCR failure may retain a scoreable native
    fallback; unknown reference/quality is None and prevents admission.
    """
    expected = list(frozen_source_groups)
    reasons = []
    if len(expected) != 128 or len(set(expected)) != 128:
        reasons.append('frozen_set_is_not_128_unique_source_documents')
    groups = [r['source_group'] for r in rows]
    if len(rows) != 128 or len(set(groups)) != len(groups) or set(groups) != set(expected):
        reasons.append('missing_duplicate_or_unexpected_source_dispositions')
    for key in REQUIRED_CHECKS:
        if checks.get(key) is not True:
            reasons.append(key+'_not_verified')
    valid = []
    for row in rows:
        scores = row.get('scores', {})
        known = row.get('disposition_valid') is True and all(
            isinstance(scores.get(v), (int, float)) and not isinstance(scores.get(v), bool)
            and math.isfinite(scores[v]) and 0 <= scores[v] <= 1 for v in VERSIONS)
        if known:
            valid.append(row)
    if len(valid) != len(rows):
        reasons.append('unscoreable_or_invalid_dispositions')
    if any(r['scores']['V7.4'] != r['scores']['V7.1'] for r in valid):
        reasons.append('all_keep_negative_comparator_differs_from_native')
    out = {'candidate': 'V7.3', 'primary_controls': ['V7.1', 'V7.2'],
           'required_documents': 128, 'dispositions': len(rows), 'scoreable_documents': len(valid),
           'means': None, 'pairwise': {}, 'checks': checks, 'inconclusive_reasons': reasons,
           'status': 'inconclusive', 'full_benchmark_permitted_by_statistics': False,
           'resamples': 2000, 'seed': 0,
           'bootstrap': 'Python random.Random(0) MT19937;128 draws with replacement; common draws for all pairs; linear2.5/97.5 percentiles',
           'order': 'SHA256(source_group UTF8) ascending; source string as collision tie-break',
           'count_tie_tolerance': 1e-6,
           'full_run_still_requires': 'Exact local comparator revalidation, remaining budget and allowed GPU mapping'}
    # Count all missing values explicitly without inventing a quality mean.
    for a, b in itertools.combinations(VERSIONS, 2):
        paired = [r for r in rows if r.get('disposition_valid') is True and all(
            isinstance(r.get('scores', {}).get(v), (int, float))
            and not isinstance(r['scores'][v], bool) and math.isfinite(r['scores'][v])
            and 0 <= r['scores'][v] <= 1 for v in (a, b))]
        differences = [r['scores'][b]-r['scores'][a] for r in paired]
        out['pairwise'][b+'-'+a] = {
            'improved': sum(x > 1e-6 for x in differences),
            'worsened': sum(x < -1e-6 for x in differences),
            'tied': sum(abs(x) <= 1e-6 for x in differences),
            'unmatchable': len(rows)-len(paired), 'mean_difference': None, 'ci95': None}
    if reasons:
        return out
    rows = sorted(rows, key=lambda r: (hashlib.sha256(r['source_group'].encode()).hexdigest(), r['source_group']))
    out['means'] = {v: math.fsum(r['scores'][v] for r in rows)/128 for v in VERSIONS}
    rng = random.Random(0)
    draws = [[rng.randrange(128) for _ in range(128)] for _ in range(2000)]
    for a, b in itertools.combinations(VERSIONS, 2):
        delta = [r['scores'][b]-r['scores'][a] for r in rows]
        samples = sorted(math.fsum(delta[i] for i in draw)/128 for draw in draws)
        out['pairwise'][b+'-'+a].update(mean_difference=math.fsum(delta)/128,
                                      ci95=[percentile(samples, .025), percentile(samples, .975)])
    out['gain_over_best_control'] = out['means']['V7.3']-max(out['means']['V7.1'], out['means']['V7.2'])
    out['practical_gain_pass'] = out['gain_over_best_control'] >= .005
    out['both_intervals_positive'] = all(out['pairwise']['V7.3-'+v]['ci95'][0] > 0 for v in ('V7.1', 'V7.2'))
    passed = out['practical_gain_pass'] and out['both_intervals_positive']
    out.update(status='passed' if passed else 'failed', full_benchmark_permitted_by_statistics=passed)
    return out
